# Filename: app.py (Full Code with Timing Logs)
import os
import uuid
import time
import traceback
import io
import sys
import logging # Import logging

from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dask.distributed import Client, Future, wait, TimeoutError # Keep TimeoutError if using wait

# --- Configuration ---
UPLOAD_FOLDER = '/opt/comp0239_coursework/webapp/uploads' # Store uploads here
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Ensure UPLOAD_FOLDER exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dask Scheduler location (running on the same host)
DASK_SCHEDULER = '127.0.0.1:8786'

# Specific log file for timing events on the host
HOST_LOG_FILE = '/opt/comp0239_coursework/webapp/host_timing.log'

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limit uploads to 16MB
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # Needed for flash messages

# --- Specific Timing Logger Setup ---
# Configure a logger specifically for timing information
timing_log = logging.getLogger('WebAppTiming')
timing_log.setLevel(logging.INFO) # Set level (INFO or DEBUG)
timing_log.propagate = False # Prevent messages from going to root logger if already configured

# Ensure handlers aren't added multiple times if Flask reloads in debug mode (should be off now)
if not timing_log.handlers:
    # File Handler for timing logs
    file_handler = logging.FileHandler(HOST_LOG_FILE)
    # Include milliseconds in the timestamp format
    file_formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    timing_log.addHandler(file_handler)

    # Optional: Console Handler for timing logs (duplicates output to terminal where Flask runs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(file_formatter)
    timing_log.addHandler(console_handler)


# --- Global Variables ---
# Dask Client (Connect once when app starts)
client = None
try:
    timing_log.info(f"Attempting to connect to Dask scheduler at {DASK_SCHEDULER}...")
    # Increase connection timeout slightly for robustness
    client = Client(DASK_SCHEDULER, timeout="20s")
    timing_log.info(f"Dask client connected: {client}")
    timing_log.info(f"Dask dashboard link: {client.dashboard_link}")
except Exception as e:
    timing_log.critical(f"CRITICAL: Failed to connect to Dask scheduler at {DASK_SCHEDULER}. App cannot process tasks.", exc_info=True)
    # Optionally raise the exception or exit if Dask is essential
    # raise e

# Simple in-memory store for job status and results
# Structure: jobs[job_id] = {'status': 'pending'/'processing'/'complete'/'error',
#                          'filenames': [list of original filenames],
#                          'futures': [list of Dask Future objects] or None (if complete),
#                          'results': [list of captions or error strings],
#                          'task_times': [list of worker durations or None],
#                          'total_tasks': int,
#                          'completed_tasks': int,
#                          'timestamps': {'request_received': float,
#                                         'submission_start': float,
#                                         'submission_end': float,
#                                         'last_results_check': float or None,
#                                         'first_result_received': float or None,
#                                         'all_results_received': float or None}
#                         }
jobs = {}

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Dask Task Functions (These run on Workers) ---
# Note: These functions CANNOT use the 'timing_log' directly as it's configured in the Flask app process.
# They use print statements which *might* appear in worker logs (journalctl).

def load_blip_and_caption(image_bytes):
    """Loads BLIP model and generates caption INSIDE the task function."""
    print("WORKER_INFO: Entering load_blip_and_caption") # Worker log
    model_name = "Salesforce/blip-image-captioning-base"; processor = None; model = None
    task_start_inner = time.time()
    try:
        print(f"WORKER_INFO: Importing transformers/torch/PIL within task...") # Worker log
        import torch; from transformers import BlipProcessor, BlipForConditionalGeneration; from PIL import Image; import io
        print("WORKER_INFO: Imports successful inside task.") # Worker log

        load_start = time.time()
        print(f"WORKER_INFO: Loading BLIP Processor: {model_name}") # Worker log
        processor = BlipProcessor.from_pretrained(model_name)
        print(f"WORKER_INFO: Loading BLIP Model: {model_name}") # Worker log
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        device = torch.device("cpu"); model.to(device); model.eval()
        load_end = time.time()
        print(f"WORKER_INFO: BLIP Processor and Model loaded successfully in {load_end - load_start:.3f}s.") # Worker log

        prep_start = time.time()
        print("WORKER_INFO: Preparing image...") # Worker log
        raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print("WORKER_INFO: Processing image with BlipProcessor...") # Worker log
        inputs = processor(raw_image, return_tensors="pt").to(device)
        prep_end = time.time()
        print(f"WORKER_INFO: Image prep took {prep_end - prep_start:.3f}s.") # Worker log


        gen_start = time.time()
        print("WORKER_INFO: Generating caption (max_length=50)...") # Worker log
        with torch.no_grad(): out = model.generate(**inputs, max_length=50, num_beams=4)
        print("WORKER_INFO: Decoding caption...") # Worker log
        caption = processor.decode(out[0], skip_special_tokens=True)
        gen_end = time.time()
        print(f"WORKER_INFO: Caption generation took {gen_end - gen_start:.3f}s. Result: '{caption}'") # Worker log

        return caption.replace("\n", " ").replace(",", ";").strip()
    except Exception as e:
        print(f"WORKER_ERROR in load_blip_and_caption: {e}\n{traceback.format_exc()}", file=sys.stderr) # Worker log
        return f"ERROR: Caption generation failed ({e.__class__.__name__})"
    finally:
        # Clean up memory
        del processor
        del model
        print(f"WORKER_INFO: load_blip_and_caption finished in {time.time() - task_start_inner:.3f}s total.") # Worker log


def process_image_bytes(image_bytes):
    """
    Wrapper task that takes image bytes, calls captioning, AND returns timing.
    Returns tuple: (caption_or_error, duration_on_worker)
    """
    task_start_time = time.time()
    result = "ERROR: Task failed before processing"
    try:
        if not image_bytes:
            print("WORKER_ERROR: Received empty image bytes.", file=sys.stderr) # Worker log
            result = "ERROR: Empty image data received"
        else:
            result = load_blip_and_caption(image_bytes) # Call the main work function
    except Exception as e:
        result = f"ERROR: Processing wrapper failed ({e.__class__.__name__})"
        print(f"WORKER_ERROR in process_image_bytes wrapper: {e}\n{traceback.format_exc()}", file=sys.stderr) # Worker log
    finally:
        task_end_time = time.time()
        duration = task_end_time - task_start_time
        print(f"WORKER_INFO: process_image_bytes task duration: {duration:.4f}s") # Worker log
        # Return both the original result and the duration
        return (result, duration)


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Display the upload form."""
    timing_log.info("Request for index page.")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads, submit tasks to Dask, log timing."""
    request_received_time = time.time()
    timing_log.info(f"Received upload request.")

    if client is None:
         timing_log.error("Upload attempt failed: Dask client not connected.")
         flash('Dask connection error. Cannot process uploads.', 'error')
         return redirect(url_for('index'))

    if 'images' not in request.files:
        timing_log.warning("Upload attempt failed: No 'images' file part in request.")
        flash('No file part in request.', 'error')
        return redirect(url_for('index'))

    files = request.files.getlist('images')

    if not files or files[0].filename == '':
        timing_log.warning("Upload attempt failed: No files selected.")
        flash('No selected files.', 'warning')
        return redirect(url_for('index'))

    submitted_futures = []; original_filenames = []; job_id = str(uuid.uuid4())
    processed_count = 0
    submission_start_time = time.time()
    timing_log.info(f"[Job {job_id}] Starting file processing and task submission.")

    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_read_start = time.time()
                image_bytes = file.read()
                file_read_end = time.time()
                read_duration_ms = (file_read_end-file_read_start)*1000
                if not image_bytes:
                    timing_log.warning(f"[Job {job_id}] Skipping empty file: {filename}")
                    flash(f'Skipping empty file: {filename}', 'warning')
                    continue

                timing_log.debug(f"[Job {job_id}] Read file '{filename}' ({len(image_bytes)} bytes) in {read_duration_ms:.2f}ms.")

                submit_start = time.time()
                # Submit the task that returns (result, duration)
                future = client.submit(process_image_bytes, image_bytes, pure=False)
                submit_end = time.time()
                submit_duration_ms = (submit_end-submit_start)*1000

                submitted_futures.append(future)
                original_filenames.append(filename)
                processed_count += 1
                timing_log.info(f"[Job {job_id}] Submitted task for '{filename}'. Submit call took {submit_duration_ms:.2f}ms.")

            except Exception as e:
                timing_log.error(f"[Job {job_id}] Error processing/submitting file {file.filename}: {e}", exc_info=True)
                flash(f'Error processing file {file.filename}: {e}', 'error')
        elif file and file.filename != '':
            timing_log.warning(f"[Job {job_id}] File type not allowed: {file.filename}")
            flash(f'File type not allowed: {file.filename}', 'warning')

    submission_end_time = time.time()

    if not submitted_futures:
         timing_log.warning("Upload processed, but no valid files were submitted.")
         flash('No valid image files were processed.', 'error')
         return redirect(url_for('index'))

    # Store job information WITH timestamps
    jobs[job_id] = {
        'status': 'processing',
        'filenames': original_filenames,
        'futures': submitted_futures,
        'results': [None] * len(submitted_futures),
        'task_times': [None] * len(submitted_futures), # Store worker times
        'total_tasks': len(submitted_futures),
        'completed_tasks': 0,
        'timestamps': { # Store timestamps for this job
             'request_received': request_received_time,
             'submission_start': submission_start_time,
             'submission_end': submission_end_time,
             'last_results_check': None,
             'first_result_received': None,
             'all_results_received': None
             }
    }
    timing_log.info(f"[Job {job_id}] Finished submission. Submitted {processed_count} tasks. Total submission phase time: {(submission_end_time-submission_start_time)*1000:.2f}ms")

    flash(f'Successfully submitted {processed_count} image(s). Job ID: {job_id}', 'success')
    return redirect(url_for('show_results', job_id=job_id))


@app.route('/results/<job_id>', methods=['GET'])
def show_results(job_id):
    """Display status, results, and log timing for result checks."""
    results_check_start_time = time.time()
    job_info = jobs.get(job_id)
    if not job_info:
        timing_log.warning(f"Request for non-existent Job ID: {job_id}")
        flash(f'Job ID {job_id} not found.', 'error');
        return redirect(url_for('index'))

    timing_log.info(f"[Job {job_id}] Request for results page. Current status: {job_info.get('status', 'unknown')}")
    job_info['timestamps']['last_results_check'] = results_check_start_time

    progress = 0
    num_done = 0
    all_accounted_for = True # Assume complete until proven otherwise

    # Check status of futures if job is still processing or has futures list
    if job_info.get('futures'):
        futures_to_check = job_info['futures']
        # Quick non-blocking check first
        done_set, _ = wait(futures_to_check, timeout=0) # timeout=0 is non-blocking

        for i, future in enumerate(futures_to_check):
            if job_info['results'][i] is not None: # Already processed
                num_done += 1
                continue

            # Try to get result only if wait indicated it's done
            if future in done_set:
                result_fetch_start = time.time()
                try:
                    # task_result_tuple = (caption_or_error, duration_on_worker)
                    task_result_tuple = future.result(timeout=0.1) # Use short timeout
                    job_info['results'][i] = task_result_tuple[0] # Store caption/error
                    job_info['task_times'][i] = task_result_tuple[1] # Store worker duration
                    num_done += 1
                    if job_info['timestamps']['first_result_received'] is None:
                        job_info['timestamps']['first_result_received'] = time.time()
                    timing_log.info(f"[Job {job_id}] Received result for task {i} ('{job_info['filenames'][i]}'). Worker time: {task_result_tuple[1]:.3f}s. Fetch took {(time.time()-result_fetch_start)*1000:.2f}ms.")

                except TimeoutError:
                    timing_log.debug(f"[Job {job_id}] Timeout getting result for done future {future.key}, will retry.")
                    all_accounted_for = False # Not really done if we can't get result yet
                except Exception as e:
                    # Handle case where future is done but retrieving result/exception fails
                    job_info['results'][i] = f"ERROR: Failed retrieval - {e}"
                    job_info['task_times'][i] = -1 # Indicate error
                    num_done += 1 # Count as 'done' for progress tracking
                    if job_info['timestamps']['first_result_received'] is None:
                         job_info['timestamps']['first_result_received'] = time.time()
                    timing_log.error(f"[Job {job_id}] Error getting result/exception for done future {future.key}, task {i}: {e}. Fetch took {(time.time()-result_fetch_start)*1000:.2f}ms.", exc_info=True)
            else:
                # Future not in done_set, implies still pending or running
                all_accounted_for = False

        # Update overall job status based on checks
        if all_accounted_for and num_done == job_info['total_tasks']:
            if job_info['status'] != 'complete': # Log only once
                 job_info['status'] = 'complete'
                 job_info['timestamps']['all_results_received'] = time.time()
                 timing_log.info(f"[Job {job_id}] All {num_done} results received. Status set to complete. Total processing time (request to last result): {(job_info['timestamps']['all_results_received'] - job_info['timestamps']['request_received']):.2f}s")
                 if 'futures' in job_info: del job_info['futures'] # Cleanup futures list
        elif job_info['status'] != 'error': # Don't override error status
             job_info['status'] = 'processing'


    # Recalculate progress based on num_done (tasks with results/errors stored)
    job_info['completed_tasks'] = num_done
    if job_info['total_tasks'] > 0: progress = int((num_done / job_info['total_tasks']) * 100)
    # Ensure progress is 100 if status is complete
    if job_info['status'] == 'complete': progress = 100

    # Combine results for display
    results_display = list(zip(job_info['filenames'], job_info['results'])) # Contains (filename, caption/error) tuples
    results_check_end_time = time.time()
    timing_log.info(f"[Job {job_id}] Results page generated. Check took {(results_check_end_time - results_check_start_time)*1000:.2f}ms. Progress: {progress}%. Status: {job_info['status']}")

    return render_template('results.html',
                           job_id=job_id,
                           status=job_info['status'],
                           results=results_display,
                           progress=progress)


# --- Run the App ---
if __name__ == '__main__':
    if client is None:
        timing_log.critical("Dask client is not connected. Flask app cannot start processing.")
        # Depending on requirements, you might exit or run a limited version
        # For this coursework, exiting is likely appropriate
        sys.exit("Exiting: Dask client connection failed.")
    timing_log.info(f"Starting Flask app on http://0.0.0.0:5000, logging timings to {HOST_LOG_FILE}")
    # Set debug=False for stability during testing/production
    app.run(host='0.0.0.0', port=5000, debug=False)