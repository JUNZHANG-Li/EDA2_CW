# Filename: app.py (with Detailed Timing Logs)
import os
import uuid
import time # Import time module
import traceback
import io
import logging # Import logging module
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dask.distributed import Client, Future, wait, TimeoutError # Removed FIRST_COMPLETED

# --- Configuration ---
UPLOAD_FOLDER = '/opt/comp0239_coursework/webapp/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DASK_SCHEDULER = '127.0.0.1:8786'

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# --- Setup Host-Side Logging ---
# Reuse the logger from run_capacity_test or setup similarly
# Log to a different file to separate webapp logs from capacity test logs
WEBAPP_LOG_FILE = '/opt/comp0239_coursework/output/webapp.log' # New log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [WebApp] - %(message)s', # Added [WebApp] tag
    handlers=[
        logging.FileHandler(WEBAPP_LOG_FILE),
        logging.StreamHandler(sys.stdout) # Also log to console where Flask runs
    ]
)
log = logging.getLogger('WebAppLogger') # Use a specific logger instance

# --- Dask Client ---
client = None
try:
    log.info(f"Connecting to Dask scheduler at {DASK_SCHEDULER}...")
    t_connect_start = time.time()
    client = Client(DASK_SCHEDULER, timeout="10s")
    t_connect_end = time.time()
    log.info(f"Dask client connected: {client} (took {t_connect_end - t_connect_start:.3f}s)")
    log.info(f"Dashboard Link: {client.dashboard_link}")
except Exception as e:
    log.critical(f"Failed to connect to Dask scheduler at {DASK_SCHEDULER}. App may not function.", exc_info=True)

# --- Job Store ---
jobs = {}

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Dask Task Functions ---
# (load_blip_and_caption and process_image_bytes remain unchanged)
# These run on workers, their logs go to worker journals (or stdout/stderr)
def load_blip_and_caption(image_bytes):
    print("WORKER_INFO: Entering load_blip_and_caption")
    model_name = "Salesforce/blip-image-captioning-base"; processor = None; model = None
    try:
        print(f"WORKER_INFO: Importing transformers/torch/PIL within task...")
        import torch; from transformers import BlipProcessor, BlipForConditionalGeneration; from PIL import Image; import io
        print("WORKER_INFO: Imports successful inside task.")
        print(f"WORKER_INFO: Loading BLIP Processor: {model_name}"); processor = BlipProcessor.from_pretrained(model_name)
        print(f"WORKER_INFO: Loading BLIP Model: {model_name}"); model = BlipForConditionalGeneration.from_pretrained(model_name)
        device = torch.device("cpu"); model.to(device); model.eval()
        print("WORKER_INFO: BLIP Processor and Model loaded successfully.")
        print("WORKER_INFO: Preparing image..."); raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print("WORKER_INFO: Processing image with BlipProcessor..."); inputs = processor(raw_image, return_tensors="pt").to(device)
        print("WORKER_INFO: Generating caption (max_length=50)...")
        with torch.no_grad(): out = model.generate(**inputs, max_length=50, num_beams=4)
        print("WORKER_INFO: Decoding caption..."); caption = processor.decode(out[0], skip_special_tokens=True)
        print(f"WORKER_INFO: Caption generated successfully: '{caption}'")
        return caption.replace("\n", " ").replace(",", ";").strip()
    except Exception as e: print(f"WORKER_ERROR in load_blip_and_caption: {e}\n{traceback.format_exc()}", file=sys.stderr); return f"ERROR: Caption generation failed ({e.__class__.__name__})"
    finally: del processor; del model

def process_image_bytes(image_bytes):
    t_start = time.time()
    if not image_bytes: print("WORKER_ERROR: Received empty image bytes.", file=sys.stderr); return "ERROR: Empty image data received"
    try: return load_blip_and_caption(image_bytes)
    except Exception as e: print(f"WORKER_ERROR in process_image_bytes wrapper: {e}\n{traceback.format_exc()}", file=sys.stderr); return f"ERROR: Processing wrapper failed ({e.__class__.__name__})"


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Display the upload form."""
    log.info(f"Request received for index page from {request.remote_addr}")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads, submit tasks to Dask."""
    t_upload_start = time.time()
    log.info(f"Upload request received from {request.remote_addr}")

    if client is None:
         log.error("Dask client not connected. Aborting upload.")
         flash('Dask connection error. Cannot process uploads.', 'error')
         return redirect(url_for('index'))

    if 'images' not in request.files:
        log.warning("No 'images' file part in request.")
        flash('No file part in request.', 'error')
        return redirect(url_for('index'))

    files = request.files.getlist('images')
    submitted_futures = []; original_filenames = []; job_id = str(uuid.uuid4())
    processed_count = 0; skipped_count = 0; error_count = 0
    t_before_loop = time.time()

    if not files or files[0].filename == '':
         log.warning("No files selected in upload request.")
         flash('No selected files.', 'warning')
         return redirect(url_for('index'))

    log.info(f"Job {job_id}: Processing {len(files)} file(s) from upload.")
    for file in files:
        t_file_start = time.time()
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) # Basic security outside try
            try:
                log.debug(f"Job {job_id}: Reading bytes for {filename}...")
                image_bytes = file.read()
                read_time = time.time() - t_file_start
                log.debug(f"Job {job_id}: Read {len(image_bytes)} bytes for {filename} (took {read_time:.3f}s).")

                if not image_bytes:
                     log.warning(f"Job {job_id}: Skipping empty file: {filename}")
                     flash(f'Skipping empty file: {filename}', 'warning')
                     skipped_count += 1
                     continue

                # Submit task to Dask using image bytes
                log.debug(f"Job {job_id}: Submitting task for {filename}...")
                t_submit_start = time.time()
                future = client.submit(process_image_bytes, image_bytes, pure=False)
                t_submit_end = time.time()
                log.debug(f"Job {job_id}: Task for {filename} submitted (took {t_submit_end - t_submit_start:.3f}s). Future key: {future.key}")

                submitted_futures.append(future)
                original_filenames.append(filename)
                processed_count += 1

            except Exception as e:
                log.error(f"Job {job_id}: Error processing/submitting file {filename}: {e}", exc_info=True)
                flash(f'Error processing file {filename}: {e}', 'error')
                error_count += 1

        elif file and file.filename != '':
            log.warning(f"Job {job_id}: File type not allowed: {file.filename}")
            flash(f'File type not allowed: {file.filename}', 'warning')
            skipped_count += 1
        else:
             # Should not happen with check above, but defensive
             log.warning(f"Job {job_id}: Encountered invalid file object in list.")
             skipped_count += 1

    t_after_loop = time.time()
    log.info(f"Job {job_id}: File processing loop finished (took {t_after_loop - t_before_loop:.3f}s). Submitted: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")

    if not submitted_futures:
         log.error(f"Job {job_id}: No valid image files were submitted.")
         flash('No valid image files were processed.', 'error')
         return redirect(url_for('index'))

    # Store job information
    jobs[job_id] = {
        'status': 'processing',
        'filenames': original_filenames,
        'futures': submitted_futures,
        'results': [None] * len(submitted_futures),
        'total_tasks': len(submitted_futures),
        'completed_tasks': 0
    }
    log.info(f"Job {job_id}: Stored job info. Redirecting to results page.")

    flash(f'Successfully submitted {processed_count} image(s) for captioning. Job ID: {job_id}', 'success')
    t_upload_end = time.time()
    log.info(f"Job {job_id}: Upload request finished (total time: {t_upload_end - t_upload_start:.3f}s).")
    return redirect(url_for('show_results', job_id=job_id))


@app.route('/results/<job_id>', methods=['GET'])
def show_results(job_id):
    """Display status and results for a given job ID."""
    t_results_start = time.time()
    log.info(f"Request received for results page for job {job_id} from {request.remote_addr}")

    job_info = jobs.get(job_id)
    if not job_info:
        log.warning(f"Job ID {job_id} not found for results request.")
        flash(f'Job ID {job_id} not found.', 'error')
        return redirect(url_for('index'))

    progress = 0
    num_done = 0
    all_accounted_for = True # Assume complete unless proven otherwise

    # Check status of futures if job is still processing or has futures listed
    if job_info.get('futures'):
        futures_to_check = job_info['futures']
        log.debug(f"Job {job_id}: Checking status of {len(futures_to_check)} futures.")
        t_check_start = time.time()
        try:
            # Use wait with timeout=0 for a quick check
            # No need for FIRST_COMPLETED if just checking status
            done_set, _ = wait(futures_to_check, timeout=0) # Non-blocking check
            log.debug(f"Job {job_id}: {len(done_set)} futures reported done by wait().")

            for i, future in enumerate(futures_to_check):
                if job_info['results'][i] is not None:
                    num_done += 1 # Already have result
                    continue

                # Check if the future is in the set returned by wait()
                if future in done_set:
                    t_get_result_start = time.time()
                    if future.status == 'finished':
                        try:
                            job_info['results'][i] = future.result(timeout=1) # Short timeout to retrieve
                            log.debug(f"Job {job_id}: Retrieved result for future {future.key} (index {i}).")
                        except TimeoutError:
                             log.warning(f"Job {job_id}: Timeout getting result for done future {future.key}. Will retry.")
                             all_accounted_for = False # Not truly done yet
                             continue # Skip incrementing num_done
                        except Exception as e:
                            log.error(f"Job {job_id}: Error getting result for finished future {future.key}: {e}", exc_info=True)
                            job_info['results'][i] = f"ERROR: Failed to get result - {e}"
                    elif future.status == 'error':
                        try:
                            job_info['results'][i] = f"ERROR: Task failed - {future.exception(timeout=1)}" # Get exception
                            log.warning(f"Job {job_id}: Retrieved error for future {future.key} (index {i}).")
                        except TimeoutError:
                             log.warning(f"Job {job_id}: Timeout getting exception for error future {future.key}. Will retry.")
                             all_accounted_for = False # Not truly done yet
                             continue # Skip incrementing num_done
                        except Exception as e:
                            log.error(f"Job {job_id}: Error getting exception for error future {future.key}: {e}", exc_info=True)
                            job_info['results'][i] = f"ERROR: Failed to get task exception - {e}"
                    else: # Should not happen if future is in done_set, but defensive
                        log.warning(f"Job {job_id}: Future {future.key} in done_set but status is {future.status}?")
                        all_accounted_for = False # Not truly done
                        continue # Skip incrementing num_done

                    t_get_result_end = time.time()
                    log.debug(f"Job {job_id}: Result/Error retrieval took {t_get_result_end - t_get_result_start:.3f}s")
                    num_done += 1 # Count tasks where we stored a result/error

                else:
                    # Future was not in done_set, so still pending/running
                    all_accounted_for = False

            t_check_end = time.time()
            log.debug(f"Job {job_id}: Future status check loop took {t_check_end - t_check_start:.3f}s.")

            # Update overall job status
            if all_accounted_for:
                log.info(f"Job {job_id}: All tasks accounted for. Setting status to complete.")
                job_info['status'] = 'complete'
                if 'futures' in job_info: del job_info['futures'] # Cleanup futures list
            else:
                log.debug(f"Job {job_id}: Still processing. Num done: {num_done}/{job_info['total_tasks']}")
                job_info['status'] = 'processing'

        except Exception as e:
             log.error(f"Error during future status check for job {job_id}: {e}", exc_info=True)
             # Keep status as processing or set to error? For now, keep processing.
             # job_info['status'] = 'error'

    # Recalculate progress based on num_done
    job_info['completed_tasks'] = num_done
    if job_info['total_tasks'] > 0:
        progress = int((num_done / job_info['total_tasks']) * 100)
    if job_info['status'] == 'complete': progress = 100 # Ensure 100% if complete

    results_display = list(zip(job_info['filenames'], job_info['results']))

    t_results_end = time.time()
    log.info(f"Results page render for job {job_id} finished (total time: {t_results_end - t_results_start:.3f}s). Status: {job_info['status']}, Progress: {progress}%")
    return render_template('results.html',
                           job_id=job_id,
                           status=job_info['status'],
                           results=results_display,
                           progress=progress)

# --- Run the App ---
if __name__ == '__main__':
    if client is None:
        log.critical("Cannot start Flask app without Dask connection.")
        sys.exit(1)
    log.info("Starting Flask app on http://0.0.0.0:5000")
    # Turn off reloader and debug for production/service use
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)