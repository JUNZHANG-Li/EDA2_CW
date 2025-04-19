import os
import uuid
import time
import traceback
import io
import logging # <<< Import logging
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dask.distributed import Client, Future, wait, TimeoutError

# --- Configuration ---
WEBAPP_BASE_DIR = "/opt/comp0239_coursework/webapp" # Define base for logs/uploads
UPLOAD_FOLDER = os.path.join(WEBAPP_BASE_DIR, 'uploads')
LOG_FILE = os.path.join(WEBAPP_BASE_DIR, 'user_jobs.log') # <<< Define log file path
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DASK_SCHEDULER = '127.0.0.1:8786'

# --- Job Logger Setup ---
# Configure a specific logger for job tracking
job_logger = logging.getLogger('JobLogger')
job_logger.setLevel(logging.INFO)
# Prevent job logs from propagating to the root Flask logger if desired
# job_logger.propagate = False
try:
    # Ensure directory exists (though playbook should handle it)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    # Create file handler
    file_handler = logging.FileHandler(LOG_FILE)
    # Create formatter - include Job ID in the message itself
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add handler to the logger
    if not job_logger.handlers: # Add handler only once
         job_logger.addHandler(file_handler)
except Exception as log_e:
    print(f"CRITICAL: Failed to configure job logger: {log_e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# --- Dask Client ---
client = None
try:
    job_logger.info("Attempting to connect to Dask scheduler...")
    client = Client(DASK_SCHEDULER, timeout="10s")
    job_logger.info(f"Dask client connected: {client}")
    job_logger.info(f"Dask dashboard link: {client.dashboard_link}")
except Exception as e:
    job_logger.critical(f"Failed to connect to Dask scheduler at {DASK_SCHEDULER}. App may not function.", exc_info=True)
    # Keep client as None

# --- Job Store ---
jobs = {}

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Dask Task Functions ---
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
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads, submit tasks to Dask."""
    # --- LOGGING: Request received ---
    job_logger.info(f"Received upload request from {request.remote_addr}")

    if client is None:
        job_logger.error("Upload failed: Dask client not connected.")
        flash('Dask connection error. Cannot process uploads.', 'error')
        return redirect(url_for('index'))

    if 'images' not in request.files:
        job_logger.warning("Upload failed: No 'images' file part in request.")
        flash('No file part in request.', 'error')
        return redirect(url_for('index'))

    files = request.files.getlist('images')
    submitted_futures = []; original_filenames = []; job_id = str(uuid.uuid4()) # Generate unique job ID first
    processed_count = 0

    # --- LOGGING: Job ID Generated ---
    job_logger.info(f"JID:{job_id} - Generated Job ID.")

    if not files or files[0].filename == '':
         job_logger.warning(f"JID:{job_id} - Upload failed: No files selected.")
         flash('No selected files.', 'warning')
         return redirect(url_for('index'))

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) # Secure filename early
            try:
                image_bytes = file.read()
                if not image_bytes:
                     job_logger.warning(f"JID:{job_id} - Skipping empty file: {filename}")
                     flash(f'Skipping empty file: {filename}', 'warning')
                     continue

                # --- LOGGING: Submitting task ---
                job_logger.debug(f"JID:{job_id} - Submitting task for {filename}...") # Debug level for individual tasks
                future = client.submit(process_image_bytes, image_bytes, pure=False)
                submitted_futures.append(future)
                original_filenames.append(filename)
                processed_count += 1

            except Exception as e:
                job_logger.error(f"JID:{job_id} - Error processing/submitting file {filename}: {e}", exc_info=True)
                flash(f'Error processing file {filename}: {e}', 'error')
                # Continue with other files

        elif file and file.filename != '':
            job_logger.warning(f"JID:{job_id} - File type not allowed: {file.filename}")
            flash(f'File type not allowed: {file.filename}', 'warning')

    if not submitted_futures:
         job_logger.error(f"JID:{job_id} - No valid image files were submitted.")
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
    # --- LOGGING: Job Submission Summary ---
    job_logger.info(f"JID:{job_id} - Submitted {processed_count} tasks to Dask cluster.")

    flash(f'Successfully submitted {processed_count} image(s) for captioning. Job ID: {job_id}', 'success')
    return redirect(url_for('show_results', job_id=job_id))


@app.route('/results/<job_id>', methods=['GET'])
def show_results(job_id):
    """Display status and results for a given job ID."""
    # --- LOGGING: Results page requested ---
    job_logger.info(f"JID:{job_id} - Request to view results.")

    job_info = jobs.get(job_id)
    if not job_info:
        job_logger.error(f"JID:{job_id} - Job ID not found in store.")
        flash(f'Job ID {job_id} not found.', 'error')
        return redirect(url_for('index'))

    progress = 0
    num_done = 0
    all_accounted_for = True
    previous_status = job_info['status'] # Store previous status

    if job_info.get('futures'):
        futures_to_check = job_info['futures']
        # --- LOGGING: Checking future status ---
        job_logger.debug(f"JID:{job_id} - Checking status of {len(futures_to_check)} futures.")
        for i, future in enumerate(futures_to_check):
            if job_info['results'][i] is not None:
                num_done += 1
                continue

            if future.done():
                try:
                    if future.status == 'finished':
                       job_info['results'][i] = future.result(timeout=1)
                       num_done += 1
                    elif future.status == 'error':
                        try:
                            exc = future.exception(timeout=1) # Get exception
                            job_info['results'][i] = f"ERROR: Task failed - {exc}"
                            # --- LOGGING: Task Error ---
                            job_logger.error(f"JID:{job_id} - Task for file {job_info['filenames'][i]} failed: {exc}")
                        except Exception as e_inner:
                            job_info['results'][i] = f"ERROR: Failed to get task exception - {e_inner}"
                            job_logger.error(f"JID:{job_id} - Failed to get exception for failed task {future.key}: {e_inner}")
                        num_done += 1
                    else: all_accounted_for = False
                except TimeoutError:
                    job_logger.warning(f"JID:{job_id} - Timeout getting result/exception for done future {future.key}")
                    all_accounted_for = False
                except Exception as e:
                     job_logger.error(f"JID:{job_id} - Error getting result/exception for done future {future.key}: {e}", exc_info=True)
                     job_info['results'][i] = f"ERROR: Failed to retrieve result/exception - {e}"
                     num_done += 1
            else: all_accounted_for = False

        job_info['completed_tasks'] = num_done
        if all_accounted_for:
            job_info['status'] = 'complete'
            # --- LOGGING: Job Completed ---
            if previous_status != 'complete': # Log only on transition
                 job_logger.info(f"JID:{job_id} - Job marked as complete ({num_done}/{job_info['total_tasks']} tasks finished).")
            if 'futures' in job_info: del job_info['futures'] # Cleanup
        else:
             job_info['status'] = 'processing'

    # Recalculate progress
    if job_info['total_tasks'] > 0:
        progress = int((job_info.get('completed_tasks', 0) / job_info['total_tasks']) * 100)
    if job_info['status'] == 'complete': progress = 100

    # --- LOGGING: Results page status update ---
    job_logger.debug(f"JID:{job_id} - Displaying results. Status: {job_info['status']}, Progress: {progress}%")

    results_display = list(zip(job_info['filenames'], job_info['results']))

    return render_template('results.html',
                           job_id=job_id,
                           status=job_info['status'],
                           results=results_display,
                           progress=progress)

# --- Run the App ---
if __name__ == '__main__':
    if client is None:
        job_logger.critical("Flask app exiting: Dask client connection failed on startup.")
        sys.exit(1)
    job_logger.info("Starting Flask app on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
