# Filename: app.py (Radically Simplified Actor for Debugging)
import os
import sys # Import sys for exit check below
print("DEBUG: Script Started - Top Level") # <<< ADDED

import uuid
import time
import traceback
import io
import logging # <<< Import logging
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
# Dask libraries
from dask.distributed import Client, Future, wait, TimeoutError, Actor # Import Actor

print("DEBUG: Imports Seem OK") # <<< ADDED

# --- Configuration ---
WEBAPP_BASE_DIR = "/opt/comp0239_coursework/webapp" # Define base for logs/uploads
UPLOAD_FOLDER = os.path.join(WEBAPP_BASE_DIR, 'uploads')
LOG_FILE = os.path.join(WEBAPP_BASE_DIR, 'user_jobs.log') # <<< Define log file path
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DASK_SCHEDULER = '127.0.0.1:8786'
print("DEBUG: Config Parsed") # <<< ADDED

# --- Job Logger Setup ---
job_logger = logging.getLogger('JobLogger')
job_logger.setLevel(logging.INFO)
try:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    if not job_logger.handlers: job_logger.addHandler(file_handler)
except Exception as log_e:
    print(f"CRITICAL: Failed to configure job logger: {log_e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
print("DEBUG: Logger Setup Done") # <<< ADDED

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
print("DEBUG: Flask App Initialized") # <<< ADDED

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Dask Actor Definition ---  <<< SIMPLIFIED
class BlipCaptionActor(Actor):
    # --- SIMPLIFIED __init__ ---
    def __init__(self):
        # ONLY print statements, no heavy imports or model loading
        print("ACTOR_DEBUG: Entering BlipCaptionActor.__init__ (SIMPLIFIED)")
        self.init_flag = True # Simple flag
        self._initialized = True # Set immediately for testing check_ready
        print("ACTOR_DEBUG: Exiting BlipCaptionActor.__init__ (SIMPLIFIED)")
    # --- END SIMPLIFIED __init__ ---

    def check_ready(self):
        print("ACTOR_INFO: check_ready() called.")
        # Check the simple flag
        ready = hasattr(self, 'init_flag') and self.init_flag and \
                hasattr(self, '_initialized') and self._initialized
        print(f"ACTOR_INFO: check_ready() returning: {ready}")
        return ready

    def caption_image(self, image_bytes):
        # --- SIMPLIFIED caption_image ---
        print("ACTOR_INFO: Received caption_image request.")
        if not hasattr(self, '_initialized') or not self._initialized:
             print("ACTOR_ERROR: Actor not initialized correctly!", file=sys.stderr)
             return "ERROR: Actor not initialized correctly"
        # Just return a dummy caption for testing
        # Make sure time is imported if you use it here
        import time
        dummy_caption = f"Processed image of size {len(image_bytes)} bytes at {time.time()}"
        print(f"ACTOR_INFO: Returning dummy caption: {dummy_caption}")
        return dummy_caption
        # --- END SIMPLIFIED caption_image ---


# --- Dask Client and Actor Future ---
client = None
blip_actor_future = None
actor_initialized_ok = False # Flag remains useful
print("DEBUG: Attempting Dask Connection...")
try:
    job_logger.info("Attempting to connect to Dask scheduler...")
    client = Client(DASK_SCHEDULER, timeout="10s")
    job_logger.info(f"Dask client connected: {client}")
    job_logger.info(f"Dask dashboard link: {client.dashboard_link}")
    print(f"DEBUG: Dask Client Connected: {client}")

    job_logger.info("Submitting BlipCaptionActor to Dask cluster...")
    # Submit the (now simplified) Actor class
    blip_actor_future = client.submit(BlipCaptionActor, actor=True)
    job_logger.info(f"BlipCaptionActor submission task created. Future: {blip_actor_future}")
    print(f"DEBUG: Blip Actor Submitted. Future: {blip_actor_future}")

    # --- Force Initialization and Check ---
    job_logger.info("Waiting for Actor to initialize...")
    print("DEBUG: Waiting for Actor initialization...")
    try:
        wait(blip_actor_future, timeout=30) # Shorter timeout should be fine now

        if blip_actor_future.status == 'finished':
             # --- Explicitly call check_ready method via client.submit ---
             print("DEBUG: Actor future finished. Submitting check_ready...")
             # Submit the CLASS METHOD, passing the actor future as the implicit 'self'
             check_future = client.submit(BlipCaptionActor.check_ready, blip_actor_future)
             is_ready = check_future.result(timeout=10) # Wait for check result
             print(f"DEBUG: check_ready() returned: {is_ready}")
             if is_ready:
                 actor_initialized_ok = True
                 job_logger.info("BlipCaptionActor initialized and checked successfully.")
                 print("DEBUG: Actor initialization and check successful.")
             else:
                  job_logger.critical("Blip Actor check_ready() returned False. Initialization likely incomplete or failed check.")
                  print("DEBUG: Actor check_ready() returned False.")
             # --- END Check ---
        elif blip_actor_future.status == 'error': # Handle init failure reported by wait()
            actor_exception = blip_actor_future.exception()
            job_logger.critical(f"Blip Actor FAILED initialization: {actor_exception}", exc_info=actor_exception)
            print(f"DEBUG: Actor FAILED initialization: {actor_exception}")
        else: # Should not happen if wait() returns without timeout, but handle defensively
             job_logger.warning(f"Blip Actor initialization status unclear after wait: {blip_actor_future.status}")
             print(f"DEBUG: Actor initialization status unclear: {blip_actor_future.status}")

    except TimeoutError:
         job_logger.critical("Timeout waiting for Blip Actor to initialize.")
         print("DEBUG: Timeout waiting for Actor initialization.")
    except Exception as init_e:
         job_logger.critical(f"Exception during Blip Actor initialization wait/check: {init_e}", exc_info=True)
         print(f"DEBUG: Exception during Actor initialization wait/check: {init_e}")
    # --- END Initialization Check ---

except Exception as e:
    job_logger.critical(f"Failed to connect to Dask or submit Actor: {e}", exc_info=True)
    print(f"DEBUG: EXCEPTION during Dask setup: {e}")
    client = None
    blip_actor_future = None


# --- Job Store ---
jobs = {}


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    job_logger.info(f"Received upload request from {request.remote_addr}")
    # Check Dask connection and Actor Status
    if client is None: job_logger.error("Upload failed: Dask client not connected."); flash('Service error: Dask client unavailable.', 'error'); return redirect(url_for('index'))
    # Use the flag set during startup
    if not actor_initialized_ok: job_logger.error(f"Upload failed: Blip Actor not initialized correctly."); flash('Service error: Captioning actor not ready.', 'error'); return redirect(url_for('index'))

    if 'images' not in request.files: job_logger.warning("Upload failed: No 'images' file part."); flash('No file part in request.', 'error'); return redirect(url_for('index'))

    files = request.files.getlist('images'); submitted_futures = []; original_filenames = []; job_id = str(uuid.uuid4()); processed_count = 0
    job_logger.info(f"JID:{job_id} - Generated Job ID.")

    if not files or files[0].filename == '': job_logger.warning(f"JID:{job_id} - Upload failed: No files selected."); flash('No selected files.', 'warning'); return redirect(url_for('index'))

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                image_bytes = file.read()
                if not image_bytes: job_logger.warning(f"JID:{job_id} - Skipping empty file: {filename}"); flash(f'Skipping empty file: {filename}', 'warning'); continue
                job_logger.debug(f"JID:{job_id} - Submitting task for {filename} to Actor...")
                # Use attribute access on future syntax (reverted previously)
                # Let's stick to the Class.method, actor_future syntax for consistency now
                future = client.submit(BlipCaptionActor.caption_image, blip_actor_future, image_bytes, pure=False)
                submitted_futures.append(future); original_filenames.append(filename); processed_count += 1
            except Exception as e: job_logger.error(f"JID:{job_id} - Error processing/submitting file {filename}: {e}", exc_info=True); flash(f'Error processing file {filename}: {e}', 'error')
        elif file and file.filename != '': job_logger.warning(f"JID:{job_id} - File type not allowed: {file.filename}"); flash(f'File type not allowed: {file.filename}', 'warning')

    if not submitted_futures: job_logger.error(f"JID:{job_id} - No valid image files were submitted."); flash('No valid image files were processed.', 'error'); return redirect(url_for('index'))

    jobs[job_id] = {'status': 'processing', 'filenames': original_filenames, 'futures': submitted_futures, 'results': [None] * len(submitted_futures), 'total_tasks': len(submitted_futures), 'completed_tasks': 0}
    job_logger.info(f"JID:{job_id} - Submitted {processed_count} tasks to Actor.")
    flash(f'Successfully submitted {processed_count} image(s) for captioning. Job ID: {job_id}', 'success')
    return redirect(url_for('show_results', job_id=job_id))

@app.route('/results/<job_id>', methods=['GET'])
def show_results(job_id):
    job_logger.info(f"JID:{job_id} - Request to view results.")
    job_info = jobs.get(job_id)
    if not job_info: job_logger.error(f"JID:{job_id} - Job ID not found."); flash(f'Job ID {job_id} not found.', 'error'); return redirect(url_for('index'))

    progress = 0; num_done = 0; all_accounted_for = True; previous_status = job_info['status']

    if job_info.get('futures'):
        futures_to_check = job_info['futures']
        job_logger.debug(f"JID:{job_id} - Checking status of {len(futures_to_check)} futures.")
        try: done_set, _ = wait(futures_to_check, timeout=0) # Non-blocking check
        except Exception as e: job_logger.error(f"JID:{job_id} - Error during wait() check: {e}", exc_info=True); done_set = set()

        for i, future in enumerate(futures_to_check):
            if job_info['results'][i] is not None: num_done += 1; continue
            if future in done_set:
                try:
                    status = future.status
                    if status == 'finished': job_info['results'][i] = future.result(timeout=1); num_done += 1
                    elif status == 'error':
                        try: exc = future.exception(timeout=1); job_info['results'][i] = f"ERROR: Task failed - {exc}"; job_logger.error(f"JID:{job_id} - Task for file {job_info['filenames'][i]} failed: {exc}")
                        except Exception as e_inner: job_info['results'][i] = f"ERROR: Failed to get task exception - {e_inner}"; job_logger.error(f"JID:{job_id} - Failed to get exception for failed task {future.key}: {e_inner}")
                        num_done += 1
                    else: all_accounted_for = False
                except TimeoutError: job_logger.warning(f"JID:{job_id} - Timeout getting result/exception for done future {future.key}"); all_accounted_for = False
                except Exception as e: job_logger.error(f"JID:{job_id} - Error getting result/exception for done future {future.key}: {e}", exc_info=True); job_info['results'][i] = f"ERROR: Failed to retrieve result/exception - {e}"; num_done += 1
            else: all_accounted_for = False

        job_info['completed_tasks'] = num_done
        if all_accounted_for:
            job_info['status'] = 'complete'
            if previous_status != 'complete': job_logger.info(f"JID:{job_id} - Job marked as complete ({num_done}/{job_info['total_tasks']} tasks finished).")
            if 'futures' in job_info: del job_info['futures']
        else: job_info['status'] = 'processing'

    if job_info['total_tasks'] > 0: progress = int((job_info.get('completed_tasks', 0) / job_info['total_tasks']) * 100)
    if job_info['status'] == 'complete': progress = 100

    job_logger.debug(f"JID:{job_id} - Displaying results. Status: {job_info['status']}, Progress: {progress}%")
    results_display = list(zip(job_info['filenames'], job_info['results']))
    return render_template('results.html', job_id=job_id, status=job_info['status'], results=results_display, progress=progress)

# --- Run the App ---
if __name__ == '__main__':
    print("DEBUG: Entered __main__ block.")
    if client is None: print("DEBUG: Exiting because client is None."); job_logger.critical("Flask app exiting: Dask client connection failed on startup."); sys.exit(1)
    # --- CHECK THE FLAG set during initialization ---
    if not actor_initialized_ok: print("DEBUG: Exiting because Blip Actor did not initialize successfully."); job_logger.critical("Flask app exiting: BLIP Actor failed to initialize."); sys.exit(1)

    print("DEBUG: Starting Flask app.run()..."); job_logger.info("Starting Flask app on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True) # Use threaded=True usually
    print("DEBUG: Flask app.run() finished.")
else: print("DEBUG: Script is being imported, not run directly.")

# # Filename: app.py (with Job Logging)
# import os
# import uuid
# import time
# import traceback
# import io
# import logging # <<< Import logging
# from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
# from werkzeug.utils import secure_filename
# from dask.distributed import Client, Future, wait, TimeoutError

# # --- Configuration ---
# WEBAPP_BASE_DIR = "/opt/comp0239_coursework/webapp" # Define base for logs/uploads
# UPLOAD_FOLDER = os.path.join(WEBAPP_BASE_DIR, 'uploads')
# LOG_FILE = os.path.join(WEBAPP_BASE_DIR, 'user_jobs.log') # <<< Define log file path
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# DASK_SCHEDULER = '127.0.0.1:8786'

# # --- Job Logger Setup ---
# # Configure a specific logger for job tracking
# job_logger = logging.getLogger('JobLogger')
# job_logger.setLevel(logging.INFO)
# # Prevent job logs from propagating to the root Flask logger if desired
# # job_logger.propagate = False
# try:
#     # Ensure directory exists (though playbook should handle it)
#     os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
#     # Create file handler
#     file_handler = logging.FileHandler(LOG_FILE)
#     # Create formatter - include Job ID in the message itself
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#     # Add handler to the logger
#     if not job_logger.handlers: # Add handler only once
#          job_logger.addHandler(file_handler)
# except Exception as log_e:
#     print(f"CRITICAL: Failed to configure job logger: {log_e}", file=sys.stderr)
#     traceback.print_exc(file=sys.stderr)

# # Flask App Setup
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# # --- Dask Client ---
# client = None
# try:
#     job_logger.info("Attempting to connect to Dask scheduler...")
#     client = Client(DASK_SCHEDULER, timeout="10s")
#     job_logger.info(f"Dask client connected: {client}")
#     job_logger.info(f"Dask dashboard link: {client.dashboard_link}")
# except Exception as e:
#     job_logger.critical(f"Failed to connect to Dask scheduler at {DASK_SCHEDULER}. App may not function.", exc_info=True)
#     # Keep client as None

# # --- Job Store ---
# jobs = {}

# # --- Helper Functions ---
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # --- Dask Task Functions ---
# def load_blip_and_caption(image_bytes):
#     print("WORKER_INFO: Entering load_blip_and_caption")
#     model_name = "Salesforce/blip-image-captioning-base"; processor = None; model = None
#     try:
#         print(f"WORKER_INFO: Importing transformers/torch/PIL within task...")
#         import torch; from transformers import BlipProcessor, BlipForConditionalGeneration; from PIL import Image; import io
#         print("WORKER_INFO: Imports successful inside task.")
#         print(f"WORKER_INFO: Loading BLIP Processor: {model_name}"); processor = BlipProcessor.from_pretrained(model_name)
#         print(f"WORKER_INFO: Loading BLIP Model: {model_name}"); model = BlipForConditionalGeneration.from_pretrained(model_name)
#         device = torch.device("cpu"); model.to(device); model.eval()
#         print("WORKER_INFO: BLIP Processor and Model loaded successfully.")
#         print("WORKER_INFO: Preparing image..."); raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#         print("WORKER_INFO: Processing image with BlipProcessor..."); inputs = processor(raw_image, return_tensors="pt").to(device)
#         print("WORKER_INFO: Generating caption (max_length=50)...")
#         with torch.no_grad(): out = model.generate(**inputs, max_length=50, num_beams=4)
#         print("WORKER_INFO: Decoding caption..."); caption = processor.decode(out[0], skip_special_tokens=True)
#         print(f"WORKER_INFO: Caption generated successfully: '{caption}'")
#         return caption.replace("\n", " ").replace(",", ";").strip()
#     except Exception as e: print(f"WORKER_ERROR in load_blip_and_caption: {e}\n{traceback.format_exc()}", file=sys.stderr); return f"ERROR: Caption generation failed ({e.__class__.__name__})"
#     finally: del processor; del model

# def process_image_bytes(image_bytes):
#     t_start = time.time()
#     if not image_bytes: print("WORKER_ERROR: Received empty image bytes.", file=sys.stderr); return "ERROR: Empty image data received"
#     try: return load_blip_and_caption(image_bytes)
#     except Exception as e: print(f"WORKER_ERROR in process_image_bytes wrapper: {e}\n{traceback.format_exc()}", file=sys.stderr); return f"ERROR: Processing wrapper failed ({e.__class__.__name__})"


# # --- Flask Routes ---
# @app.route('/', methods=['GET'])
# def index():
#     """Display the upload form."""
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     """Handle file uploads, submit tasks to Dask."""
#     # --- LOGGING: Request received ---
#     job_logger.info(f"Received upload request from {request.remote_addr}")

#     if client is None:
#         job_logger.error("Upload failed: Dask client not connected.")
#         flash('Dask connection error. Cannot process uploads.', 'error')
#         return redirect(url_for('index'))

#     if 'images' not in request.files:
#         job_logger.warning("Upload failed: No 'images' file part in request.")
#         flash('No file part in request.', 'error')
#         return redirect(url_for('index'))

#     files = request.files.getlist('images')
#     submitted_futures = []; original_filenames = []; job_id = str(uuid.uuid4()) # Generate unique job ID first
#     processed_count = 0

#     # --- LOGGING: Job ID Generated ---
#     job_logger.info(f"JID:{job_id} - Generated Job ID.")

#     if not files or files[0].filename == '':
#          job_logger.warning(f"JID:{job_id} - Upload failed: No files selected.")
#          flash('No selected files.', 'warning')
#          return redirect(url_for('index'))

#     for file in files:
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename) # Secure filename early
#             try:
#                 image_bytes = file.read()
#                 if not image_bytes:
#                      job_logger.warning(f"JID:{job_id} - Skipping empty file: {filename}")
#                      flash(f'Skipping empty file: {filename}', 'warning')
#                      continue

#                 # --- LOGGING: Submitting task ---
#                 job_logger.debug(f"JID:{job_id} - Submitting task for {filename}...") # Debug level for individual tasks
#                 future = client.submit(process_image_bytes, image_bytes, pure=False)
#                 submitted_futures.append(future)
#                 original_filenames.append(filename)
#                 processed_count += 1

#             except Exception as e:
#                 job_logger.error(f"JID:{job_id} - Error processing/submitting file {filename}: {e}", exc_info=True)
#                 flash(f'Error processing file {filename}: {e}', 'error')
#                 # Continue with other files

#         elif file and file.filename != '':
#             job_logger.warning(f"JID:{job_id} - File type not allowed: {file.filename}")
#             flash(f'File type not allowed: {file.filename}', 'warning')

#     if not submitted_futures:
#          job_logger.error(f"JID:{job_id} - No valid image files were submitted.")
#          flash('No valid image files were processed.', 'error')
#          return redirect(url_for('index'))

#     # Store job information
#     jobs[job_id] = {
#         'status': 'processing',
#         'filenames': original_filenames,
#         'futures': submitted_futures,
#         'results': [None] * len(submitted_futures),
#         'total_tasks': len(submitted_futures),
#         'completed_tasks': 0
#     }
#     # --- LOGGING: Job Submission Summary ---
#     job_logger.info(f"JID:{job_id} - Submitted {processed_count} tasks to Dask cluster.")

#     flash(f'Successfully submitted {processed_count} image(s) for captioning. Job ID: {job_id}', 'success')
#     return redirect(url_for('show_results', job_id=job_id))


# @app.route('/results/<job_id>', methods=['GET'])
# def show_results(job_id):
#     """Display status and results for a given job ID."""
#     # --- LOGGING: Results page requested ---
#     job_logger.info(f"JID:{job_id} - Request to view results.")

#     job_info = jobs.get(job_id)
#     if not job_info:
#         job_logger.error(f"JID:{job_id} - Job ID not found in store.")
#         flash(f'Job ID {job_id} not found.', 'error')
#         return redirect(url_for('index'))

#     progress = 0
#     num_done = 0
#     all_accounted_for = True
#     previous_status = job_info['status'] # Store previous status

#     if job_info.get('futures'):
#         futures_to_check = job_info['futures']
#         # --- LOGGING: Checking future status ---
#         job_logger.debug(f"JID:{job_id} - Checking status of {len(futures_to_check)} futures.")
#         for i, future in enumerate(futures_to_check):
#             if job_info['results'][i] is not None:
#                 num_done += 1
#                 continue

#             if future.done():
#                 try:
#                     if future.status == 'finished':
#                        job_info['results'][i] = future.result(timeout=1)
#                        num_done += 1
#                     elif future.status == 'error':
#                         try:
#                             exc = future.exception(timeout=1) # Get exception
#                             job_info['results'][i] = f"ERROR: Task failed - {exc}"
#                             # --- LOGGING: Task Error ---
#                             job_logger.error(f"JID:{job_id} - Task for file {job_info['filenames'][i]} failed: {exc}")
#                         except Exception as e_inner:
#                             job_info['results'][i] = f"ERROR: Failed to get task exception - {e_inner}"
#                             job_logger.error(f"JID:{job_id} - Failed to get exception for failed task {future.key}: {e_inner}")
#                         num_done += 1
#                     else: all_accounted_for = False
#                 except TimeoutError:
#                     job_logger.warning(f"JID:{job_id} - Timeout getting result/exception for done future {future.key}")
#                     all_accounted_for = False
#                 except Exception as e:
#                      job_logger.error(f"JID:{job_id} - Error getting result/exception for done future {future.key}: {e}", exc_info=True)
#                      job_info['results'][i] = f"ERROR: Failed to retrieve result/exception - {e}"
#                      num_done += 1
#             else: all_accounted_for = False

#         job_info['completed_tasks'] = num_done
#         if all_accounted_for:
#             job_info['status'] = 'complete'
#             # --- LOGGING: Job Completed ---
#             if previous_status != 'complete': # Log only on transition
#                  job_logger.info(f"JID:{job_id} - Job marked as complete ({num_done}/{job_info['total_tasks']} tasks finished).")
#             if 'futures' in job_info: del job_info['futures'] # Cleanup
#         else:
#              job_info['status'] = 'processing'

#     # Recalculate progress
#     if job_info['total_tasks'] > 0:
#         progress = int((job_info.get('completed_tasks', 0) / job_info['total_tasks']) * 100)
#     if job_info['status'] == 'complete': progress = 100

#     # --- LOGGING: Results page status update ---
#     job_logger.debug(f"JID:{job_id} - Displaying results. Status: {job_info['status']}, Progress: {progress}%")

#     results_display = list(zip(job_info['filenames'], job_info['results']))

#     return render_template('results.html',
#                            job_id=job_id,
#                            status=job_info['status'],
#                            results=results_display,
#                            progress=progress)

# # --- Run the App ---
# if __name__ == '__main__':
#     if client is None:
#         job_logger.critical("Flask app exiting: Dask client connection failed on startup.")
#         sys.exit(1)
#     job_logger.info("Starting Flask app on http://0.0.0.0:5000")
#     app.run(host='0.0.0.0', port=5000, debug=False)
