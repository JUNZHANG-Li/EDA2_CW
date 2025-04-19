# Filename: app.py
import os
import uuid
import time
import traceback
import io
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dask.distributed import Client, Future, wait, TimeoutError # Keep TimeoutError if using wait with timeout

# --- Configuration ---
UPLOAD_FOLDER = '/opt/comp0239_coursework/webapp/uploads' # Store uploads here
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Ensure UPLOAD_FOLDER exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dask Scheduler location (running on the same host)
DASK_SCHEDULER = '127.0.0.1:8786'

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limit uploads to 16MB
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # Needed for flash messages

# --- Global Variables ---
# Dask Client (Connect once)
try:
    print(f"Connecting to Dask scheduler at {DASK_SCHEDULER}...")
    client = Client(DASK_SCHEDULER, timeout="10s")
    print(f"Dask client connected: {client}")
    print(f"Dashboard Link: {client.dashboard_link}")
except Exception as e:
    print(f"CRITICAL: Failed to connect to Dask scheduler at {DASK_SCHEDULER}. App will not function.", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    client = None # Set client to None if connection fails

# Simple in-memory store for job status and results
# Structure: jobs[job_id] = {'status': 'pending'/'processing'/'complete'/'error',
#                          'filenames': [list of original filenames],
#                          'futures': [list of Dask Future objects],
#                          'results': [list of captions or error strings]}
jobs = {}

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Dask Task Functions (Copied/Adapted from run_capacity_test.py) ---
# NOTE: These functions are defined globally so Dask can find them.
# They will be executed on the WORKER nodes.

def load_blip_and_caption(image_bytes):
    """Loads BLIP model and generates caption INSIDE the task function."""
    # Using print for worker logs as logging setup is complex inside tasks
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
        # Return just the caption/error
        return caption.replace("\n", " ").replace(",", ";").strip()
    except Exception as e:
        print(f"WORKER_ERROR in load_blip_and_caption: {e}\n{traceback.format_exc()}", file=sys.stderr)
        # Return a clear error message back to the Flask app
        return f"ERROR: Caption generation failed ({e.__class__.__name__})"
    finally:
        # Attempt to clean up memory on worker
        del processor
        del model

# This wrapper is slightly simplified as we don't need download logic
def process_image_bytes(image_bytes):
    """
    Wrapper task that takes image bytes and calls the captioning function.
    Returns the caption string or an error message.
    """
    t_start = time.time()
    if not image_bytes:
        print("WORKER_ERROR: Received empty image bytes.", file=sys.stderr)
        return "ERROR: Empty image data received"
    try:
        # Directly call the captioning function
        caption_result = load_blip_and_caption(image_bytes)
        return caption_result
    except Exception as e:
        # Catch any unexpected errors in this wrapper
        print(f"WORKER_ERROR in process_image_bytes wrapper: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return f"ERROR: Processing wrapper failed ({e.__class__.__name__})"


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Display the upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads, submit tasks to Dask."""
    if client is None:
         flash('Dask connection error. Cannot process uploads.', 'error')
         return redirect(url_for('index'))

    if 'images' not in request.files:
        flash('No file part in request.', 'error')
        return redirect(url_for('index'))

    files = request.files.getlist('images') # Get list of files

    submitted_futures = []
    original_filenames = []
    job_id = str(uuid.uuid4()) # Generate unique job ID

    if not files or files[0].filename == '':
         flash('No selected files.', 'warning')
         return redirect(url_for('index'))

    processed_count = 0
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename) # Basic security
                # Read image bytes directly from the uploaded file stream
                image_bytes = file.read()
                if not image_bytes:
                     flash(f'Skipping empty file: {filename}', 'warning')
                     continue

                # Submit task to Dask using image bytes
                print(f"Submitting task for {filename}...")
                future = client.submit(process_image_bytes, image_bytes, pure=False)
                submitted_futures.append(future)
                original_filenames.append(filename)
                processed_count += 1

            except Exception as e:
                flash(f'Error processing file {file.filename}: {e}', 'error')
                print(f"Error submitting task for {file.filename}: {e}")
                traceback.print_exc()
                # Don't stop processing other files

        elif file and file.filename != '':
            flash(f'File type not allowed: {file.filename}', 'warning')

    if not submitted_futures:
         flash('No valid image files were processed.', 'error')
         return redirect(url_for('index'))

    # Store job information
    jobs[job_id] = {
        'status': 'processing',
        'filenames': original_filenames,
        'futures': submitted_futures,
        'results': [None] * len(submitted_futures) # Placeholder for results
    }

    flash(f'Successfully submitted {processed_count} image(s) for captioning. Job ID: {job_id}', 'success')
    # Redirect to the results page for this job
    return redirect(url_for('show_results', job_id=job_id))


@app.route('/results/<job_id>', methods=['GET'])
def show_results(job_id):
    """Display status and results for a given job ID."""
    job_info = jobs.get(job_id)

    if not job_info:
        flash(f'Job ID {job_id} not found.', 'error')
        return redirect(url_for('index'))

    # Check status of futures if job is still processing
    if job_info['status'] == 'processing':
        all_done = True
        try:
            # Check status without waiting long
            # Note: future.status can be 'pending', 'running', 'finished', 'error'
            for i, future in enumerate(job_info['futures']):
                if future.status == 'finished':
                    if job_info['results'][i] is None: # Get result if not already stored
                         job_info['results'][i] = future.result(timeout=0.1) # Short timeout
                elif future.status == 'error':
                     if job_info['results'][i] is None:
                         try:
                            # Store the exception string as the result
                            job_info['results'][i] = f"ERROR: Task failed - {future.exception()}"
                         except Exception as e:
                             job_info['results'][i] = f"ERROR: Failed to get exception - {e}"
                else: # 'pending' or 'running'
                    all_done = False

            if all_done:
                job_info['status'] = 'complete'
                # Clean up futures list to save memory (optional)
                # job_info['futures'] = None
        except TimeoutError:
            log.debug(f"Timeout getting result for job {job_id}, will try again.")
            # Status remains 'processing'
        except Exception as e:
             print(f"Error checking/getting results for job {job_id}: {e}")
             traceback.print_exc()
             # Optionally set job status to 'error'
             # job_info['status'] = 'error'


    # Combine filenames and results for easy display in template
    results_display = list(zip(job_info['filenames'], job_info['results']))

    return render_template('results.html',
                           job_id=job_id,
                           status=job_info['status'],
                           results=results_display)


# --- Run the App ---
if __name__ == '__main__':
    if client is None:
        print("Cannot start Flask app without Dask connection.", file=sys.stderr)
        sys.exit(1)
    # Run on host 0.0.0.0 to make it accessible externally on the host node's IP
    # Use a port that's open (e.g., 5000 by default) - ensure firewall allows it
    # Turn off debug mode for production/testing
    print("Starting Flask app on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)