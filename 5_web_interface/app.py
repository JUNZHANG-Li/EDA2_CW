# Filename: app.py (with Progress Calculation)
import os
import uuid
import time
import traceback
import io
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify # Add jsonify
from werkzeug.utils import secure_filename
from dask.distributed import Client, Future, wait, TimeoutError # Removed FIRST_COMPLETED if still there

# --- Configuration ---
UPLOAD_FOLDER = '/opt/comp0239_coursework/webapp/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DASK_SCHEDULER = '127.0.0.1:8786'

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # Keep or change secret key

# --- Dask Client ---
client = None
try:
    print(f"Connecting to Dask scheduler at {DASK_SCHEDULER}...")
    client = Client(DASK_SCHEDULER, timeout="10s")
    print(f"Dask client connected: {client}")
    print(f"Dashboard Link: {client.dashboard_link}")
except Exception as e:
    print(f"CRITICAL: Failed to connect to Dask scheduler at {DASK_SCHEDULER}. App will not function.", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

# --- Job Store ---
jobs = {} # Keep simple in-memory store

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Dask Task Functions ---
# (load_blip_and_caption and process_image_bytes remain unchanged from previous BLIP version)
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
    if client is None: flash('Dask connection error. Cannot process uploads.', 'error'); return redirect(url_for('index'))
    if 'images' not in request.files: flash('No file part in request.', 'error'); return redirect(url_for('index'))

    files = request.files.getlist('images')
    submitted_futures = []; original_filenames = []; job_id = str(uuid.uuid4())
    processed_count = 0

    if not files or files[0].filename == '': flash('No selected files.', 'warning'); return redirect(url_for('index'))

    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                image_bytes = file.read()
                if not image_bytes: flash(f'Skipping empty file: {filename}', 'warning'); continue
                print(f"Submitting task for {filename}...")
                future = client.submit(process_image_bytes, image_bytes, pure=False)
                submitted_futures.append(future)
                original_filenames.append(filename)
                processed_count += 1
            except Exception as e: flash(f'Error processing file {file.filename}: {e}', 'error'); print(f"Error submitting task for {file.filename}: {e}"); traceback.print_exc()
        elif file and file.filename != '': flash(f'File type not allowed: {file.filename}', 'warning')

    if not submitted_futures: flash('No valid image files were processed.', 'error'); return redirect(url_for('index'))

    jobs[job_id] = {
        'status': 'processing',
        'filenames': original_filenames,
        'futures': submitted_futures,
        'results': [None] * len(submitted_futures),
        'total_tasks': len(submitted_futures), # Store total count
        'completed_tasks': 0 # Initialize completed count
    }

    flash(f'Successfully submitted {processed_count} image(s) for captioning. Job ID: {job_id}', 'success')
    return redirect(url_for('show_results', job_id=job_id))

@app.route('/results/<job_id>', methods=['GET'])
def show_results(job_id):
    """Display status and results for a given job ID."""
    job_info = jobs.get(job_id)
    if not job_info: flash(f'Job ID {job_id} not found.', 'error'); return redirect(url_for('index'))

    progress = 0 # Default progress

    # Check status of futures if job is still processing
    if job_info['status'] == 'processing':
        num_done = 0
        try:
            futures_to_check = job_info.get('futures', []) # Get the list of futures
            if futures_to_check: # Check only if list exists
                # Use wait with timeout=0 for non-blocking check
                done_set, _ = wait(futures_to_check, timeout=0)

                for i, future in enumerate(futures_to_check):
                    # Process only if the future is done AND result not yet stored
                    if future in done_set and job_info['results'][i] is None:
                        if future.status == 'finished':
                            try:
                                job_info['results'][i] = future.result(timeout=0.1) # Short timeout
                            except Exception as e:
                                job_info['results'][i] = f"ERROR: Failed to get result - {e}"
                        elif future.status == 'error':
                            try:
                                job_info['results'][i] = f"ERROR: Task failed - {future.exception()}"
                            except Exception as e:
                                job_info['results'][i] = f"ERROR: Failed to get exception - {e}"

                    # Count how many have results (are done and processed)
                    if job_info['results'][i] is not None:
                        num_done += 1

            # Update completed count and calculate progress
            job_info['completed_tasks'] = num_done
            if job_info['total_tasks'] > 0:
                progress = int((num_done / job_info['total_tasks']) * 100)

            # Check if all tasks are now done
            if num_done == job_info['total_tasks']:
                job_info['status'] = 'complete'
                # Remove futures from job info once complete to save memory
                if 'futures' in job_info: del job_info['futures']

        except Exception as e:
             print(f"Error checking/getting results for job {job_id}: {e}")
             traceback.print_exc()
             # Optionally set job status to 'error' if check fails badly
             # job_info['status'] = 'error'

    # If already complete, ensure progress is 100
    elif job_info['status'] == 'complete':
        progress = 100
    elif job_info['status'] == 'error':
        # Optionally calculate progress based on completed tasks even if overall job errored
        if job_info['total_tasks'] > 0:
            progress = int((job_info.get('completed_tasks', 0) / job_info['total_tasks']) * 100)


    results_display = list(zip(job_info['filenames'], job_info['results']))

    return render_template('results.html',
                           job_id=job_id,
                           status=job_info['status'],
                           results=results_display,
                           progress=progress) # Pass progress to template

# --- Run the App ---
if __name__ == '__main__':
    if client is None: print("Cannot start Flask app without Dask connection.", file=sys.stderr); sys.exit(1)
    print("Starting Flask app on http://0.0.0.0:5000"); app.run(host='0.0.0.0', port=5000, debug=False)