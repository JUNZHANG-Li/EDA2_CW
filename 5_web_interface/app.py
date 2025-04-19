# Filename: app.py (with Real-time Status API)
import os
import uuid
import time
import traceback
import io
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify # Import jsonify
from werkzeug.utils import secure_filename
from dask.distributed import Client, Future, wait, TimeoutError # Keep TimeoutError

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
jobs = {} # Global job store

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Dask Task Functions ---
# (load_blip_and_caption and process_image_bytes remain unchanged)
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
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if client is None: flash('Dask connection error.', 'error'); return redirect(url_for('index'))
    if 'images' not in request.files: flash('No file part.', 'error'); return redirect(url_for('index'))

    files = request.files.getlist('images')
    submitted_futures = []; original_filenames = []; job_id = str(uuid.uuid4())
    processed_count = 0

    if not files or files[0].filename == '': flash('No selected files.', 'warning'); return redirect(url_for('index'))

    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename); image_bytes = file.read()
                if not image_bytes: flash(f'Skipping empty file: {filename}', 'warning'); continue
                print(f"Submitting task for {filename}...")
                future = client.submit(process_image_bytes, image_bytes, pure=False)
                submitted_futures.append(future); original_filenames.append(filename); processed_count += 1
            except Exception as e: flash(f'Error processing file {file.filename}: {e}', 'error'); print(f"Error submitting task: {e}"); traceback.print_exc()
        elif file and file.filename != '': flash(f'File type not allowed: {file.filename}', 'warning')

    if not submitted_futures: flash('No valid images processed.', 'error'); return redirect(url_for('index'))

    jobs[job_id] = {
        'status': 'processing', # Start as processing
        'filenames': original_filenames,
        'futures': submitted_futures,
        'results': [None] * len(submitted_futures),
        'total_tasks': len(submitted_futures),
        'completed_tasks': 0
    }
    flash(f'Submitted {processed_count} image(s). Job ID: {job_id}', 'success')
    return redirect(url_for('show_results', job_id=job_id))


@app.route('/results/<job_id>', methods=['GET'])
def show_results(job_id):
    """Render the initial results page structure."""
    job_info = jobs.get(job_id)
    if not job_info:
        flash(f'Job ID {job_id} not found.', 'error')
        return redirect(url_for('index'))

    # Pass only necessary info for initial render
    # JavaScript will fetch detailed status/results
    return render_template('results.html',
                           job_id=job_id,
                           initial_filenames=job_info['filenames']) # Pass filenames


@app.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id):
    """API endpoint to get detailed status and results for a job."""
    job_info = jobs.get(job_id)
    if not job_info:
        return jsonify({"error": "Job not found"}), 404
    if client is None:
         return jsonify({"error": "Dask client not available"}), 503

    image_statuses = []
    num_done = 0
    overall_status = job_info.get('status', 'unknown') # Use stored status

    if overall_status == 'processing':
        all_accounted_for = True # Assume complete until proven otherwise
        futures_to_check = job_info.get('futures', [])

        if not futures_to_check: # Handle case where futures might have been cleared
            overall_status = 'complete' if job_info.get('results') else 'error' # Infer status
            all_accounted_for = True
        else:
            for i, future in enumerate(futures_to_check):
                filename = job_info['filenames'][i]
                current_result = job_info['results'][i]
                status_str = 'queued' # Default

                # If we already have the result, don't check future again
                if current_result is not None:
                    num_done += 1
                    status_str = 'error' if current_result.startswith("ERROR:") else 'complete'
                elif future.done():
                    try:
                        if future.status == 'finished':
                            # Attempt to get result only if not already stored
                            current_result = future.result(timeout=0.1) # Short timeout
                            job_info['results'][i] = current_result # Store it
                            status_str = 'complete'
                        elif future.status == 'error':
                            exception_str = f"ERROR: Task failed - {future.exception(timeout=0.1)}"
                            job_info['results'][i] = exception_str # Store error
                            current_result = exception_str
                            status_str = 'error'
                        else:
                            # Should be impossible if future.done() is true
                            status_str = 'unknown'
                            all_accounted_for = False # Mark as not done
                        num_done += 1
                    except TimeoutError:
                        # Result wasn't available immediately, keep polling
                        status_str = 'finishing' # Indicate it's done but result pending
                        all_accounted_for = False
                    except Exception as e:
                        print(f"Error retrieving result/exception for job {job_id}, future {future.key}: {e}")
                        error_str = f"ERROR: Failed retrieval - {e.__class__.__name__}"
                        job_info['results'][i] = error_str
                        current_result = error_str
                        status_str = 'error'
                        num_done += 1
                else:
                    # Future not done, check Dask status
                    status_str = future.status # e.g., 'pending', 'running'
                    all_accounted_for = False

                image_statuses.append({
                    "filename": filename,
                    "status": status_str,
                    "result": current_result # Send current result (or None)
                })

            # Update overall status
            if all_accounted_for:
                overall_status = 'complete'
                # Optional: Clear futures list from job_info to save memory
                if 'futures' in job_info: del job_info['futures']
            else:
                overall_status = 'processing'

            # Update job store
            job_info['status'] = overall_status
            job_info['completed_tasks'] = num_done

    else: # Job already marked as complete or error
        # Just retrieve stored results if status isn't 'processing'
        for i, filename in enumerate(job_info['filenames']):
            status_str = 'error' if str(job_info['results'][i]).startswith("ERROR:") else 'complete'
            image_statuses.append({
                "filename": filename,
                "status": status_str,
                "result": job_info['results'][i]
            })
        num_done = job_info.get('completed_tasks', len(image_statuses)) # Use stored count or total


    # Calculate progress
    progress_percent = 0
    if job_info['total_tasks'] > 0:
        progress_percent = int((num_done / job_info['total_tasks']) * 100)
    if overall_status == 'complete':
         progress_percent = 100


    return jsonify({
        "job_id": job_id,
        "overall_status": overall_status,
        "progress_percent": progress_percent,
        "image_statuses": image_statuses
    })


# --- Run the App ---
if __name__ == '__main__':
    if client is None: print("Cannot start Flask app without Dask connection.", file=sys.stderr); sys.exit(1)
    print("Starting Flask app on http://0.0.0.0:5000"); app.run(host='0.0.0.0', port=5000, debug=False)