# Filename: app.py (with Detailed Status, Timing & Collapsible Logs)
import os
import uuid
import time
import traceback
import io
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dask.distributed import Client, Future, wait, TimeoutError

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
    print(f"CRITICAL: Failed to connect to Dask scheduler: {e}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

# --- Job Store ---
# Enhanced structure for jobs[job_id]['image_data']
# image_data will be a list of dictionaries:
# [{'filename': str, 'future': DaskFuture, 'status': str,
#   'result': str/None, 'submit_time': float/None, 'start_processing_time': float/None,
#   'end_time': float/None, 'duration': float/None}, ...]
jobs = {}

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Dask Task Functions ---
# (load_blip_and_caption remains the same)
def load_blip_and_caption(image_bytes):
    # ... (same as previous version) ...
    print("WORKER_INFO: Entering load_blip_and_caption")
    model_name = "Salesforce/blip-image-captioning-base"; processor = None; model = None
    try:
        # Start timing *inside* the worker task
        task_start_time = time.time()
        print(f"WORKER_INFO: Importing transformers/torch/PIL within task...")
        import torch; from transformers import BlipProcessor, BlipForConditionalGeneration; from PIL import Image; import io
        print("WORKER_INFO: Imports successful inside task.")
        print(f"WORKER_INFO: Loading BLIP Processor: {model_name}"); processor = BlipProcessor.from_pretrained(model_name)
        print(f"WORKER_INFO: Loading BLIP Model: {model_name}"); model = BlipForConditionalGeneration.from_pretrained(model_name)
        device = torch.device("cpu"); model.to(device); model.eval()
        load_end_time = time.time()
        print(f"WORKER_INFO: BLIP Processor and Model loaded successfully in {load_end_time - task_start_time:.2f}s.")
        print("WORKER_INFO: Preparing image..."); raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        prep_end_time = time.time()
        print(f"WORKER_INFO: Image prepared in {prep_end_time - load_end_time:.2f}s.")
        print("WORKER_INFO: Processing image with BlipProcessor..."); inputs = processor(raw_image, return_tensors="pt").to(device)
        proc_end_time = time.time()
        print(f"WORKER_INFO: BlipProcessor finished in {proc_end_time - prep_end_time:.2f}s.")
        print("WORKER_INFO: Generating caption (max_length=50)...")
        with torch.no_grad(): out = model.generate(**inputs, max_length=50, num_beams=4)
        gen_end_time = time.time()
        print(f"WORKER_INFO: Caption generated in {gen_end_time - proc_end_time:.2f}s.")
        print("WORKER_INFO: Decoding caption..."); caption = processor.decode(out[0], skip_special_tokens=True)
        decode_end_time = time.time()
        print(f"WORKER_INFO: Caption decoded in {decode_end_time - gen_end_time:.2f}s.")
        print(f"WORKER_INFO: Caption generated successfully: '{caption}'")
        return caption.replace("\n", " ").replace(",", ";").strip()
    except Exception as e: print(f"WORKER_ERROR in load_blip_and_caption: {e}\n{traceback.format_exc()}", file=sys.stderr); return f"ERROR: Caption generation failed ({e.__class__.__name__})"
    finally: del processor; del model


def process_image_bytes(image_bytes):
    # (This wrapper is mostly the same, calls load_blip_and_caption)
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
    job_id = str(uuid.uuid4())
    image_data_list = [] # Use a list to store data for each image

    if not files or files[0].filename == '': flash('No selected files.', 'warning'); return redirect(url_for('index'))

    processed_count = 0
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                image_bytes = file.read()
                if not image_bytes: flash(f'Skipping empty file: {filename}', 'warning'); continue

                submit_time = time.time() # Record submission time
                print(f"Submitting task for {filename} at {submit_time:.2f}...")
                future = client.submit(process_image_bytes, image_bytes, pure=False)

                # Initialize image data dictionary
                image_data_list.append({
                    'filename': filename,
                    'future': future,
                    'status': 'submitted', # Initial status
                    'result': None,
                    'submit_time': submit_time,
                    'start_processing_time': None, # Worker sets this conceptually
                    'end_time': None,
                    'duration': None
                })
                processed_count += 1
            except Exception as e: flash(f'Error processing file {file.filename}: {e}', 'error'); print(f"Error submitting task: {e}"); traceback.print_exc()
        elif file and file.filename != '': flash(f'File type not allowed: {file.filename}', 'warning')

    if not image_data_list: flash('No valid images processed.', 'error'); return redirect(url_for('index'))

    # Store job information with the new image_data list
    jobs[job_id] = {
        'status': 'processing',
        'image_data': image_data_list, # Store the list of dicts
        'total_tasks': len(image_data_list),
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
    # Pass filenames for initial rendering
    initial_filenames = [img['filename'] for img in job_info.get('image_data', [])]
    return render_template('results.html',
                           job_id=job_id,
                           initial_filenames=initial_filenames)


@app.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id):
    """API endpoint to get detailed status and results for a job."""
    job_info = jobs.get(job_id)
    if not job_info: return jsonify({"error": "Job not found"}), 404
    if client is None: return jsonify({"error": "Dask client not available"}), 503

    image_statuses_api = [] # Data to return via API
    num_done = 0
    overall_status = job_info.get('status', 'unknown')
    all_tasks_accounted_for = True # Assume complete until proven otherwise

    image_data_list = job_info.get('image_data', [])

    if overall_status == 'processing': # Only check futures if still processing
        if not image_data_list: # Should not happen if upload worked
             overall_status = 'error'
             all_tasks_accounted_for = True
        else:
            current_futures = [img['future'] for img in image_data_list if img['future'] is not None and img['end_time'] is None]
            done_set = set()
            if current_futures:
                try:
                    # Check which futures completed without significant waiting
                    done_set, _ = wait(current_futures, timeout=0) # Non-blocking check
                except Exception as e:
                     print(f"Error during dask wait for job {job_id}: {e}") # Log error but continue

            for i, img_data in enumerate(image_data_list):
                # Skip if already processed (has end_time)
                if img_data['end_time'] is not None:
                    num_done += 1
                    continue # Already have final data for this image

                future = img_data['future']
                if future is None: # Should not happen in this flow
                    all_tasks_accounted_for = False
                    continue

                status_str = img_data['status'] # Keep last known status unless updated

                if future in done_set: # If wait() found it's done
                    try:
                        # Try to get result/error and record end time
                        end_time = time.time()
                        img_data['end_time'] = end_time
                        if future.status == 'finished':
                            img_data['result'] = future.result(timeout=0.1) # Short timeout
                            img_data['status'] = 'complete'
                        elif future.status == 'error':
                            exc = future.exception(timeout=0.1)
                            img_data['result'] = f"ERROR: Task failed - {exc.__class__.__name__}"
                            img_data['status'] = 'error'
                        else: # Should not happen
                             img_data['status'] = 'unknown_done'

                        # Calculate duration
                        if img_data['submit_time']:
                            img_data['duration'] = end_time - img_data['submit_time']

                        num_done += 1 # Mark as done for progress

                    except TimeoutError:
                        # Still processing result retrieval, keep polling
                        img_data['status'] = 'finishing'
                        all_tasks_accounted_for = False
                    except Exception as e:
                        print(f"Error retrieving result/exception for job {job_id}, future {future.key}: {e}")
                        img_data['result'] = f"ERROR: Failed retrieval - {e.__class__.__name__}"
                        img_data['status'] = 'error'
                        img_data['end_time'] = time.time() # Mark end time even on retrieval error
                        if img_data['submit_time']: img_data['duration'] = img_data['end_time'] - img_data['submit_time']
                        num_done += 1
                    finally:
                        # We don't need the future object anymore once processed
                        img_data['future'] = None

                else: # Future not in done_set
                     status_str = future.status # Get current dask status ('pending', 'running', etc)
                     img_data['status'] = status_str # Update status
                     all_tasks_accounted_for = False

            # Update overall job status
            if all_tasks_accounted_for:
                overall_status = 'complete'
            else:
                overall_status = 'processing'
            job_info['status'] = overall_status
            job_info['completed_tasks'] = num_done

    # Prepare API response data AFTER checking/updating statuses
    for img_data in image_data_list:
         # Selectively send data to frontend
        image_statuses_api.append({
            "filename": img_data['filename'],
            "status": img_data['status'], # Send the latest status string
            "result": img_data['result'], # Send caption or error or None
            "submit_time_str": f"{img_data['submit_time']:.2f}" if img_data['submit_time'] else "N/A",
            "end_time_str": f"{img_data['end_time']:.2f}" if img_data['end_time'] else "N/A",
            "duration_str": f"{img_data['duration']:.2f}s" if img_data['duration'] is not None else "N/A"
        })
        if img_data['end_time'] is not None: # Count completed tasks for progress based on end_time
            num_done += 1 # Recalculate based on final data

    # Final progress calculation
    progress_percent = 0
    total_tasks = job_info.get('total_tasks', 0)
    if total_tasks > 0:
        # Use the count of items with an end_time for progress
        final_done_count = sum(1 for item in image_data_list if item.get('end_time') is not None)
        progress_percent = int((final_done_count / total_tasks) * 100)
    if overall_status == 'complete':
         progress_percent = 100


    return jsonify({
        "job_id": job_id,
        "overall_status": overall_status,
        "progress_percent": progress_percent,
        "image_statuses": image_statuses_api # Send the structured data
    })

# --- Run the App ---
if __name__ == '__main__':
    if client is None: print("Cannot start Flask app without Dask connection.", file=sys.stderr); sys.exit(1)
    print("Starting Flask app on http://0.0.0.0:5000"); app.run(host='0.0.0.0', port=5000, debug=False)