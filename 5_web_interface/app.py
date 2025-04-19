# --- Inside app.py ---
import logging
import time
# ... other imports ...

# --- Configuration ---
# ... (UPLOAD_FOLDER, ALLOWED_EXTENSIONS, DASK_SCHEDULER) ...
HOST_LOG_FILE = '/opt/comp0239_coursework/webapp/host_timing.log' # Specific log file

# Flask App Setup
# ... (app creation, config, secret_key) ...

# --- Specific Timing Logger Setup ---
# Remove basicConfig if you used it before, configure root logger or specific loggers
timing_log = logging.getLogger('WebAppTiming')
timing_log.setLevel(logging.INFO) # Or DEBUG for more verbosity
# Ensure handlers aren't added multiple times if app reloads
if not timing_log.handlers:
    # File Handler for timing logs
    file_handler = logging.FileHandler(HOST_LOG_FILE)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    timing_log.addHandler(file_handler)
    # Optional: Console Handler as well
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(file_formatter)
    timing_log.addHandler(console_handler)

# --- Dask Client ---
# ... (Connect client as before) ...

# --- Job Store ---
# Add fields to store timestamps
# jobs[job_id] = {...,
#                 'timestamps': {'received': None, 'submitted': None, 'results_checked': [], 'complete': None},
#                 'task_times': [None] * len(...) # Optional: Store worker times
#                }
jobs = {}

# --- Dask Task Functions ---
# Modify task to return timing info (optional)
def process_image_bytes(image_bytes):
    """
    Wrapper task that takes image bytes, calls captioning, AND returns timing.
    Returns tuple: (caption_or_error, duration_on_worker)
    """
    task_start_time = time.time()
    result = "ERROR: Task failed before processing" # Default
    try:
        if not image_bytes:
            print("WORKER_ERROR: Received empty image bytes.", file=sys.stderr)
            result = "ERROR: Empty image data received"
        else:
            result = load_blip_and_caption(image_bytes) # Call the captioning func
    except Exception as e:
        result = f"ERROR: Processing wrapper failed ({e.__class__.__name__})"
        print(f"WORKER_ERROR in process_image_bytes wrapper: {e}\n{traceback.format_exc()}", file=sys.stderr)
    finally:
        task_end_time = time.time()
        duration = task_end_time - task_start_time
        print(f"WORKER_INFO: process_image_bytes took {duration:.4f}s")
        # Return both the original result and the duration
        return (result, duration)

# (load_blip_and_caption remains the same internally, still prints to worker log)

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # (No changes needed here)
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads, submit tasks to Dask, log timing."""
    request_received_time = time.time() # Time request handling starts
    if client is None: flash('Dask connection error...', 'error'); return redirect(url_for('index'))
    # (File checking logic - same as before) ...

    files = request.files.getlist('images')
    # ... check if files exist ...

    submitted_futures = []; original_filenames = []; job_id = str(uuid.uuid4())
    processed_count = 0
    submission_start_time = time.time() # Time just before loop

    timing_log.info(f"[Job {job_id}] Received upload request.")

    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_read_start = time.time()
                image_bytes = file.read()
                file_read_end = time.time()
                if not image_bytes: flash(f'Skipping empty file: {filename}', 'warning'); continue

                submit_start = time.time()
                future = client.submit(process_image_bytes, image_bytes, pure=False)
                submit_end = time.time()
                submitted_futures.append(future)
                original_filenames.append(filename)
                processed_count += 1
                timing_log.info(f"[Job {job_id}] File '{filename}': Read took {(file_read_end-file_read_start)*1000:.2f}ms, Submit took {(submit_end-submit_start)*1000:.2f}ms")

            except Exception as e: # ... error handling ...
        # ... other file checks ...

    submission_end_time = time.time() # Time after loop

    if not submitted_futures: # ... error handling ...

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
             'last_results_check': None, # Track last check time
             'first_result_received': None, # Track when first result arrives
             'all_results_received': None # Track when last result arrives
             }
    }
    timing_log.info(f"[Job {job_id}] Submitted {processed_count} tasks. Total submission time: {(submission_end_time-submission_start_time)*1000:.2f}ms")

    flash(f'Successfully submitted {processed_count} image(s). Job ID: {job_id}', 'success')
    return redirect(url_for('show_results', job_id=job_id))


@app.route('/results/<job_id>', methods=['GET'])
def show_results(job_id):
    """Display status, results, and log timing for result checks."""
    results_check_start_time = time.time()
    job_info = jobs.get(job_id)
    if not job_info: flash(f'Job ID {job_id} not found.', 'error'); return redirect(url_for('index'))

    timing_log.info(f"[Job {job_id}] Request for results page.")
    job_info['timestamps']['last_results_check'] = results_check_start_time

    progress = 0
    num_done = 0
    all_accounted_for = True

    if job_info.get('futures'):
        futures_to_check = job_info['futures']
        # Use wait(..., timeout=0) for a quick non-blocking check first
        done_set, _ = wait(futures_to_check, timeout=0)

        for i, future in enumerate(futures_to_check):
            if job_info['results'][i] is not None: # Already processed
                num_done += 1
                continue

            # Only try to get result if wait indicated it's done
            if future in done_set:
                result_fetch_start = time.time()
                try:
                    # Get tuple result: (caption_or_error, duration_on_worker)
                    task_result_tuple = future.result(timeout=0.1) # Short timeout
                    job_info['results'][i] = task_result_tuple[0] # Store caption/error
                    job_info['task_times'][i] = task_result_tuple[1] # Store worker duration
                    num_done += 1
                    if job_info['timestamps']['first_result_received'] is None:
                        job_info['timestamps']['first_result_received'] = time.time()
                    timing_log.info(f"[Job {job_id}] Received result for task {i} (Worker time: {task_result_tuple[1]:.3f}s). Fetch took {(time.time()-result_fetch_start)*1000:.2f}ms.")

                except TimeoutError:
                    timing_log.debug(f"[Job {job_id}] Timeout getting result for done future {future.key}")
                    all_accounted_for = False # Not really done if we can't get result
                except Exception as e:
                    job_info['results'][i] = f"ERROR: Failed retrieval - {e}"
                    job_info['task_times'][i] = -1 # Indicate error
                    num_done += 1 # Count as 'done' for progress
                    if job_info['timestamps']['first_result_received'] is None:
                        job_info['timestamps']['first_result_received'] = time.time()
                    timing_log.error(f"[Job {job_id}] Error getting result for task {i}: {e}. Fetch took {(time.time()-result_fetch_start)*1000:.2f}ms.")
            else:
                # Future not in done_set, still pending/running
                all_accounted_for = False

        # Update overall job status
        if all_accounted_for and num_done == job_info['total_tasks']:
            job_info['status'] = 'complete'
            job_info['timestamps']['all_results_received'] = time.time()
            timing_log.info(f"[Job {job_id}] All {num_done} results received. Status set to complete.")
            if 'futures' in job_info: del job_info['futures'] # Cleanup
        elif job_info['status'] != 'error': # Don't override error status
             job_info['status'] = 'processing'


    # Recalculate progress
    job_info['completed_tasks'] = num_done
    if job_info['total_tasks'] > 0: progress = int((num_done / job_info['total_tasks']) * 100)
    if job_info['status'] == 'complete': progress = 100

    results_display = list(zip(job_info['filenames'], job_info['results']))
    results_check_end_time = time.time()
    timing_log.info(f"[Job {job_id}] Results page loaded. Check took {(results_check_end_time - results_check_start_time)*1000:.2f}ms. Progress: {progress}%.")

    return render_template('results.html', job_id=job_id, status=job_info['status'], results=results_display, progress=progress)


# --- Run the App ---
if __name__ == '__main__':
    if client is None: print("Cannot start Flask app without Dask connection.", file=sys.stderr); sys.exit(1)
    print(f"Starting Flask app on http://0.0.0.0:5000, logging timings to {HOST_LOG_FILE}")
    app.run(host='0.0.0.0', port=5000, debug=False) # IMPORTANT: debug=False for stability