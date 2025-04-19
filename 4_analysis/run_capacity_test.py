# Filename: run_capacity_test.py (Use "URL Unavailable" Error String)
import time
import datetime
import argparse
import logging
import os
import random
import requests
from requests.exceptions import HTTPError
from PIL import Image
import io
import sys
import re
import traceback

# Dask libraries
from dask.distributed import Client, Future, wait, TimeoutError

# PyTorch/Torchvision
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# --- Configuration ---
DEFAULT_BASE_DIR = "/opt/comp0239_coursework"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_BASE_DIR, "output")
DEFAULT_ID_FILE_NAME = "image_urls_to_process.txt" # File contains URLs
DEFAULT_URL_FILE = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_ID_FILE_NAME)
DEFAULT_OUTPUT_FILE_NAME = "capacity_test_results.csv"
DEFAULT_OUTPUT_FILE = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILE_NAME)
DEFAULT_LOG_FILE_NAME = "capacity_test.log"
DEFAULT_LOG_FILE = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_LOG_FILE_NAME)
DEFAULT_DURATION_HOURS = 24
# DEFAULT_DURATION_HOURS = 0.1 # Use a SHORT duration for testing!

DEFAULT_SCHEDULER = '127.0.0.1:8786'

NUM_WORKER_CORES = 16
MAX_PENDING_BUFFER = 8
MAX_ACTIVE_FUTURES = NUM_WORKER_CORES + MAX_PENDING_BUFFER

# --- NEW: Define the user-friendly error string ---
URL_UNAVAILABLE_ERROR = "URL Unavailable"

# --- Logging Setup ---
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', handlers=[logging.FileHandler(DEFAULT_LOG_FILE), logging.StreamHandler(sys.stdout)])
log = logging.getLogger('CapacityTestRunner')
dask_log = logging.getLogger('distributed')
dask_log.setLevel(logging.WARNING)

# --- Helper Function ---
def extract_id_from_url(url):
    if not isinstance(url, str): return None
    try:
        match = re.search(r'/([^/]+)\.(jpg|jpeg|png|gif)$', url, re.IGNORECASE)
        if match: return match.group(1)
        else: filename = url.split('/')[-1]; return filename.split('.')[0] if '.' in filename else filename
    except Exception: return None

# --- Configuration specific to workers ---
DOWNLOAD_DIR_TEMPLATE = "/data/dask-worker-space/images/{image_id}.jpg"

# --- Model Loading and Captioning Function (Runs on Worker) ---
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
    except Exception as e: print(f"WORKER_ERROR in load_blip_and_caption: {e}\n{traceback.format_exc()}", file=sys.stderr); return f"ERROR_CAPTION_FAILED_{e.__class__.__name__}"
    finally: del processor; del model

# --- Image Processing Wrapper Function (Sent to Workers) ---
def process_image(image_url):
    t_start = time.time(); image_url = image_url.strip()
    if not image_url: return (None, "ERROR_EMPTY_URL")
    image_id = extract_id_from_url(image_url)
    if not image_id: print(f"WORKER_ERROR: Failed ID extraction for URL: {image_url}", file=sys.stderr); return (image_url, "ERROR_ID_EXTRACTION_FAILED")
    local_path = DOWNLOAD_DIR_TEMPLATE.format(image_id=image_id); local_dir = os.path.dirname(local_path)
    try:
        if not os.path.exists(local_path):
            os.makedirs(local_dir, exist_ok=True)
            response = requests.get(image_url, timeout=60); response.raise_for_status()
            image_bytes = response.content
            with open(local_path, 'wb') as f: f.write(image_bytes)
        else:
            with open(local_path, 'rb') as f: image_bytes = f.read()
        caption_result = load_blip_and_caption(image_bytes)
        return (image_id, caption_result)
    except HTTPError as http_err:
        status_code = http_err.response.status_code if http_err.response else None # Check if response exists
        print(f"WORKER_WARN: Download failed for {image_id} from {image_url}: HTTP Status {status_code if status_code else 'None'}")
        # --- UPDATED ERROR STRING ---
        if status_code is None: # If no response object, likely connection/DNS/SSL error
            return (image_id, URL_UNAVAILABLE_ERROR)
        else: # Otherwise, return the specific HTTP error code
            return (image_id, f"ERROR_DOWNLOAD_HTTP_{status_code}")
        # --- END UPDATED ERROR STRING ---
    except requests.exceptions.RequestException as req_err: # Catch other request errors (Timeout, ConnectionError, etc.)
        print(f"WORKER_WARN: Download failed for {image_id} from {image_url}: {req_err}", file=sys.stderr)
        # --- UPDATED ERROR STRING (Treat most request errors as URL Unavailable) ---
        return (image_id, URL_UNAVAILABLE_ERROR)
        # --- END UPDATED ERROR STRING ---
    except Exception as e: print(f"WORKER_ERROR in process_image wrapper for {image_id}: {e}\n{traceback.format_exc()}", file=sys.stderr); return (image_id, f"ERROR_PROCESS_WRAPPER_{e.__class__.__name__}")

# --- Main Control Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run Dask Capacity Test (BLIP Captioning - Target Active: {MAX_ACTIVE_FUTURES})")
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER, help="Dask scheduler address"); parser.add_argument("--urls", default=DEFAULT_URL_FILE, help="Path to image URL list file"); parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Path to output results CSV"); parser.add_argument("--log", default=DEFAULT_LOG_FILE, help="Path to log file"); parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_HOURS, help="Test duration in hours")
    args = parser.parse_args()

    if args.log != DEFAULT_LOG_FILE: log.warning(f"Logging to: {DEFAULT_LOG_FILE}")

    test_duration_seconds = args.duration * 3600
    log.info(f"--- Starting Capacity Test (BLIP Captioning - Target Active: {MAX_ACTIVE_FUTURES}) ---")
    log.info(f"Duration: {args.duration:.2f} hours ({test_duration_seconds:.0f} seconds)"); log.info(f"Scheduler: {args.scheduler}"); log.info(f"Image URL File: {args.urls}"); log.info(f"Output Results File: {args.output}"); log.info(f"Log File: {args.log}")

    if not os.path.exists(args.urls): log.error(f"Image URL file not found: {args.urls}"); sys.exit(1)

    results_count = 0; submitted_count = 0; error_count = 0
    # --- UPDATED COUNTER NAME ---
    url_unavailable_count = 0
    start_time = time.time(); end_time = start_time + test_duration_seconds
    client = None; url_iterator = None; outfile = None
    active_futures = set()

    try:
        log.info("Connecting to Dask scheduler...")
        client = Client(args.scheduler, timeout="90s", heartbeat_interval='15s')
        log.info(f"Successfully connected to scheduler."); log.info(f"Dask dashboard link: {client.dashboard_link}")
        workers_info = client.scheduler_info()['workers']; num_workers = len(workers_info)
        total_cores = sum(w.get('nthreads', 1) for w in workers_info.values())
        log.info(f"Cluster workers found: {num_workers}. Total Cores/Threads reported: {total_cores}")
        if total_cores != NUM_WORKER_CORES: log.warning(f"Expected {NUM_WORKER_CORES} cores, scheduler reports {total_cores}. Using reported value."); MAX_ACTIVE_FUTURES = total_cores + MAX_PENDING_BUFFER; log.info(f"Adjusted MAX_ACTIVE_FUTURES to: {MAX_ACTIVE_FUTURES}")
        if not workers_info: log.error("No workers connected! Exiting."); sys.exit(1)

        outfile = open(args.output, 'w')
        url_file = open(args.urls, 'r')
        url_iterator = iter(url_file)

        log.info(f"Opened Image URL file: {args.urls}"); log.info(f"Opened Output results file: {args.output}")
        outfile.write("ImageID,Caption\n")

        log.info(f"Starting main processing loop (Target Active Futures: {MAX_ACTIVE_FUTURES})...")
        loop_start_time = time.time()
        exit_submission_loop = False

        while time.time() < end_time:
            # Submission Phase
            while len(active_futures) < MAX_ACTIVE_FUTURES and not exit_submission_loop and time.time() < end_time:
                try:
                    image_url = next(url_iterator).strip()
                    if image_url:
                        future = client.submit(process_image, image_url, pure=False)
                        active_futures.add(future); submitted_count += 1
                    else: continue
                except StopIteration: log.warning("Reached end of Image URL file."); exit_submission_loop = True; break
                except Exception as e: log.error(f"Error reading URL file or submitting task: {e}", exc_info=True); time.sleep(1); exit_submission_loop = True; break

            # Result Processing Phase
            done_set = set()
            try:
                if active_futures:
                    done_set, _ = wait(active_futures, timeout=0.1, return_when='FIRST_COMPLETED')
            except TimeoutError: log.debug("No tasks completed within wait timeout."); pass
            except Exception as e: log.error(f"Error during dask.distributed.wait(): {e}", exc_info=True); time.sleep(1)

            processed_in_loop = 0
            for future in done_set:
                try:
                    result = future.result()
                    # --- UPDATED ERROR COUNTING ---
                    if result and result[0] is not None:
                        outfile.write(f"{result[0]},{result[1]}\n")
                        processed_in_loop += 1
                        prediction_result_str = str(result[1])
                        # Check for the specific "URL Unavailable" string first
                        if prediction_result_str == URL_UNAVAILABLE_ERROR:
                            url_unavailable_count += 1
                        elif "ERROR" in prediction_result_str: # Count other errors
                            error_count += 1
                        else: # Only count as success if no error
                            results_count += 1
                    elif result and result[0] is None:
                         log.warning(f"Task returned None ID with result: {result[1]}"); error_count += 1
                    # --- END UPDATED ERROR COUNTING ---
                except Exception as e:
                    task_exception = future.exception()
                    log.error(f"Failed to retrieve result for completed future {future.key}: {e}. Task exception: {task_exception}", exc_info=True)
                    error_count += 1 # Count failure to get result as other error
                finally:
                     active_futures.remove(future)

            if processed_in_loop > 0: outfile.flush()

            # Logging
            now = time.time()
            if now - loop_start_time > 30:
                elapsed_time = now - start_time
                rate = results_count / elapsed_time if elapsed_time > 0 else 0
                pending = len(active_futures)
                # --- UPDATED LOG MESSAGE ---
                log.info(f"Submitted: {submitted_count}, Success: {results_count} (URL Unavail: {url_unavailable_count}, Other Err: {error_count}), "
                         f"Active: {pending}, Rate: {rate:.2f} img/s, Elapsed: {elapsed_time/3600:.2f} hrs")
                # --- END UPDATED LOG MESSAGE ---
                loop_start_time = now

            # Prevent Tight Loop
            if (len(active_futures) >= MAX_ACTIVE_FUTURES or exit_submission_loop) and processed_in_loop == 0 and done_set == set():
                 log.debug(f"Limit reached or EOF, and no results processed. Sleeping briefly.")
                 time.sleep(0.1)

            # Exit Condition Check
            if exit_submission_loop and not active_futures:
                log.info("URL file finished and all submitted tasks completed. Exiting main loop.")
                break

        log.info("Test duration reached or processing finished. Finalizing...")

    except Exception as e: log.critical(f"A critical error occurred in the main script: {e}", exc_info=True)
    finally:
        if url_iterator is not None and hasattr(url_iterator, 'close'):
             if isinstance(url_iterator, io.IOBase) and not getattr(url_iterator, 'closed', True): log.info("Closing URL file."); url_iterator.close()
        if outfile is not None and not outfile.closed: log.info("Closing output file."); outfile.close()

        if client:
            if active_futures:
                log.info(f"Cancelling {len(active_futures)} outstanding futures...")
                for future in list(active_futures): future.cancel()
            log.info("Closing Dask client connection..."); client.close(); log.info("Dask client closed.")

        actual_duration = time.time() - start_time
        log.info("--- Test Summary ---")
        # --- UPDATED SUMMARY LOGGING ---
        log.info(f"Requested Duration: {args.duration:.2f} hours ({test_duration_seconds:.0f} s)")
        log.info(f"Actual Duration: {actual_duration / 3600:.2f} hours ({actual_duration:.0f} s)")
        log.info(f"Tasks Submitted: {submitted_count}")
        log.info(f"Successful Results: {results_count}")
        log.info(f"URL Unavailable Errors: {url_unavailable_count}") # New line
        log.info(f"Other Errors: {error_count}")
        # Adjust final pending calculation
        final_pending = submitted_count - results_count - error_count - url_unavailable_count
        log.info(f"Tasks Pending at End (Not Processed): {final_pending}")
        if actual_duration > 0:
            throughput = results_count / actual_duration
            log.info(f"Average Throughput (Successful Tasks): {throughput:.2f} images/sec ({throughput * 3600:.0f} images/hour)")
        # --- END UPDATED SUMMARY LOGGING ---
        log.info(f"Results saved to: {args.output}"); log.info(f"Log saved to: {args.log}"); log.info("--- Capacity test finished ---")