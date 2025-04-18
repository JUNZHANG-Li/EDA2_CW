# Filename: run_capacity_test.py (Limit Active Futures <= 8)
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
# Use 'wait' for more controlled result checking
from dask.distributed import Client, Future, wait, TimeoutError, FIRST_COMPLETED

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
MAX_ACTIVE_FUTURES = 8 # Limit concurrent tasks submitted but not finished

# --- Logging Setup ---
# (Same as before)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', handlers=[logging.FileHandler(DEFAULT_LOG_FILE), logging.StreamHandler(sys.stdout)])
log = logging.getLogger('CapacityTestRunner')
dask_log = logging.getLogger('distributed')
dask_log.setLevel(logging.WARNING)

# --- Helper Function ---
# (extract_id_from_url - same as before)
def extract_id_from_url(url):
    if not isinstance(url, str): return None
    try:
        match = re.search(r'/([^/]+)\.(jpg|jpeg|png|gif)$', url, re.IGNORECASE)
        if match: return match.group(1)
        else:
            filename = url.split('/')[-1]
            return filename.split('.')[0] if '.' in filename else filename
    except Exception: return None

# --- Configuration specific to workers ---
DOWNLOAD_DIR_TEMPLATE = "/data/dask-worker-space/images/{image_id}.jpg"

# --- Model Loading and Captioning Function (Runs on Worker) ---
# (load_blip_and_caption - same as before)
def load_blip_and_caption(image_bytes):
    print("WORKER_INFO: Entering load_blip_and_caption")
    model_name = "Salesforce/blip-image-captioning-base"
    processor = None; model = None
    try:
        print(f"WORKER_INFO: Importing transformers/torch/PIL within task...")
        import torch; from transformers import BlipProcessor, BlipForConditionalGeneration; from PIL import Image; import io
        print("WORKER_INFO: Imports successful inside task.")
        print(f"WORKER_INFO: Loading BLIP Processor: {model_name}")
        processor = BlipProcessor.from_pretrained(model_name)
        print(f"WORKER_INFO: Loading BLIP Model: {model_name}")
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        device = torch.device("cpu"); model.to(device); model.eval()
        print("WORKER_INFO: BLIP Processor and Model loaded successfully.")
        print("WORKER_INFO: Preparing image...")
        raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print("WORKER_INFO: Processing image with BlipProcessor...")
        inputs = processor(raw_image, return_tensors="pt").to(device)
        print("WORKER_INFO: Generating caption (max_length=50)...")
        with torch.no_grad(): out = model.generate(**inputs, max_length=50, num_beams=4)
        print("WORKER_INFO: Decoding caption...")
        caption = processor.decode(out[0], skip_special_tokens=True)
        print(f"WORKER_INFO: Caption generated successfully: '{caption}'")
        return caption.replace("\n", " ").replace(",", ";").strip()
    except Exception as e:
        print(f"WORKER_ERROR in load_blip_and_caption: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return f"ERROR_CAPTION_FAILED_{e.__class__.__name__}"
    finally: del processor; del model

# --- Image Processing Wrapper Function (Sent to Workers) ---
# (process_image - same as before)
def process_image(image_url):
    t_start = time.time()
    image_url = image_url.strip()
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
    except HTTPError as http_err: status_code = http_err.response.status_code if http_err.response else 'UNKNOWN'; print(f"WORKER_WARN: Download failed for {image_id} from {image_url}: HTTP {status_code}", file=sys.stderr); return (image_id, f"ERROR_DOWNLOAD_HTTP_{status_code}")
    except requests.exceptions.RequestException as req_err: print(f"WORKER_WARN: Download failed for {image_id} from {image_url}: {req_err}", file=sys.stderr); return (image_id, f"ERROR_DOWNLOAD_{req_err.__class__.__name__}")
    except Exception as e: print(f"WORKER_ERROR in process_image wrapper for {image_id}: {e}\n{traceback.format_exc()}", file=sys.stderr); return (image_id, f"ERROR_PROCESS_WRAPPER_{e.__class__.__name__}")

# --- Main Control Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run Dask Capacity Test (BLIP Captioning - Active Limit: {MAX_ACTIVE_FUTURES})") # Updated title
    # (Argument parsing - same as before)
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER, help="Dask scheduler address")
    parser.add_argument("--urls", default=DEFAULT_URL_FILE, help="Path to image URL list file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Path to output results CSV")
    parser.add_argument("--log", default=DEFAULT_LOG_FILE, help="Path to log file")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_HOURS, help="Test duration in hours")
    args = parser.parse_args()


    if args.log != DEFAULT_LOG_FILE: log.warning(f"Logging to: {DEFAULT_LOG_FILE}")

    test_duration_seconds = args.duration * 3600
    log.info(f"--- Starting Capacity Test (BLIP Captioning - Active Limit: {MAX_ACTIVE_FUTURES}) ---") # Updated title
    # (Log other arguments - same as before)
    log.info(f"Duration: {args.duration:.2f} hours ({test_duration_seconds:.0f} seconds)"); log.info(f"Scheduler: {args.scheduler}"); log.info(f"Image URL File: {args.urls}"); log.info(f"Output Results File: {args.output}"); log.info(f"Log File: {args.log}")


    if not os.path.exists(args.urls): log.error(f"Image URL file not found: {args.urls}"); sys.exit(1)

    results_count = 0
    submitted_count = 0
    error_count = 0
    start_time = time.time()
    end_time = start_time + test_duration_seconds
    client = None
    url_iterator = None
    outfile = None # Initialize file handle

    # --- NEW: Use a set to track active futures ---
    active_futures = set()

    try:
        log.info("Connecting to Dask scheduler...")
        client = Client(args.scheduler, timeout="90s", heartbeat_interval='15s')
        log.info(f"Successfully connected to scheduler.")
        log.info(f"Dask dashboard link: {client.dashboard_link}")
        workers_info = client.scheduler_info()['workers']; num_workers = len(workers_info)
        log.info(f"Cluster workers found: {num_workers}")
        if not workers_info: log.error("No workers connected! Exiting."); sys.exit(1)

        outfile = open(args.output, 'w')
        url_file = open(args.urls, 'r')
        url_iterator = iter(url_file)

        log.info(f"Opened Image URL file: {args.urls}")
        log.info(f"Opened Output results file: {args.output}")
        outfile.write("ImageID,Caption\n")

        log.info(f"Starting main processing loop (Max Active Futures: {MAX_ACTIVE_FUTURES})...")
        loop_start_time = time.time() # For periodic logging

        while time.time() < end_time:
            # --- Submission Phase ---
            # Submit tasks until the active limit is reached or time runs out or file ends
            while len(active_futures) < MAX_ACTIVE_FUTURES and time.time() < end_time:
                try:
                    image_url = next(url_iterator).strip()
                    if image_url:
                        future = client.submit(process_image, image_url, pure=False)
                        active_futures.add(future) # Add future to the active set
                        submitted_count += 1
                    else: continue # Skip empty lines
                except StopIteration:
                    log.warning("Reached end of Image URL file.")
                    end_time = time.time() # Stop test now
                    break # Exit submission loop
                except Exception as e:
                     log.error(f"Error reading URL file or submitting task: {e}", exc_info=True)
                     time.sleep(1); break # Pause and break inner loop on error

            # --- Result Processing Phase ---
            # Check for completed futures without blocking if possible
            # Use wait() with timeout=0 to check non-blockingly
            if active_futures: # Only wait if there are active futures
                done_set, _ = wait(active_futures, timeout=0.1, return_when=FIRST_COMPLETED) # Short wait for *any* result
            else:
                done_set = set() # No futures to check

            processed_in_loop = 0
            for future in done_set: # Iterate through futures that completed
                try:
                    result = future.result() # Get result (should be ready)
                    if result and result[0] is not None:
                        outfile.write(f"{result[0]},{result[1]}\n")
                        if "ERROR" in str(result[1]): error_count += 1
                        results_count += 1
                        processed_in_loop += 1
                    elif result and result[0] is None:
                         log.warning(f"Task returned None ID with result: {result[1]}"); error_count += 1
                except Exception as e:
                    # Future completed but accessing result failed
                    log.error(f"Failed to retrieve result for completed future {future.key}: {e}", exc_info=True)
                    error_count += 1
                finally:
                     # IMPORTANT: Remove the completed future from the active set
                     active_futures.remove(future)

            # Flush output buffer periodically
            if processed_in_loop > 0:
                outfile.flush()

            # --- Logging ---
            now = time.time()
            if now - loop_start_time > 30: # Log approx every 30 seconds
                elapsed_time = now - start_time
                rate = results_count / elapsed_time if elapsed_time > 0 else 0
                pending = len(active_futures) # Get current pending count from set size
                log.info(f"Submitted: {submitted_count}, Results: {results_count} (Errors: {error_count}), "
                         f"Active: {pending}, Rate: {rate:.2f} img/s, "
                         f"Elapsed: {elapsed_time/3600:.2f} hrs")
                loop_start_time = now # Reset timer for next log interval

            # --- Prevent Tight Loop ---
            # If submission stopped (EOF or limit reached) and no results were processed, sleep briefly.
            if len(active_futures) >= MAX_ACTIVE_FUTURES and processed_in_loop == 0:
                log.debug(f"Active limit ({MAX_ACTIVE_FUTURES}) reached and no results processed. Sleeping.")
                time.sleep(0.1) # Short sleep to yield

        log.info("Test duration reached or Image URL file ended. Finalizing...")

        # --- NO FINAL WAIT LOOP ---
        log.info("Script will now exit. Any tasks still pending in Dask will be cancelled by client.close().")


    except Exception as e:
        log.critical(f"A critical error occurred in the main script: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        if url_iterator is not None and hasattr(url_iterator, 'close'):
             if isinstance(url_iterator, io.IOBase) and not url_iterator.closed:
                 log.info("Closing URL file.")
                 url_iterator.close()
        if outfile is not None and not outfile.closed:
             log.info("Closing output file.")
             outfile.close()

        if client:
            # Cancel remaining futures before closing (optional, helps cleanup)
            if active_futures:
                log.info(f"Cancelling {len(active_futures)} outstanding futures...")
                for future in list(active_futures): # Iterate over a copy
                     future.cancel()
            log.info("Closing Dask client connection...")
            client.close()
            log.info("Dask client closed.")

        actual_duration = time.time() - start_time
        log.info("--- Test Summary ---")
        log.info(f"Requested Duration: {args.duration:.2f} hours ({test_duration_seconds:.0f} s)")
        log.info(f"Actual Duration: {actual_duration / 3600:.2f} hours ({actual_duration:.0f} s)")
        log.info(f"Tasks Submitted: {submitted_count}")
        log.info(f"Results Collected: {results_count}")
        log.info(f"Errors Reported: {error_count}")
        # Final pending calculated based on counters
        final_pending = submitted_count - results_count - error_count
        log.info(f"Tasks Pending at End (Not Processed): {final_pending}")
        if actual_duration > 0:
             throughput = results_count / actual_duration
             log.info(f"Average Throughput (Processed Tasks): {throughput:.2f} images/sec ({throughput * 3600:.0f} images/hour)")
        log.info(f"Results saved to: {args.output}")
        log.info(f"Log saved to: {args.log}")
        log.info("--- Capacity test finished ---")