# Filename: run_capacity_test.py (BLIP Captioning - Model Load Per Task Version)
import time
import datetime
import argparse
import logging
import os
import random
import requests # For image download
from requests.exceptions import HTTPError # Specific exception
from PIL import Image # For image loading
import io # For handling image bytes
import sys
import re # Import regex for ID extraction
import traceback # For detailed error logging

# Dask libraries
from dask.distributed import Client, as_completed, Future, TimeoutError

# Required ML Libraries (ensure installed on workers)
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

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
# DEFAULT_DURATION_HOURS = 0.05 # Use a SHORT duration for testing!

DEFAULT_SCHEDULER = '127.0.0.1:8786'

# --- Logging Setup ---
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.FileHandler(DEFAULT_LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger('CapacityTestRunner')
dask_log = logging.getLogger('distributed')
dask_log.setLevel(logging.WARNING)


# --- Helper Function ---
def extract_id_from_url(url):
    """Extracts the image ID (filename without extension) from the URL."""
    if not isinstance(url, str): return None
    try:
        match = re.search(r'/([^/]+)\.(jpg|jpeg|png|gif)$', url, re.IGNORECASE)
        if match: return match.group(1)
        else:
            filename = url.split('/')[-1]
            return filename.split('.')[0] if '.' in filename else filename
    except Exception: return None

# --- Configuration specific to workers ---
DOWNLOAD_DIR_TEMPLATE = "/data/dask-worker-space/images/{image_id}.jpg" # Cache location

# --- Model Loading and Captioning Function (Runs on Worker) ---
def load_blip_and_caption(image_bytes):
    """Loads BLIP model and generates caption INSIDE the task function."""
    # Using print for worker logs as logging setup is complex inside tasks
    print("WORKER_INFO: Entering load_blip_and_caption")
    model_name = "Salesforce/blip-image-captioning-base"
    processor = None
    model = None
    try:
        print(f"WORKER_INFO: Importing transformers/torch/PIL within task...")
        # Re-import necessary libraries within the task function scope
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
        import io
        print("WORKER_INFO: Imports successful inside task.")

        # 1. Load Processor and Model (per task - inefficient)
        print(f"WORKER_INFO: Loading BLIP Processor: {model_name}")
        processor = BlipProcessor.from_pretrained(model_name)
        print(f"WORKER_INFO: Loading BLIP Model: {model_name}")
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        device = torch.device("cpu") # Force CPU
        model.to(device)
        model.eval() # Set to evaluation mode
        print("WORKER_INFO: BLIP Processor and Model loaded successfully.")

        # 2. Prepare Image
        print("WORKER_INFO: Preparing image...")
        raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 3. Process Image and Generate Caption
        print("WORKER_INFO: Processing image with BlipProcessor...")
        inputs = processor(raw_image, return_tensors="pt").to(device)

        print("WORKER_INFO: Generating caption (max_length=50)...")
        with torch.no_grad():
            # Adjust generation parameters if needed
            out = model.generate(**inputs, max_length=50, num_beams=4)

        print("WORKER_INFO: Decoding caption...")
        caption = processor.decode(out[0], skip_special_tokens=True)

        print(f"WORKER_INFO: Caption generated successfully: '{caption}'")
        # Clean up caption for CSV output
        return caption.replace("\n", " ").replace(",", ";").strip() # Replace newline and comma

    except Exception as e:
        print(f"WORKER_ERROR in load_blip_and_caption: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return f"ERROR_CAPTION_FAILED_{e.__class__.__name__}"
    finally:
         # Explicitly delete large objects to potentially help garbage collection
         # This might slightly reduce memory pressure between tasks on a worker
         del processor
         del model
         # No need for torch.cuda.empty_cache() on CPU


# --- Image Processing Wrapper Function (Sent to Workers) ---
def process_image(image_url):
    """
    Downloads the image and calls the function to load BLIP and generate caption.
    Returns tuple (image_id, caption_or_error).
    """
    t_start = time.time()
    image_url = image_url.strip()
    if not image_url: return (None, "ERROR_EMPTY_URL")

    image_id = extract_id_from_url(image_url)
    if not image_id:
        print(f"WORKER_ERROR: Failed ID extraction for URL: {image_url}", file=sys.stderr)
        return (image_url, "ERROR_ID_EXTRACTION_FAILED")

    local_path = DOWNLOAD_DIR_TEMPLATE.format(image_id=image_id)
    local_dir = os.path.dirname(local_path)

    try:
        # 1. Download Image (with simple caching)
        if not os.path.exists(local_path):
            os.makedirs(local_dir, exist_ok=True)
            # print(f"WORKER_INFO: Downloading {image_id} from {image_url}")
            response = requests.get(image_url, timeout=60) # Increased timeout slightly
            response.raise_for_status()
            image_bytes = response.content
            with open(local_path, 'wb') as f:
                f.write(image_bytes)
        else:
            # print(f"WORKER_INFO: Using cached image for {image_id}")
            with open(local_path, 'rb') as f:
                image_bytes = f.read()

        # 2. Load BLIP model and generate caption
        # print(f"WORKER_INFO: Calling load_blip_and_caption for {image_id}")
        caption_result = load_blip_and_caption(image_bytes) # Call the new function

        # print(f"WORKER_INFO: Finished processing {image_id}")
        return (image_id, caption_result) # Return tuple (ID, Caption/Error)

    except HTTPError as http_err:
        status_code = http_err.response.status_code if http_err.response else 'UNKNOWN'
        print(f"WORKER_WARN: Download failed for {image_id} from {image_url}: HTTP {status_code}", file=sys.stderr)
        return (image_id, f"ERROR_DOWNLOAD_HTTP_{status_code}")
    except requests.exceptions.RequestException as req_err:
        print(f"WORKER_WARN: Download failed for {image_id} from {image_url}: {req_err}", file=sys.stderr)
        return (image_id, f"ERROR_DOWNLOAD_{req_err.__class__.__name__}")
    except Exception as e:
        print(f"WORKER_ERROR in process_image wrapper for {image_id}: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return (image_id, f"ERROR_PROCESS_WRAPPER_{e.__class__.__name__}")


# --- Main Control Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dask Capacity Test (BLIP Captioning - Model Load Per Task)") # Updated title
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER, help="Dask scheduler address")
    parser.add_argument("--urls", default=DEFAULT_URL_FILE, help="Path to image URL list file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Path to output results CSV")
    parser.add_argument("--log", default=DEFAULT_LOG_FILE, help="Path to log file")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_HOURS, help="Test duration in hours")
    args = parser.parse_args()

    if args.log != DEFAULT_LOG_FILE:
        log.warning(f"Log file argument ignored after setup. Logging to: {DEFAULT_LOG_FILE}")

    test_duration_seconds = args.duration * 3600
    log.info(f"--- Starting Capacity Test (BLIP Captioning - Model Load Per Task) ---") # Updated title
    log.info(f"Duration: {args.duration:.2f} hours ({test_duration_seconds:.0f} seconds)")
    log.info(f"Scheduler: {args.scheduler}")
    log.info(f"Image URL File: {args.urls}")
    log.info(f"Output Results File: {args.output}")
    log.info(f"Log File: {args.log}")

    if not os.path.exists(args.urls):
        log.error(f"Image URL file not found: {args.urls}")
        sys.exit(1)

    results_count = 0
    submitted_count = 0
    error_count = 0
    start_time = time.time()
    end_time = start_time + test_duration_seconds
    client = None

    try:
        log.info("Connecting to Dask scheduler...")
        client = Client(args.scheduler, timeout="90s", heartbeat_interval='15s')
        log.info(f"Successfully connected to scheduler.")
        log.info(f"Dask dashboard link: {client.dashboard_link}")
        workers_info = client.scheduler_info()['workers']
        num_workers = len(workers_info)
        log.info(f"Cluster workers found: {num_workers}")
        if not workers_info:
             log.error("No workers connected to the scheduler! Exiting.")
             sys.exit(1)

        futures = as_completed()

        with open(args.urls, 'r') as url_file, open(args.output, 'w') as outfile:
            log.info(f"Opened Image URL file: {args.urls}")
            log.info(f"Opened Output results file: {args.output}")
            outfile.write("ImageID,Caption\n") # Updated CSV header

            log.info("Starting task submission loop...")
            while time.time() < end_time:
                submit_batch_size = 500
                current_batch = 0
                while current_batch < submit_batch_size and time.time() < end_time:
                    try:
                        image_url = next(url_file).strip()
                        if image_url:
                            future = client.submit(process_image, image_url, pure=False)
                            futures.add(future)
                            submitted_count += 1
                            current_batch += 1
                    except StopIteration:
                        log.warning("Reached end of Image URL file before duration ended.")
                        end_time = time.time(); break
                    except Exception as e:
                         log.error(f"Error reading URL file or submitting task: {e}", exc_info=True); time.sleep(1)

                if current_batch == 0 and time.time() >= end_time: break

                processed_in_batch = 0
                batch_start_time = time.time()
                max_batch_process_time = 1.0 # Process results for max 1 second

                for future in futures: # Iterate directly over as_completed
                    try:
                        if future.status == 'finished':
                           result = future.result(timeout=0.1)
                           if result and result[0] is not None:
                                outfile.write(f"{result[0]},{result[1]}\n") # Write ID, Caption/Error
                                if "ERROR" in str(result[1]): error_count += 1
                                results_count += 1
                                processed_in_batch += 1
                           elif result and result[0] is None:
                                log.warning(f"Task returned None ID with result: {result[1]}"); error_count += 1

                        elif future.status == 'error':
                           log.error(f"Task {future.key} failed with exception: {future.exception()}"); error_count +=1

                        if time.time() - batch_start_time > max_batch_process_time:
                             log.debug("Result processing time limit reached for this batch.")
                             break

                    except TimeoutError:
                         log.debug(f"Future {future.key} not ready within timeout, will check later.")
                    except Exception as e:
                        log.error(f"Failed to retrieve result for {future.key}: {e}", exc_info=True); error_count += 1

                # Calculate pending tasks *before* the backpressure check
                pending = submitted_count - results_count - error_count

                # Log progress
                if submitted_count % 1000 == 0 or processed_in_batch > 0: # Log slightly less often
                    elapsed_time = time.time() - start_time
                    rate = results_count / elapsed_time if elapsed_time > 0 else 0
                    log.info(f"Submitted: {submitted_count}, Results: {results_count} (Errors: {error_count}), "
                             f"Pending: {pending}, Rate: {rate:.2f} img/s, "
                             f"Elapsed: {elapsed_time/3600:.2f} hrs")

                # Basic backpressure
                if pending > (num_workers * 50): # Adjust threshold maybe lower for heavier tasks
                     sleep_time = 0.1 + (pending / (num_workers * 500))
                     log.debug(f"Backpressure active: {pending} pending tasks. Sleeping for {sleep_time:.2f}s")
                     time.sleep(min(sleep_time, 3.0)) # Allow slightly longer sleep


            log.info("Test duration reached or Image URL file ended. Stopping submission.")

            # --- Final Result Collection ---
            remaining_tasks = submitted_count - results_count - error_count
            log.info(f"Waiting for {remaining_tasks} remaining tasks...")
            for future in futures:
                try:
                    result = future.result(timeout=600) # Longer timeout for potentially slow final captions
                    if result and result[0] is not None:
                        outfile.write(f"{result[0]},{result[1]}\n")
                        if "ERROR" in str(result[1]): error_count += 1
                        results_count += 1
                    elif result and result[0] is None:
                         log.warning(f"Task returned None ID during final wait: {result[1]}"); error_count += 1
                except TimeoutError:
                     log.error(f"Timeout waiting for final task result for {future.key}."); error_count += 1
                except Exception as e:
                    log.error(f"Failed to retrieve result for {future.key} during final wait: {e}", exc_info=True); error_count += 1

                if (results_count + error_count) % 500 == 0: # Log progress less often
                     current_pending = submitted_count - results_count - error_count
                     if current_pending > 0:
                         log.info(f"Approx {current_pending} tasks remaining...")

            log.info("All tasks completed or failed.")

    except Exception as e:
        log.critical(f"A critical error occurred in the main script: {e}", exc_info=True)
    finally:
        if client:
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
        if actual_duration > 0:
             # Ensure results_count is used for throughput calculation
             throughput = results_count / actual_duration
             log.info(f"Average Throughput: {throughput:.2f} images/sec ({throughput * 3600:.0f} images/hour)")
        log.info(f"Results saved to: {args.output}")
        log.info(f"Log saved to: {args.log}")
        log.info("--- Capacity test finished ---")