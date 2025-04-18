# Filename: run_capacity_test.py (Model Load Per Task Version - Fixed Result Loop)
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

# PyTorch/Torchvision - Primarily used within the task function now.
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
        # Regex to capture the filename part before common image extensions
        match = re.search(r'/([^/]+)\.(jpg|jpeg|png|gif)$', url, re.IGNORECASE)
        if match: return match.group(1)
        else:
            # Fallback attempt: simple split (less robust)
            filename = url.split('/')[-1]
            return filename.split('.')[0] if '.' in filename else filename
    except Exception: return None

# --- Configuration specific to workers ---
DOWNLOAD_DIR_TEMPLATE = "/data/dask-worker-space/images/{image_id}.jpg" # Cache location

# --- Model Loading and Prediction Function (Runs on Worker) ---
def load_and_predict(image_bytes):
    """Loads model and predicts INSIDE the task function."""
    # NOTE: Logging from worker tasks to a central file is complex.
    # Using print statements which might appear in worker logs (journalctl).
    print("WORKER_INFO: Entering load_and_predict")
    try:
        print("WORKER_INFO: Importing torch/torchvision within task...")
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
        from PIL import Image # Need PIL here too
        import io # Need io here too
        print("WORKER_INFO: Imports successful inside task.")

        print("WORKER_INFO: Loading model INSIDE task...")
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        preprocess = weights.transforms()
        print("WORKER_INFO: Model loaded INSIDE task.")

        print("WORKER_INFO: Preprocessing image...")
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)

        print("WORKER_INFO: Running inference...")
        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        pred_idx = top_catid.item()
        print("WORKER_INFO: Prediction successful INSIDE task.")
        return f"PRED_IDX_{pred_idx}"

    except Exception as e:
        # Print error details to worker's stdout/stderr
        print(f"WORKER_ERROR in load_and_predict: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return f"ERROR_TASK_FAILED_{e.__class__.__name__}"


# --- Image Processing Wrapper Function (Sent to Workers) ---
def process_image(image_url):
    """
    Downloads the image and calls the function to load the model and predict.
    Returns tuple (image_id, prediction_result).
    """
    t_start = time.time()
    image_url = image_url.strip()
    if not image_url: return (None, "ERROR_EMPTY_URL")

    image_id = extract_id_from_url(image_url)
    if not image_id:
        # Cannot proceed without an ID for caching/results
        print(f"WORKER_ERROR: Failed ID extraction for URL: {image_url}", file=sys.stderr)
        return (image_url, "ERROR_ID_EXTRACTION_FAILED") # Return URL itself as ID placeholder

    local_path = DOWNLOAD_DIR_TEMPLATE.format(image_id=image_id)
    local_dir = os.path.dirname(local_path)

    try:
        # 1. Download Image (with simple caching)
        if not os.path.exists(local_path):
            # Ensure directory exists (needed if first time for this worker)
            os.makedirs(local_dir, exist_ok=True)
            # print(f"WORKER_INFO: Downloading {image_id} from {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            image_bytes = response.content
            with open(local_path, 'wb') as f:
                f.write(image_bytes)
        else:
            # print(f"WORKER_INFO: Using cached image for {image_id}")
            with open(local_path, 'rb') as f:
                image_bytes = f.read()

        # 2. Load model and predict by calling the other function
        # print(f"WORKER_INFO: Calling load_and_predict for {image_id}")
        prediction_result = load_and_predict(image_bytes)

        # print(f"WORKER_INFO: Finished processing {image_id}")
        return (image_id, prediction_result) # Return tuple (ID, Result)

    except HTTPError as http_err: # Catch HTTPError specifically
        status_code = http_err.response.status_code if http_err.response else 'UNKNOWN'
        print(f"WORKER_WARN: Download failed for {image_id} from {image_url}: HTTP {status_code}", file=sys.stderr)
        return (image_id, f"ERROR_DOWNLOAD_HTTP_{status_code}")
    except requests.exceptions.RequestException as req_err: # Catch other request errors
        print(f"WORKER_WARN: Download failed for {image_id} from {image_url}: {req_err}", file=sys.stderr)
        return (image_id, f"ERROR_DOWNLOAD_{req_err.__class__.__name__}")
    except Exception as e:
        # Catch other errors (e.g., disk write, unexpected issues before prediction call)
        print(f"WORKER_ERROR in process_image wrapper for {image_id}: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return (image_id, f"ERROR_PROCESS_WRAPPER_{e.__class__.__name__}")


# --- Main Control Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dask Capacity Test (Model Load Per Task)")
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER, help="Dask scheduler address")
    parser.add_argument("--urls", default=DEFAULT_URL_FILE, help="Path to image URL list file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Path to output results CSV")
    parser.add_argument("--log", default=DEFAULT_LOG_FILE, help="Path to log file")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_HOURS, help="Test duration in hours")
    args = parser.parse_args()

    # --- Update logging file path ---
    if args.log != DEFAULT_LOG_FILE:
        log.warning(f"Log file argument ignored after setup. Logging to: {DEFAULT_LOG_FILE}")

    test_duration_seconds = args.duration * 3600
    log.info(f"--- Starting Capacity Test (Model Load Per Task) ---")
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
        log.info(f"Cluster workers found: {len(workers_info)}")
        if not workers_info:
             log.error("No workers connected to the scheduler! Exiting.")
             sys.exit(1)
        num_workers = len(workers_info)

        # --- Actor Deployment is Removed ---

        futures = as_completed()

        with open(args.urls, 'r') as url_file, open(args.output, 'w') as outfile:
            log.info(f"Opened Image URL file: {args.urls}")
            log.info(f"Opened Output results file: {args.output}")
            outfile.write("ImageID,Prediction\n")

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

                # *** --- CORRECTED RESULT PROCESSING LOOP --- ***
                processed_in_batch = 0
                # Iterate directly over the as_completed iterator
                # Process results that are ready within a short time window
                batch_start_time = time.time()
                max_batch_process_time = 1.0 # Process results for max 1 second per submission batch

                for future in futures: # Iterate directly over as_completed
                    try:
                        # Check future status without blocking indefinitely initially
                        if future.status == 'finished':
                           # Use a small timeout as it should be ready
                           result = future.result(timeout=0.1)
                           # Result is expected tuple: (image_id, prediction_result)
                           if result and result[0] is not None:
                                outfile.write(f"{result[0]},{result[1]}\n")
                                if "ERROR" in str(result[1]): error_count += 1
                                results_count += 1
                                processed_in_batch += 1
                           elif result and result[0] is None:
                                # Should only happen if input URL was empty or ID extraction failed
                                log.warning(f"Task returned None/URL as ID with result: {result[1]}")
                                error_count += 1
                           # Remove future from set (as_completed handles this implicitly)

                        elif future.status == 'error':
                            log.error(f"Task {future.key} failed with exception: {future.exception()}")
                            error_count += 1
                            # Remove future from set (as_completed handles this implicitly)

                        # Break processing loop if we spend too long here to avoid blocking submission
                        if time.time() - batch_start_time > max_batch_process_time:
                             log.debug("Result processing time limit reached for this batch.")
                             break

                    except TimeoutError:
                         # Future wasn't ready within the short timeout,
                         # leave it in the as_completed set and try again later.
                         log.debug(f"Future {future.key} not ready within timeout, will check later.")
                    except Exception as e:
                        log.error(f"Failed to retrieve result for {future.key}: {e}", exc_info=True)
                        error_count += 1
                        # Remove future from set (as_completed handles this implicitly)
                # *** --- END CORRECTED RESULT PROCESSING LOOP --- ***

                # Log progress
                if submitted_count % 5000 == 0 or processed_in_batch > 0:
                    elapsed_time = time.time() - start_time
                    rate = results_count / elapsed_time if elapsed_time > 0 else 0
                    pending = submitted_count - results_count - error_count
                    log.info(f"Submitted: {submitted_count}, Results: {results_count} (Errors: {error_count}), "
                             f"Pending: {pending}, Rate: {rate:.2f} img/s, "
                             f"Elapsed: {elapsed_time/3600:.2f} hrs")

                # Basic backpressure
                current_pending = len(futures) # Check actual outstanding futures
                if current_pending > (num_workers * 100):
                     sleep_time = 0.1 + (current_pending / (num_workers * 1000))
                     time.sleep(min(sleep_time, 2.0))


            log.info("Test duration reached or Image URL file ended. Stopping submission.")

            # --- Final Result Collection ---
            remaining_tasks = len(futures)
            log.info(f"Waiting for {remaining_tasks} remaining tasks...")
            for future in futures: # Iterate over remaining futures
                try:
                    result = future.result(timeout=300) # Longer timeout
                    if result and result[0] is not None:
                        outfile.write(f"{result[0]},{result[1]}\n")
                        if "ERROR" in str(result[1]): error_count += 1
                        results_count += 1
                    elif result and result[0] is None:
                         log.warning(f"Task returned None/URL as ID during final wait: {result[1]}"); error_count += 1
                except TimeoutError:
                     log.error(f"Timeout waiting for final task result for {future.key}."); error_count += 1
                except Exception as e:
                    log.error(f"Failed to retrieve result for {future.key} during final wait: {e}", exc_info=True); error_count += 1

                remaining_tasks -= 1
                if remaining_tasks % 1000 == 0 and remaining_tasks > 0:
                    log.info(f"Waiting for {remaining_tasks} more tasks...")

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
             throughput = results_count / actual_duration
             log.info(f"Average Throughput: {throughput:.2f} images/sec ({throughput * 3600:.0f} images/hour)")
        log.info(f"Results saved to: {args.output}")
        log.info(f"Log saved to: {args.log}")
        log.info("--- Capacity test finished ---")