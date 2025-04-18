import time
import datetime
import argparse
import logging
import os
import random
import requests # For image download
from PIL import Image # For image loading
import io # For handling image bytes
import sys
import re # Import regex for ID extraction

# Dask libraries
from dask.distributed import Client, as_completed, Actor, Future

# --- Configuration ---
DEFAULT_BASE_DIR = "/opt/comp0239_coursework"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_BASE_DIR, "output")
DEFAULT_ID_FILE_NAME = "image_urls_to_process.txt" # Name is legacy, file now contains URLs
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
    if not isinstance(url, str):
        return None
    try:
        match = re.search(r'/([^/]+)\.(jpg|jpeg|png|gif)$', url, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            filename = url.split('/')[-1]
            return filename.split('.')[0] if '.' in filename else filename
    except Exception:
        return None

# --- Configuration specific to workers ---
DOWNLOAD_DIR_TEMPLATE = "/data/dask-worker-space/images/{image_id}.jpg"

# --- Dask Actor to Hold the Model (Instance Variable Version) ---
class ResNetModelActor:
    def __init__(self):
        actor_log = logging.getLogger('ResNetActor')
        actor_log.info("Initializing ResNetModelActor instance on worker...")
        self.model = None
        self.preprocess = None
        self.device = None
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            actor_log.info("Loading ResNet50 model into instance...")
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.model = models.resnet50(weights=weights)
            self.device = torch.device("cpu")
            self.model.to(self.device)
            self.model.eval()
            self.preprocess = weights.transforms()
            actor_log.info(f"ResNet50 model loaded successfully into instance (Device: {self.device}).")
        except Exception as e:
            actor_log.error(f"Failed to load model or transforms in __init__: {e}", exc_info=True)
            raise

    def predict(self, image_bytes):
        predict_log = logging.getLogger('ResNetActor.predict')
        if self.model is None or self.preprocess is None or self.device is None:
             predict_log.error("Model/preprocess/device not initialized in this instance!")
             return "ERROR_MODEL_NOT_LOADED"
        try:
            import torch
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            input_tensor = self.preprocess(img)
            input_batch = input_tensor.unsqueeze(0)
            input_batch = input_batch.to(self.device)
            with torch.no_grad():
                output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            return f"PRED_IDX_{top_catid.item()}"
        except Exception as e:
            predict_log.error(f"Error during prediction: {e}", exc_info=True)
            return "ERROR_PREDICTION_FAILED"


# --- Image Processing Function (Sent to Workers) ---
def process_image(image_url, model_actor_future):
    """
    Performs download (using full URL), preprocessing, and inference for a single image.
    Extracts ID for local caching and result reporting.
    Uses the ResNetModelActor for inference.
    """
    t_start = time.time()
    image_url = image_url.strip()
    if not image_url: return (None, "ERROR_EMPTY_URL")

    image_id = extract_id_from_url(image_url)
    if not image_id:
        # print(f"Worker ERROR: Could not extract ID from URL: {image_url}") # Use print/log if needed
        return (image_url, "ERROR_ID_EXTRACTION_FAILED") # Return URL if ID fails

    local_path = DOWNLOAD_DIR_TEMPLATE.format(image_id=image_id)
    local_dir = os.path.dirname(local_path)

    try:
        # 1. Download Image
        if not os.path.exists(local_path):
            os.makedirs(local_dir, exist_ok=True)
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
            with open(local_path, 'wb') as f:
                f.write(image_bytes)
        else:
            with open(local_path, 'rb') as f:
                image_bytes = f.read()

        # 2. Inference using the Model Actor
        model_actor = model_actor_future.result(timeout=60)
        prediction = model_actor.predict(image_bytes)

        return (image_id, prediction) # Return extracted ID and prediction

    except requests.exceptions.RequestException as e:
        # print(f"Worker WARN: Download failed for {image_id} from {image_url}: {e}")
        return (image_id, f"ERROR_DOWNLOAD_{e.__class__.__name__}")
    except Exception as e:
        # print(f"Worker ERROR: Processing failed for {image_id}: {e}")
        # import traceback
        # print(traceback.format_exc())
        return (image_id, f"ERROR_PROCESSING_{e.__class__.__name__}")


# --- Main Control Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dask Capacity Test for Image Classification using URLs")
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER, help="Dask scheduler address")
    parser.add_argument("--urls", default=DEFAULT_URL_FILE, help="Path to image URL list file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Path to output results CSV")
    parser.add_argument("--log", default=DEFAULT_LOG_FILE, help="Path to log file")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_HOURS, help="Test duration in hours")
    args = parser.parse_args()

    if args.log != DEFAULT_LOG_FILE:
        log.warning(f"Log file argument provided, but changing path after init is complex. Logging to: {DEFAULT_LOG_FILE}")

    test_duration_seconds = args.duration * 3600
    log.info(f"--- Starting Capacity Test ---")
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
        client = Client(args.scheduler, timeout="60s", heartbeat_interval='15s')
        log.info(f"Successfully connected to scheduler.")
        log.info(f"Dask dashboard link: {client.dashboard_link}")
        workers_info = client.scheduler_info()['workers']
        log.info(f"Cluster workers found: {len(workers_info)}")
        if not workers_info:
             log.error("No workers connected to the scheduler! Exiting.")
             sys.exit(1)

        log.info("Deploying ResNetModelActor to each worker...")
        model_actor_future = client.submit(ResNetModelActor, actor=True)
        model_actor_future.result(timeout=30)
        log.info("Model actor deployment task submitted/ready.")

        # --- FIX: Removed batch_size ---
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
                            future = client.submit(process_image, image_url, model_actor_future, pure=False)
                            futures.add(future)
                            submitted_count += 1
                            current_batch += 1
                    except StopIteration:
                        log.warning("Reached end of Image URL file before duration ended.")
                        end_time = time.time()
                        break
                    except Exception as e:
                         log.error(f"Error reading URL file or submitting task: {e}", exc_info=True)
                         time.sleep(1)

                if current_batch == 0 and time.time() >= end_time: break

                # --- FIX: Process results by iterating directly over as_completed ---
                processed_in_batch = 0
                # Iterate directly over the as_completed object
                for future in futures:
                    try:
                        # Check status before calling result (good practice)
                        if future.status == 'finished':
                            result = future.result(timeout=1) # Short timeout ok
                            if result and result[0] is not None:
                                outfile.write(f"{result[0]},{result[1]}\n")
                                if "ERROR" in str(result[1]): error_count += 1
                                results_count += 1
                                processed_in_batch += 1
                            elif result and result[0] is None:
                                 log.warning(f"Task returned None ID with result: {result[1]}")
                                 error_count += 1
                            # Remove processed future from the set
                            futures.discard(future)

                        elif future.status == 'error':
                           log.error(f"Task failed with exception: {future.exception()}")
                           error_count +=1
                           futures.discard(future) # Remove failed future

                        # Optional: Limit processing in one go to keep submitting
                        if processed_in_batch > 1000:
                            break # Exit inner loop to submit more if needed

                    except Future.TimeoutError:
                         log.warning("Timeout getting result from future.")
                         # Do not discard, will try again in next iteration
                    except Exception as e:
                        log.error(f"Failed to retrieve result: {e}", exc_info=True)
                        error_count += 1
                        # Remove failed future
                        futures.discard(future)

                # Log progress
                if submitted_count % 5000 == 0 or processed_in_batch > 0:
                    elapsed_time = time.time() - start_time
                    rate = results_count / elapsed_time if elapsed_time > 0 else 0
                    # --- FIX: Use len(futures) for pending count ---
                    pending = len(futures)
                    log.info(f"Submitted: {submitted_count}, Results: {results_count} (Errors: {error_count}), "
                             f"Pending: {pending}, Rate: {rate:.2f} img/s, "
                             f"Elapsed: {elapsed_time/3600:.2f} hrs")

                # Basic backpressure
                if len(futures) > (len(workers_info) * 100):
                     sleep_time = 0.1 + (len(futures) / (len(workers_info) * 1000))
                     time.sleep(min(sleep_time, 2.0))


            log.info("Test duration reached or Image URL file ended. Stopping submission.")

            # --- Final Result Collection ---
            remaining_tasks = len(futures)
            log.info(f"Waiting for {remaining_tasks} remaining tasks...")
            # Iterate directly over remaining futures
            for future in futures:
                try:
                    # Longer timeout for final tasks
                    result = future.result(timeout=300)
                    if result and result[0] is not None:
                        outfile.write(f"{result[0]},{result[1]}\n")
                        if "ERROR" in str(result[1]): error_count += 1
                        results_count += 1
                    elif result and result[0] is None:
                         log.warning(f"Task returned None ID during final wait: {result[1]}")
                         error_count += 1
                except Future.TimeoutError:
                     log.error("Timeout waiting for final task result.")
                     error_count += 1
                except Exception as e:
                    log.error(f"Failed to retrieve result during final wait: {e}", exc_info=True)
                    error_count += 1
                # No need to discard here as we are finishing

                remaining_tasks -= 1 # Simple counter decrement
                if remaining_tasks % 1000 == 0 and remaining_tasks > 0:
                    log.info(f"Waiting for {remaining_tasks} more tasks...")

            log.info("All tasks completed or failed.")

    except Exception as e:
        log.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        if client:
            log.info("Closing Dask client connection...")
            try:
                 if 'model_actor_future' in locals() and model_actor_future.done():
                      # Consider explicit actor deletion if needed/possible
                      # del model_actor_future or client.cancel([model_actor_future])
                      pass
            except Exception as actor_err:
                 log.warning(f"Error during actor cleanup: {actor_err}")
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