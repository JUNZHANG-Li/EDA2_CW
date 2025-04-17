# Filename: run_capacity_test.py
import time
import datetime
import argparse
import logging
import os
import random
import requests # For image download example
from PIL import Image # For image loading example
import io # For handling image bytes
import sys

# Dask libraries
from dask.distributed import Client, as_completed, Actor, Future # Import Future

# --- Configuration ---
# These paths are relative to where the script is run ON THE HOST NODE
# Ansible playbook places scripts in /opt/scripts and output in /opt/output
DEFAULT_ID_FILE = '/opt/output/image_ids_to_process.txt'
DEFAULT_OUTPUT_FILE = '/opt/output/capacity_test_results.csv'
DEFAULT_LOG_FILE = '/opt/output/capacity_test.log'
DEFAULT_DURATION_HOURS = 24 # Default runtime
# DEFAULT_DURATION_HOURS = 0.1 # Use a SHORT duration for testing!

DEFAULT_SCHEDULER = '127.0.0.1:8786' # Assumes running on the host node

# --- Logging Setup ---
# Ensure output directory exists before setting up file handler
os.makedirs(os.path.dirname(DEFAULT_LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.FileHandler(DEFAULT_LOG_FILE), # Log to file in output dir
        logging.StreamHandler(sys.stdout) # Also log to console
    ]
)
log = logging.getLogger('CapacityTestRunner')
dask_log = logging.getLogger('distributed')
dask_log.setLevel(logging.WARNING)

# --- Global Variables / Placeholders ---
# !! CRITICAL: VERIFY AND CORRECT IMAGE URL AND SPLIT LOGIC !!
IMAGE_URL_TEMPLATE = "https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg"
DOWNLOAD_DIR_TEMPLATE = "/data/dask-worker-space/images/{image_id}.jpg" # Uses worker's data disk

# --- Dask Actor to Hold the Model ---
class ResNetModelActor:
    # Class variable to hold the model, ensures loading once per worker process
    _model = None
    _preprocess = None
    _device = None # Add device tracking

    def __init__(self):
        actor_log = logging.getLogger('ResNetActor') # Use logger
        if ResNetModelActor._model is None:
            actor_log.info("Initializing ResNetModelActor on worker...")
            try:
                import torch
                import torchvision.models as models
                import torchvision.transforms as transforms
                
                actor_log.info("Loading ResNet50 model...")
                # Use recommended weights argument
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                ResNetModelActor._model = models.resnet50(weights=weights)
                ResNetModelActor._device = torch.device("cpu") # Force CPU
                ResNetModelActor._model.to(ResNetModelActor._device)
                ResNetModelActor._model.eval() # Set model to evaluation mode

                # Use transforms associated with the weights
                ResNetModelActor._preprocess = weights.transforms()
                
                actor_log.info(f"ResNet50 model loaded successfully on worker (Device: {ResNetModelActor._device}).")
            except Exception as e:
                actor_log.error(f"Failed to load model or transforms on worker: {e}", exc_info=True)
                raise
        else:
             actor_log.info("ResNetModelActor already initialized on this worker.")
             
    def predict(self, image_bytes):
        """Performs preprocessing and inference on image bytes."""
        predict_log = logging.getLogger('ResNetActor.predict')
        if self._model is None or self._preprocess is None or self._device is None:
             predict_log.error("Model/preprocess/device not initialized!")
             return "ERROR_MODEL_NOT_LOADED"
        
        try:
            import torch # Ensure torch is available in this method context
            
            # Open image and convert to RGB
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Apply preprocessing
            input_tensor = self._preprocess(img)
            input_batch = input_tensor.unsqueeze(0) # Create a mini-batch

            # Move tensor to the correct device (CPU)
            input_batch = input_batch.to(self._device)

            with torch.no_grad():
                output = self._model(input_batch)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            
            # Return predicted class index as string
            return f"PRED_IDX_{top_catid.item()}" 

        except Exception as e:
            predict_log.error(f"Error during prediction: {e}", exc_info=True)
            return "ERROR_PREDICTION_FAILED"

# --- Image Processing Function (Sent to Workers) ---
def process_image(image_id, model_actor_future):
    """
    Performs download, preprocessing, and inference for a single image ID.
    Uses the ResNetModelActor for inference.
    """
    # Setting up logging within the function is tricky with Dask serialization.
    # It's often better to rely on print statements or return detailed error strings.
    # worker_log = logging.getLogger('WorkerTask') # May not work reliably
    # worker_log.info(f"Processing {image_id}")

    t_start = time.time()
    image_id = image_id.strip()
    if not image_id: return (None, "ERROR_EMPTY_ID")

    # !! CRITICAL: Determine the correct 'split' (train/validation/test etc.) !!
    # This might require parsing the image_id if it contains the split,
    # or looking it up from metadata if your ID list doesn't include it.
    split = "train" # HARDCODED GUESS - MUST BE CORRECTED
    image_url = IMAGE_URL_TEMPLATE.format(split=split, image_id=image_id)
    local_path = DOWNLOAD_DIR_TEMPLATE.format(image_id=image_id)
    local_dir = os.path.dirname(local_path)

    try:
        # 1. Download Image (with simple caching)
        if not os.path.exists(local_path):
            os.makedirs(local_dir, exist_ok=True)
            response = requests.get(image_url, timeout=30) # Longer timeout for S3
            response.raise_for_status()
            image_bytes = response.content
            with open(local_path, 'wb') as f:
                f.write(image_bytes)
        else:
            with open(local_path, 'rb') as f:
                image_bytes = f.read()

        # 2. Inference using the Model Actor
        # Retrieve the actor instance on this worker
        model_actor = model_actor_future.result(timeout=60) # Add timeout for safety
        prediction = model_actor.predict(image_bytes)

        return (image_id, prediction)

    except requests.exceptions.RequestException as e:
        # Log download errors specifically
        # print(f"Worker WARN: Download failed for {image_id} from {image_url}: {e}") # Use print for worker logs
        return (image_id, f"ERROR_DOWNLOAD_{e.__class__.__name__}")
    except Exception as e:
        # Catch-all for other processing errors
        # print(f"Worker ERROR: Processing failed for {image_id}: {e}") # Use print for worker logs
        import traceback
        # print(traceback.format_exc()) # Print stack trace on worker stdoud/stderr
        return (image_id, f"ERROR_PROCESSING_{e.__class__.__name__}")

# --- Main Control Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dask Capacity Test for Image Classification")
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER, help="Dask scheduler address")
    parser.add_argument("--ids", default=DEFAULT_ID_FILE, help="Path to image ID list file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Path to output results CSV")
    parser.add_argument("--log", default=DEFAULT_LOG_FILE, help="Path to log file")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_HOURS, help="Test duration in hours")
    args = parser.parse_args()

    # --- Update logging file path if provided ---
    if args.log != DEFAULT_LOG_FILE:
        # Find the FileHandler and update its path
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close() # Close existing file
                handler.baseFilename = args.log # Set new path
                # Need way to reopen/recreate handler if necessary, complex.
                # Easier to just set path before basicConfig, but argparse is after.
                # For simplicity here, we assume default or pre-created path.
        log.warning(f"Log file argument provided, but changing path after init is complex. Logging to: {DEFAULT_LOG_FILE}")


    test_duration_seconds = args.duration * 3600
    log.info(f"--- Starting Capacity Test ---")
    log.info(f"Duration: {args.duration:.2f} hours ({test_duration_seconds:.0f} seconds)")
    log.info(f"Scheduler: {args.scheduler}")
    log.info(f"Image ID File: {args.ids}")
    log.info(f"Output Results File: {args.output}")
    log.info(f"Log File: {args.log}") # Use the default path for logging

    if not os.path.exists(args.ids):
        log.error(f"Image ID file not found: {args.ids}")
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
        # Submit actor creation task, get a future representing the actor's location(s)
        model_actor_future = client.submit(ResNetModelActor, actor=True)
        # Wait briefly for the actor future to be ready (doesn't mean model is loaded yet)
        model_actor_future.result(timeout=30) 
        log.info("Model actor deployment task submitted/ready.")
        log.info("NOTE: Actual model loading happens lazily on first use per worker.")


        futures = as_completed(batch_size=2000) # Process results in larger batches

        with open(args.ids, 'r') as id_file, open(args.output, 'w') as outfile:
            log.info(f"Opened Image ID file: {args.ids}")
            log.info(f"Opened Output results file: {args.output}")
            outfile.write("ImageID,Prediction\n") # CSV header

            log.info("Starting task submission loop...")
            while time.time() < end_time:
                submit_batch_size = 500 # Submit tasks in batches
                current_batch = 0
                while current_batch < submit_batch_size and time.time() < end_time:
                    try:
                        image_id = next(id_file).strip()
                        if image_id:
                            # Pass the actor *future* to the task
                            future = client.submit(process_image, image_id, model_actor_future, pure=False) # pure=False often needed
                            futures.add(future)
                            submitted_count += 1
                            current_batch += 1
                    except StopIteration:
                        log.warning("Reached end of Image ID file before duration ended.")
                        end_time = time.time() # Stop test now
                        break
                    except Exception as e:
                         log.error(f"Error reading ID file or submitting task: {e}", exc_info=True)
                         time.sleep(1) # Pause before retry

                if current_batch == 0 and time.time() >= end_time: break # Exit outer loop

                # Process completed results efficiently
                processed_in_batch = 0
                # Iterate through readily available completed futures
                completed_iterator = futures.fast_iterator() 
                for future in completed_iterator:
                    try:
                        result = future.result(timeout=1) # Short timeout as it should be done
                        if result and result[0] is not None: # Check if ID is valid
                            outfile.write(f"{result[0]},{result[1]}\n")
                            if "ERROR" in str(result[1]): error_count += 1
                            results_count += 1
                            processed_in_batch += 1
                        elif result and result[0] is None:
                             log.warning(f"Task returned None ID with result: {result[1]}")
                             error_count += 1
                    except Future.TimeoutError:
                         log.warning("Timeout getting result from completed future (should be rare).")
                         futures.add(future) # Re-add to check later
                    except Exception as e:
                        log.error(f"Failed to retrieve result: {e}", exc_info=True)
                        error_count += 1
                        # Can try future.key to identify task if needed for debugging

                # Log progress
                if submitted_count % 5000 == 0 or processed_in_batch > 0:
                    elapsed_time = time.time() - start_time
                    rate = results_count / elapsed_time if elapsed_time > 0 else 0
                    pending = submitted_count - results_count - error_count
                    log.info(f"Submitted: {submitted_count}, Results: {results_count} (Errors: {error_count}), "
                             f"Pending: {pending}, Rate: {rate:.2f} img/s, "
                             f"Elapsed: {elapsed_time/3600:.2f} hrs")

                # Basic backpressure: slow down if too many tasks are queued
                if len(futures) > (len(workers_info) * 100): # e.g., > 100 tasks per worker pending
                     sleep_time = 0.1 + (len(futures) / (len(workers_info) * 1000)) # Dynamic sleep
                     time.sleep(min(sleep_time, 2.0)) # Max 2s sleep


            log.info("Test duration reached or Image ID file ended. Stopping submission.")

            # --- Final Result Collection ---
            remaining_tasks = len(futures)
            log.info(f"Waiting for {remaining_tasks} remaining tasks...")
            # Use the main iterator now which blocks
            for future in futures:
                try:
                    result = future.result(timeout=300) # Longer timeout for final tasks
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

                remaining_tasks -= 1
                if remaining_tasks % 1000 == 0 and remaining_tasks > 0:
                    log.info(f"Waiting for {remaining_tasks} more tasks...")

            log.info("All tasks completed or failed.")

    except Exception as e:
        log.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        if client:
            log.info("Closing Dask client connection...")
            try:
                 # Attempt graceful shutdown of actors if needed (complex)
                 if 'model_actor_future' in locals() and model_actor_future.done():
                      pass # client.cancel might be possible or actor specific cleanup
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