            # --- Result Processing Phase ---
            done_set = set() # Initialize as empty set
            try:
                if active_futures: # Only wait if there are active futures
                    # Wait briefly for *any* result
                    done_set, _ = wait(active_futures, timeout=0.1, return_when='FIRST_COMPLETED')
            except TimeoutError:
                # This is EXPECTED if no tasks finished within the short timeout
                log.debug("No tasks completed within wait timeout.")
                pass # Continue the main loop, will check again next iteration
            except Exception as e:
                # Catch other potential errors from wait() itself
                log.error(f"Error during dask.distributed.wait(): {e}", exc_info=True)
                # Decide how to handle - maybe break or continue cautiously?
                time.sleep(1) # Pause if wait fails unexpectedly

            processed_in_loop = 0
            # --- Iterate over the done_set (which might be empty) ---
            for future in done_set:
                try:
                    # Future in done_set should be ready, get result without timeout
                    result = future.result()
                    if result and result[0] is not None:
                        outfile.write(f"{result[0]},{result[1]}\n")
                        if "ERROR" in str(result[1]): error_count += 1
                        results_count += 1
                        processed_in_loop += 1
                    elif result and result[0] is None:
                         log.warning(f"Task returned None ID with result: {result[1]}"); error_count += 1
                except Exception as e:
                    # Future was 'done' but getting result failed (e.g., task raised exception)
                    # Log the exception associated with the future if possible
                    task_exception = future.exception()
                    log.error(f"Failed to retrieve result for completed future {future.key}: {e}. Task exception: {task_exception}", exc_info=True)
                    error_count += 1
                finally:
                     # IMPORTANT: Remove the processed future from the active set
                     active_futures.remove(future)

            # (Rest of the loop: flush, logging, sleep - remains the same)
            # ...