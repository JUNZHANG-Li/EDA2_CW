#!/usr/bin/env python3

import math
import queue
import threading
import ansible_runner
import json, subprocess
import tempfile

command = "terraform output --json worker_vm_ips".split()
ip_data = json.loads(subprocess.run(command, capture_output=True, encoding='UTF-8').stdout)

# Configuration
WORKERS = list(ip_data)
CHUNK_SIZE = 100
with open('4_pipeline_parser/.TOTAL_PDB_FILES') as f:
    TOTAL_PDB_FILES = int(f.readline().split()[0])

# Path to the analysis script that already exists on each worker.
# Ensure this script is present and executable on every worker:
REMOTE_SCRIPT_PATH  = "/home/almalinux/pipeline/pipeline_script_v1.py"
INPUT_DIR           = "/home/almalinux/input/"
OUTPUT_DIR          = "/home/almalinux/output/"

def generate_tasks(total_files, chunk_size):
    """
    Generates (start_index, end_index) tuples for each chunk.
    """
    tasks = []
    start = 1
    while start <= total_files:
        end = min(start + chunk_size - 1, total_files)
        tasks.append((start, end))
        start += chunk_size
    return tasks

def running_ansible(worker, cmd):
    """
    Executes a command on a remote worker node using Ansible Runner.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        ansible_runner.run(
            private_data_dir=temp_dir,
            inventory={
                'all': {
                    'hosts': {
                        worker: {
                            'ansible_host': worker,
                            'ansible_connection': 'ssh',
                        }
                    }
                }
            },
            host_pattern=worker,
            module='shell',
            module_args=cmd,
            quiet=True
        )

def analyse_chunk(worker, start_idx, end_idx):
    """
    Runs the remote analysis script on the specified worker using Ansible Runner.
    This example uses the 'shell' module, calling Python with the required
    start and end indices as arguments.
    """
    # Copy trunk of files to worker
    cmd = (
        f"sed -n '{start_idx},{end_idx}p' /home/almalinux/paths.txt | "
        f"xargs -I{{}} -n1 -P4 mc cp local/input/{{}} {INPUT_DIR}{{}}"
    )
    print(f"[INFO] Worker {WORKERS.index(worker)+1}: Assign files {start_idx}-{end_idx}.")
    running_ansible(worker, cmd)

    # merizo analysis
    cmd = f"python {REMOTE_SCRIPT_PATH}"
    # print(f"[INFO] Worker {WORKERS.index(worker)+1}: Analyse files {start_idx}-{end_idx}.")
    running_ansible(worker, cmd)

    # Upload results to minio and delete input/output files
    cmd = (
        f"mc cp --recursive {OUTPUT_DIR} local/output; "
        f"rm -r {INPUT_DIR}* {OUTPUT_DIR}*"
    )
    # print(f"[INFO] Worker {WORKERS.index(worker)+1}: Upload files {start_idx}-{end_idx}.")
    running_ansible(worker, cmd)


def worker_thread(worker, tasks_queue):
    """
    Thread function: continuously fetches new chunks from the queue
    and processes them until no chunks remain.
    """
    while True:
        try:
            start_idx, end_idx = tasks_queue.get_nowait()
        except queue.Empty:
            break

        analyse_chunk(worker, start_idx, end_idx)
        tasks_queue.task_done()

def main():
    # Create all the tasks in chunks of CHUNK_SIZE
    tasks = generate_tasks(TOTAL_PDB_FILES, CHUNK_SIZE)

    # Create a queue to hold all chunk tasks
    tasks_queue = queue.Queue()
    for t in tasks:
        tasks_queue.put(t)

    # Create one thread per worker
    threads = []
    for worker in WORKERS:
        t = threading.Thread(target=worker_thread, args=(worker, tasks_queue))
        t.start()
        threads.append(t)

    # Wait for all chunks to be processed
    tasks_queue.join()

    # Signal threads to exit
    for t in threads:
        t.join()

    print("[INFO] All PDB files have been analysed.")

if __name__ == "__main__":
    main()


