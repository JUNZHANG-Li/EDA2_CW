import sys
from subprocess import Popen, PIPE
import glob
import subprocess
import multiprocessing
import logging
import os
from pathlib import Path

# Set up primary logging
logging.basicConfig(
    filename='/home/almalinux/data/detail.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    # filemode='w'
)

# Set up secondary logging
clean_log_file = '/home/almalinux/data/clean.log'
clean_logger = logging.getLogger('clean_logger')
clean_logger.setLevel(logging.INFO)
# clean_handler = logging.FileHandler(clean_log_file, mode='w')
clean_handler = logging.FileHandler(clean_log_file)
clean_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
clean_logger.addHandler(clean_handler)

def run_parser(input_file):
    """
    Runs the results_parser_v1.py script on the *_search.tsv file and
    logs output or errors accordingly. Returns True if there is an error.
    """
    search_file = input_file + "_search.tsv"
    cmd = ['python', '/home/almalinux/pipeline/results_parser_v1.py', search_file]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    if err:
        return True
    logging.info(f'STEP 2: RUNNING PARSER: {" ".join(cmd)}')
    logging.info(out.decode("utf-8"))
    return False

def move_files(input_file, skip, output_dir):
    """
    Moves files into the specified output directory.
    If skip is False, moves the .parsed, _search.tsv, and _segment.tsv files.
    If skip is True, moves only the _segment.tsv file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if not skip:
        files_to_move = [f"{input_file}.parsed", f"{input_file}_search.tsv", f"{input_file}_segment.tsv"]
    else:
        files_to_move = [f"{input_file}_segment.tsv"]

    for file in files_to_move:
        result = subprocess.run(["mv", file, output_dir],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            logging.info(f"Moved {file} to {output_dir}")
        else:
            logging.error(f"Failed to move {file}: {result.stderr}")

def run_merizo_search(input_file, id):
    """
    Runs merizo.py to perform an 'easy-search' on the PDB file.
    """
    cmd = [
        'python3',
        '/home/almalinux/merizo_search/merizo_search/merizo.py',
        'easy-search',
        input_file,
        '/home/almalinux/data/cath_foldclassdb/cath-4.3-foldclassdb',
        id,
        'tmp',
        '--iterate',
        '--output_headers',
        '-d',
        'cpu',
        '--threads',
        '1'
    ]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    logging.info(out.decode("utf-8"))
    if err:
        logging.info(err.decode("utf-8"))

def update_progress():
    """
    Calls the progress.py script to update any progress indicators as needed.
    """
    cmd = ['python3','/home/almalinux/pipeline/progress.py']
    Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

def read_dir(base_dir):
    """
    Recursively reads the base directory, returning a list of (pdb_path, id, relative_subdir).
    - pdb_path is the full path of the .pdb file.
    - id is the file stem (filename without extension).
    - relative_subdir is the path of the containing folder relative to the base directory.
    """
    logging.info(f'Reading directory {base_dir} (including subdirectories)')
    pdb_list = []

    # Use pathlib to recursively find *.pdb files
    for pdb_file in Path(base_dir).rglob('*.pdb'):
        # 'id' is filename without extension
        pdb_id = pdb_file.stem
        # subdir relative to base_dir (e.g. "ecoli" if file is /home/almalinux/input/ecoli/abc.pdb)
        relative_subdir = str(pdb_file.parent.relative_to(base_dir))
        pdb_list.append((str(pdb_file), pdb_id, relative_subdir))

    return pdb_list

def pipeline(filepath, pdb_id, relative_subdir):
    """
    Runs the full pipeline on a single PDB file:
    1. run_merizo_search
    2. run_parser
    3. move_files
    4. log results and update progress
    """
    # Construct the output directory based on the subdirectory structure
    output_dir = os.path.join('/home/almalinux/output', relative_subdir)

    # Run the Merizo search
    run_merizo_search(filepath, pdb_id)

    # Attempt to parse the results
    skip = run_parser(pdb_id)

    # Move the resulting files to the correct output directory
    move_files(pdb_id, skip, output_dir)

    # Log in detail.log and clean.log
    if not skip:
        logging.info(f'-------------Finished {pdb_id}-------------\n')
        clean_logger.info(pdb_id)
    else:
        logging.error(f'-------------Skipped: Failed to segment {pdb_id}------------\n')
        clean_logger.error(pdb_id)

    # Update any progress indicators
    update_progress()

if __name__ == "__main__":
    # We simply fix the input base directory to /home/almalinux/input
    input_dir = '/home/almalinux/input'

    # Gather all PDB files (including those in subdirectories)
    pdbfiles = read_dir(input_dir)

    # Use a Pool to process them in parallel
    with multiprocessing.Pool(4) as p:
        p.starmap(pipeline, pdbfiles)
