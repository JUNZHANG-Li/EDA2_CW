# Filename: generate_id_list.py
import pandas as pd
import os
import argparse
import glob # To find parquet files
import sys

# --- Configuration ---
IMAGE_ID_COLUMN_NAME = 'image_id' # !! VERIFY THIS in your Parquet file !!

def generate_list_from_parquet(metadata_dir, id_column, output_file, target_count):
    """Loads IDs from Parquet files in a directory and writes a subset to the output file."""
    
    parquet_files = glob.glob(os.path.join(metadata_dir, '*.parquet'))
    
    if not parquet_files:
        print(f"Error: No Parquet files found in directory: {metadata_dir}", file=sys.stderr)
        return False

    print(f"Found Parquet files: {parquet_files}")
    
    all_ids_list = []
    total_loaded = 0

    try:
        print("Loading image IDs from Parquet files...")
        for file_path in parquet_files:
            print(f"  Reading {file_path}...")
            df_part = pd.read_parquet(file_path, columns=[id_column])
            print(f"    Read {len(df_part)} IDs.")
            
            if id_column not in df_part.columns:
                print(f"Error: Column '{id_column}' not found in {file_path}. Found columns: {df_part.columns.tolist()}", file=sys.stderr)
                return False
                
            all_ids_list.append(df_part[id_column])
            total_loaded += len(df_part)
            del df_part 

        print(f"Loaded a total of {total_loaded} IDs (potentially with duplicates).")

        if not all_ids_list:
             print("Error: No IDs were loaded.", file=sys.stderr)
             return False
             
        all_ids_series = pd.concat(all_ids_list, ignore_index=True)
        del all_ids_list 

        print("Finding unique IDs...")
        unique_ids = all_ids_series.drop_duplicates()
        print(f"Found {len(unique_ids)} unique IDs.")
        del all_ids_series

        if len(unique_ids) >= target_count:
            selected_ids = unique_ids.head(target_count).tolist()
            print(f"Selected first {len(selected_ids)} unique ImageIDs.")
        else:
            selected_ids = unique_ids.tolist()
            print(f"Warning: Found only {len(selected_ids)} unique ImageIDs, less than target {target_count}.")
            print("Using all available unique IDs.")
        
        del unique_ids 

        print(f"Writing {len(selected_ids)} IDs to {output_file}...")
        count = 0
        with open(output_file, 'w') as f:
            for img_id in selected_ids:
                f.write(str(img_id).strip() + '\n')
                count += 1
        # Ensure script output includes this specific string for Ansible changed_when condition
        print(f"Successfully wrote {count} IDs to {output_file}.") 
        return True

    except ImportError:
        print("Error: Failed to import pandas or pyarrow. Please install them: pip3 install pandas pyarrow", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An error occurred during list generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Image ID list from OpenImages Parquet files.")
    parser.add_argument("--meta-dir", required=True, help="Directory containing metadata Parquet files.")
    parser.add_argument("--id-col", default=IMAGE_ID_COLUMN_NAME, help="Name of the column containing ImageIDs.")
    parser.add_argument("--output", required=True, help="Output file path for the ID list.")
    parser.add_argument("--count", type=int, required=True, help="Target number of unique IDs.")
    
    args = parser.parse_args()

    try:
        import pyarrow
    except ImportError:
        print("Error: pyarrow library not found. Please install it using: pip3 install pyarrow", file=sys.stderr)
        sys.exit(1)
        
    success = generate_list_from_parquet(
        metadata_dir=args.meta_dir,
        id_column=args.id_col,
        output_file=args.output,
        target_count=args.count
    )
    
    if not success:
        sys.exit(1) # Exit with error code if generation failed