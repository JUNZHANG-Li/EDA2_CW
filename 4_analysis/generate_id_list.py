# Filename: generate_id_list.py (Updated Version)
import pandas as pd
import os
import argparse
import glob # To find parquet files
import sys
import re # For more robust extraction

# --- Configuration ---
# We will now read the 'url' column by default
URL_COLUMN_NAME = 'url'

def extract_id_from_url(url):
    """Extracts the image ID (filename without extension) from the URL."""
    if not isinstance(url, str):
        return None
    try:
        # Regex to capture the filename part before .jpg (or other extensions if needed)
        # Assumes format like .../some_path/image_id.jpg
        match = re.search(r'/([^/]+)\.(jpg|jpeg|png)$', url, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            # Fallback attempt: simple split (less robust)
            filename = url.split('/')[-1]
            return filename.split('.')[0] if '.' in filename else filename
    except Exception:
        return None # Return None if extraction fails

def generate_list_from_parquet(metadata_dir, url_column, output_file, target_count):
    """Loads URLs from Parquet files, extracts IDs, and writes a unique subset."""
    
    parquet_files = glob.glob(os.path.join(metadata_dir, '*.parquet'))
    
    if not parquet_files:
        print(f"Error: No Parquet files found in directory: {metadata_dir}", file=sys.stderr)
        return False

    print(f"Found Parquet files: {parquet_files}")
    
    all_urls_list = []
    total_loaded = 0

    try:
        print("Loading URLs from Parquet files...")
        for file_path in parquet_files:
            print(f"  Reading {file_path}...")
            # Load only the required URL column
            df_part = pd.read_parquet(file_path, columns=[url_column])
            print(f"    Read {len(df_part)} URLs.")
            
            if url_column not in df_part.columns:
                print(f"Error: Column '{url_column}' not found in {file_path}. Found columns: {df_part.columns.tolist()}", file=sys.stderr)
                print("Please inspect the Parquet file schema.")
                return False
                
            all_urls_list.append(df_part[url_column])
            total_loaded += len(df_part)
            del df_part 

        print(f"Loaded a total of {total_loaded} URLs.")

        if not all_urls_list:
             print("Error: No URLs were loaded.", file=sys.stderr)
             return False
             
        all_urls_series = pd.concat(all_urls_list, ignore_index=True)
        del all_urls_list

        print("Extracting Image IDs from URLs...")
        # Apply the extraction function to the Series of URLs
        extracted_ids = all_urls_series.apply(extract_id_from_url)
        del all_urls_series # Free memory
        
        # Filter out any None values that resulted from failed extractions
        valid_ids = extracted_ids.dropna()
        print(f"Extracted {len(valid_ids)} potential Image IDs.")
        
        # Get unique IDs
        print("Finding unique IDs...")
        unique_ids = valid_ids.drop_duplicates()
        print(f"Found {len(unique_ids)} unique IDs.")
        del valid_ids

        # Select the target number
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
                f.write(str(img_id).strip() + '\n') # ID should already be string
                count += 1
        # Ensure output includes this specific string for Ansible
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
    parser = argparse.ArgumentParser(description="Generate Image ID list by extracting from URLs in OpenImages Parquet files.")
    parser.add_argument("--meta-dir", required=True, help="Directory containing metadata Parquet files.")
    # Updated argument to reflect we expect the URL column name
    parser.add_argument("--url-col", default=URL_COLUMN_NAME, help="Name of the column containing image URLs.")
    parser.add_argument("--output", required=True, help="Output file path for the extracted ID list.")
    parser.add_argument("--count", type=int, required=True, help="Target number of unique IDs.")
    
    args = parser.parse_args()

    try:
        import pyarrow
    except ImportError:
        print("Error: pyarrow library not found. Please install it using: pip3 install pyarrow", file=sys.stderr)
        sys.exit(1)
        
    success = generate_list_from_parquet(
        metadata_dir=args.meta_dir,
        url_column=args.url_col, # Pass the url column argument correctly
        output_file=args.output,
        target_count=args.count
    )
    
    if not success:
        sys.exit(1) 