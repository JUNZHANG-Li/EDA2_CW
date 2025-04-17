# Filename: generate_id_list.py (URL Version)
import pandas as pd
import os
import argparse
import glob # To find parquet files
import sys
# import re # No longer needed for ID extraction here

# --- Configuration ---
# Default column name containing the full URLs
DEFAULT_URL_COLUMN_NAME = 'url'

# --- Removed extract_id_from_url function ---

def generate_list_from_parquet(metadata_dir, url_column, output_file, target_count):
    """Loads URLs from Parquet files and writes a unique subset of full URLs."""

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
            print(f"    Read {len(df_part)} entries from column '{url_column}'.")

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

        # --- CHANGE: Work directly with URLs ---
        print("Filtering and finding unique URLs...")
        # Filter out any None values or empty strings
        valid_urls = all_urls_series.dropna()
        valid_urls = valid_urls[valid_urls.str.strip() != ''] # Remove empty/whitespace-only strings
        print(f"Found {len(valid_urls)} non-empty URLs.")

        # Get unique URLs
        unique_urls = valid_urls.drop_duplicates()
        print(f"Found {len(unique_urls)} unique URLs.")
        del valid_urls

        # Select the target number
        if len(unique_urls) >= target_count:
            selected_urls = unique_urls.head(target_count).tolist()
            print(f"Selected first {len(selected_urls)} unique URLs.")
        else:
            selected_urls = unique_urls.tolist()
            print(f"Warning: Found only {len(selected_urls)} unique URLs, less than target {target_count}.")
            print("Using all available unique URLs.")

        del unique_urls

        print(f"Writing {len(selected_urls)} URLs to {output_file}...")
        count = 0
        with open(output_file, 'w') as f:
            for url in selected_urls: # Iterate through URLs
                f.write(str(url).strip() + '\n') # Write the full URL
                count += 1
        # Ensure output includes this specific string for Ansible
        print(f"Successfully wrote {count} URLs to {output_file}.") # Updated print message
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
    parser = argparse.ArgumentParser(description="Generate unique URL list from URLs in OpenImages Parquet files.") # Updated description
    parser.add_argument("--meta-dir", required=True, help="Directory containing metadata Parquet files.")
    parser.add_argument("--url-col", default=DEFAULT_URL_COLUMN_NAME, help="Name of the column containing image URLs.") # Argument name is fine
    parser.add_argument("--output", required=True, help="Output file path for the extracted URL list.") # Updated help text
    parser.add_argument("--count", type=int, required=True, help="Target number of unique URLs.") # Updated help text

    args = parser.parse_args()

    try:
        import pyarrow
    except ImportError:
        print("Error: pyarrow library not found. Please install it using: pip3 install pyarrow", file=sys.stderr)
        sys.exit(1)

    success = generate_list_from_parquet(
        metadata_dir=args.meta_dir,
        url_column=args.url_col,
        output_file=args.output,
        target_count=args.count
    )

    if not success:
        sys.exit(1)