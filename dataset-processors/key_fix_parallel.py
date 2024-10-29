## This script takes a folder full of MIDI files
## (with subfolders etc.)
## estimates their key and mode (major/minor)
## then transposes each file to c major/minor
## writing the result out to another folder

import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def ensure_directory_exists(file_path):
    # Extract the directory path from the file path
    directory = os.path.dirname(file_path)
    if directory:
        # Create the directory and any intermediate directories if they do not exist
        os.makedirs(directory, exist_ok=True)

def find_mid_files(starting_folder):
    mid_files = []
    for root, dirs, files in os.walk(starting_folder):
        for file in files:
            if file.lower().endswith('.midi'):
                full_path = os.path.join(root, file)
                mid_files.append(full_path)
    return mid_files

def process_file(f, in_folder, out_folder):
    try:
        import os
        from music21 import converter, key, interval, analysis
        import tqdm
        from tqdm import tqdm

        # Load the MIDI file
        score = converter.parse(f)

        # Analyze the global key
        global_key = score.analyze('key')

        # Determine the mode of the global key
        mode = global_key.mode

        # Define the target key based on the mode
        if mode == 'minor':
            target_key = key.Key('C', 'minor')
        else:
            target_key = key.Key('C')  # Defaults to 'C major'

        # Calculate the transposition interval
        transposition_interval = interval.Interval(global_key.tonic, target_key.tonic)

        # Transpose the score
        transposed_score = score.transpose(transposition_interval)

        # Determine the output file path
        out_file = f.replace(in_folder, out_folder)
        ensure_directory_exists(out_file)

        # Save the transposed MIDI file
        transposed_score.write('midi', fp=out_file)
        return f, True, None  # Indicate success
    except Exception as e:
        return f, False, str(e)  # Indicate failure and return the error message


if __name__ == '__main__':
    assert len(sys.argv) == 3, "Send two args: in folder and out folder"
    in_folder = sys.argv[1]
    out_folder = sys.argv[2]

    midi_files = find_mid_files(in_folder)
    total_files = len(midi_files)
    print(f"Found {total_files} files")

    # You can specify the number of workers here
    max_workers = 8  # Adjust as needed

    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, f, in_folder, out_folder): f for f in midi_files
        }

        with tqdm(total=total_files) as pbar:
            for future in as_completed(futures):
                f = futures[future]
                try:
                    result = future.result()
                    midi_file, success, error_message = result
                    if success:
                        # Optionally log success
                        pass
                    else:
                        print(f"Failed to process {midi_file}: {error_message}")
                except Exception as e:
                    print(f"Exception occurred while processing {f}: {e}")
                pbar.update(1)  # Update the progress bar

