import sys
import os
import music21
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # Progress bar library

def find_mid_files(starting_folder):
    mid_files = []
    for root, dirs, files in os.walk(starting_folder):
        for file in files:
            if file.lower().endswith('.midi'):
                full_path = os.path.join(root, file)
                mid_files.append(full_path)
    return mid_files

def transpose_midi_file(in_file, out_folder, in_folder):
    assert os.path.exists(in_file), "Cannot find MIDI file"

    # Calculate the relative path for the file to recreate the structure
    relative_path = os.path.relpath(in_file, in_folder)
    relative_dir = os.path.dirname(relative_path)

    # Parse the MIDI file
    score = music21.converter.parse(in_file)
    
    # Save the original (untransposed) MIDI file as "plus_0"
    output_dir = os.path.join(out_folder, relative_dir, "plus_0")
    os.makedirs(output_dir, exist_ok=True)
    original_filename = os.path.join(output_dir, os.path.basename(in_file))
    score.write('midi', original_filename)

    # Transpose the score from 1 to 11 semitones
    for transpose_step in range(1, 12):
        interval_up = music21.interval.Interval(transpose_step)
        transposed_score = score.transpose(interval_up)

        # Create the output subdirectory structure in the output folder
        output_dir = os.path.join(out_folder, relative_dir, f"plus_{transpose_step}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        out_filename = os.path.join(output_dir, os.path.basename(in_file))
        
        # Write the transposed file
        transposed_score.write('midi', out_filename)

    # Return the number of files written (12: 1 original + 11 transpositions)
    return 12

def process_file(file, out_folder, in_folder):
    try:
        return transpose_midi_file(file, out_folder, in_folder)
    except Exception as e:
        print(f"Failed {file}: {e}")
        return 0  # Return 0 files if there was an error

if __name__ == '__main__':
    assert len(sys.argv) == 3, "Usage: python script.py MIDI_input_folder MIDI_output_folder"
    
    in_folder = sys.argv[1]
    assert os.path.exists(in_folder), f"Input folder {in_folder} not found."
    
    out_folder = sys.argv[2]
    os.makedirs(out_folder, exist_ok=True)

    midi_files = find_mid_files(in_folder)
    total_transpositions = len(midi_files) * 12  # 1 original + 11 transpositions for each file
    print(f"Found {len(midi_files)} files. Processing in parallel with {total_transpositions} total transpositions.")

    # Initialize the progress bar with the total number of transpositions
    with tqdm(total=total_transpositions, desc="Processing MIDI files") as progress_bar:
        # Using ProcessPoolExecutor to parallelize the processing of each MIDI file
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, f, out_folder, in_folder) for f in midi_files]
            for future in as_completed(futures):
                # Update the progress bar based on the number of files processed per future
                progress_bar.update(future.result())
