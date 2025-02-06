import sys
import os
import pretty_midi
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

def transpose_midi_pretty_midi(input_file, output_file, semitone_shift):
    """Transposes a single MIDI file using pretty_midi and saves the output."""
    midi_data = pretty_midi.PrettyMIDI(input_file)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.pitch = min(127, max(0, note.pitch + semitone_shift))
    midi_data.write(output_file)

def process_midi_file(input_file, out_folder, in_folder):
    """Processes a single MIDI file, saving 12 transpositions (including original) to output folder."""
    # Calculate the relative path for the file to recreate the structure in the output
    relative_path = os.path.relpath(input_file, in_folder)
    relative_dir = os.path.dirname(relative_path)

    # Save the original (untransposed) MIDI file as "plus_0"
    output_dir = os.path.join(out_folder, relative_dir, "plus_0")
    os.makedirs(output_dir, exist_ok=True)
    original_filename = os.path.join(output_dir, os.path.basename(input_file))
    transpose_midi_pretty_midi(input_file, original_filename, 0)  # Save original with no transposition

    # Transpose and save from 1 to 11 semitones
    for transpose_step in range(-12, 12):
        interval_dir = os.path.join(out_folder, relative_dir, f"plus_{transpose_step}")
        os.makedirs(interval_dir, exist_ok=True)
        output_filename = os.path.join(interval_dir, os.path.basename(input_file))
        transpose_midi_pretty_midi(input_file, output_filename, transpose_step)

    # Return the number of files written (12: 1 original + 11 transpositions)
    return 12

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
            futures = [executor.submit(process_midi_file, f, out_folder, in_folder) for f in midi_files]
            for future in as_completed(futures):
                # Update the progress bar based on the number of files processed per future
                progress_bar.update(future.result())
