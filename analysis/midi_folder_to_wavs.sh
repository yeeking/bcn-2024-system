#!/bin/bash

# Set the SoundFont file (modify this as needed)
SOUNDFONT="/home/matthew/.vst3/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"

# Set the directory containing MIDI files
MIDI_DIR="."
OUTPUT_DIR="$MIDI_DIR/normalized_wavs"

# Number of concurrent FFmpeg processes
BATCH_SIZE=8  # Adjust this for performance

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to process a single MIDI file
process_midi() {
    midi_file="$1"
    [[ -f "$midi_file" ]] || return  # Skip if not a file

    # Get the filename without extension
    filename=$(basename -- "$midi_file" .mid)

    # Convert MIDI to WAV using FluidSynth (44100 Hz sample rate)
    wav_file="$filename.wav"
    fluidsynth -ni "$SOUNDFONT" "$midi_file" -F "$wav_file" -r 44100

    # Normalize audio using FFmpeg (quiet mode, enforce 44100 Hz sample rate)
    normalized_wav="$OUTPUT_DIR/$filename.wav"
    ffmpeg -i "$wav_file" -af loudnorm -ar 44100 -y -hide_banner -loglevel error "$normalized_wav"

    # Remove the intermediate WAV file
    rm "$wav_file"

    echo "Processed: $midi_file -> $normalized_wav"
}

# Export function so it works in parallel subshells
export -f process_midi
export SOUNDFONT OUTPUT_DIR

# Use parallel processing with `xargs`
find "$MIDI_DIR" -maxdepth 1 -type f -name "*.mid" | xargs -P "$BATCH_SIZE" -I {} bash -c 'process_midi "$@"' _ {}

echo "All MIDI files have been processed and normalized!"
