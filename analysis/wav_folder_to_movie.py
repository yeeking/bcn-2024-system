import os
import numpy as np
import scipy.io.wavfile as wav
from PIL import Image, ImageDraw, ImageFont
import tempfile
import re
import subprocess

# Configurable Parameters
FPS = 5  # Frames per second for the final video
VIDEO_WIDTH = 480  # Set video width
VIDEO_HEIGHT = 480  # Set video height
FFMPEG_PRESET = "ultrafast"  # Fastest encoding preset

# Use a temporary directory for all output files
temp_dir = tempfile.mkdtemp()
output_images = os.path.join(temp_dir, "frames")
os.makedirs(output_images, exist_ok=True)

input_txt_path = os.path.join(temp_dir, "input.txt")
output_audio_path = os.path.join(temp_dir, "output_audio.wav")
final_video_path = os.path.join(temp_dir, "final_output.mp4")

# Define font path
font_path = "/usr/share/fonts/opentype/comic-neue/ComicNeue-Regular.otf"

# Define WAV directory
wav_directory = "./"

# Function to extract and pad numeric prefix
def extract_number(filename):
    match = re.match(r"(\d+)_0\.wav", filename)
    if match:
        return f"{int(match.group(1)):08d}"  # Convert to int and pad to 8 digits
    return filename  # Fallback if format is incorrect

# Get sorted list of .wav files with correct order
wav_files = sorted(
    [f for f in os.listdir(wav_directory) if f.endswith(".wav")],
    key=extract_number  # Sort by extracted number
)

# Lists for concatenating audio
all_audio = []
sample_rate = None  # Will be set to the first file's sample rate
total_duration = 0  # Track total video duration

# Open the input.txt file for FFmpeg metadata
with open(input_txt_path, "w") as f:
    for i, wav_file in enumerate(wav_files):
        audio_path = os.path.join(wav_directory, wav_file)

        # Read WAV file using scipy
        rate, data = wav.read(audio_path)

        # Ensure consistent sample rate
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ValueError(f"Sample rate mismatch: {rate} vs {sample_rate}")

        # Convert stereo to mono if needed
        if len(data.shape) > 1:
            data = np.mean(data, axis=1).astype(np.int16)

        # Append to audio list
        all_audio.append(data)

        # Compute duration in seconds
        duration = len(data) / float(rate)
        total_duration += duration  # Track total duration

        # Generate image with filename text
        img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), "black")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, 80)  # Load font
        
        # Get text size and center it
        text_size = draw.textbbox((0, 0), wav_file, font=font)
        text_position = ((VIDEO_WIDTH - text_size[2]) // 2, (VIDEO_HEIGHT - text_size[3]) // 2)
        
        draw.text(text_position, wav_file, font=font, fill="white")
        
        # Save the image
        image_filename = os.path.join(output_images, f"frame_{i:04d}.png")
        img.save(image_filename)

        # Write FFmpeg metadata
        f.write(f"file '{image_filename}'\n")
        f.write(f"duration {duration:.2f}\n")

# Concatenate all audio files
full_audio = np.concatenate(all_audio)

# Save concatenated audio file
wav.write(output_audio_path, sample_rate, full_audio)

# Print expected video length
print(f"🎬 Expected video length: {total_duration:.2f} seconds")

# FFmpeg command to generate the final video with configurable FPS, resolution, and fastest encoding
ffmpeg_cmd = [
    "ffmpeg", "-f", "concat", "-safe", "0",
    "-i", input_txt_path, "-i", output_audio_path,
    "-r", str(FPS),  # Set frames per second
    "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",  # Set video resolution
    "-c:v", "libx264", "-preset", FFMPEG_PRESET,  # Fastest encoding preset
    "-c:a", "aac", "-strict", "experimental",
    "-y", final_video_path,
    "-progress", "pipe:1", "-nostats"  # Enable progress bar, suppress FFmpeg spam
]

# Run FFmpeg with progress bar
try:
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        if "out_time=" in line:  # Extract progress from FFmpeg output
            time_str = line.split("out_time=")[-1].split(" ")[0].strip()
            print(f"🟢 Progress: {time_str}", end="\r", flush=True)
    
    process.wait()
    
    if process.returncode == 0:
        print(f"\n✅ Final video generated: {final_video_path}")
    else:
        print("\n❌ FFmpeg encountered an error.")

except subprocess.CalledProcessError as e:
    print(f"\n❌ FFmpeg failed: {e}")



# import os
# import numpy as np
# import scipy.io.wavfile as wav
# from PIL import Image, ImageDraw, ImageFont
# import tempfile
# import re
# import subprocess

# # Use a temporary directory for all output files
# temp_dir = tempfile.mkdtemp()
# output_images = os.path.join(temp_dir, "frames")
# os.makedirs(output_images, exist_ok=True)

# input_txt_path = os.path.join(temp_dir, "input.txt")
# output_audio_path = os.path.join(temp_dir, "output_audio.wav")
# # final_video_path = os.path.join(temp_dir, "final_output.mp4")
# final_video_path = "final_output.mp4"

# # Define font path
# font_path = "/usr/share/fonts/opentype/comic-neue/ComicNeue-Regular.otf"

# # Define WAV directory
# wav_directory = "./"

# # Function to extract and pad numeric prefix
# def extract_number(filename):
#     match = re.match(r"(\d+)_0\.wav", filename)
#     if match:
#         return f"{int(match.group(1)):08d}"  # Convert to int and pad to 8 digits
#     return filename  # Fallback if format is incorrect

# # Get sorted list of .wav files with correct order
# wav_files = sorted(
#     [f for f in os.listdir(wav_directory) if f.endswith(".wav")],
#     key=extract_number  # Sort by extracted number
# )

# # Lists for concatenating audio
# all_audio = []
# sample_rate = None  # Will be set to the first file's sample rate

# # Open the input.txt file for FFmpeg metadata
# with open(input_txt_path, "w") as f:
#     for i, wav_file in enumerate(wav_files):
#         audio_path = os.path.join(wav_directory, wav_file)

#         # Read WAV file using scipy
#         rate, data = wav.read(audio_path)

#         # Ensure consistent sample rate
#         if sample_rate is None:
#             sample_rate = rate
#         elif rate != sample_rate:
#             raise ValueError(f"Sample rate mismatch: {rate} vs {sample_rate}")

#         # Convert stereo to mono if needed
#         if len(data.shape) > 1:
#             data = np.mean(data, axis=1).astype(np.int16)

#         # Append to audio list
#         all_audio.append(data)

#         # Compute duration in seconds
#         duration = len(data) / float(rate)

#         # Generate image with filename text
#         img = Image.new("RGB", (1280, 720), "black")
#         draw = ImageDraw.Draw(img)
#         font = ImageFont.truetype(font_path, 80)  # Load font
        
#         # Get text size and center it
#         text_size = draw.textbbox((0, 0), wav_file, font=font)
#         text_position = ((1280 - text_size[2]) // 2, (720 - text_size[3]) // 2)
        
#         draw.text(text_position, wav_file, font=font, fill="white")
        
#         # Save the image
#         image_filename = os.path.join(output_images, f"frame_{i:04d}.png")
#         img.save(image_filename)

#         # Write FFmpeg metadata
#         f.write(f"file '{image_filename}'\n")
#         f.write(f"duration {duration:.2f}\n")

# # Concatenate all audio files
# full_audio = np.concatenate(all_audio)

# # Save concatenated audio file
# wav.write(output_audio_path, sample_rate, full_audio)

# print(f"✅ Generated {len(wav_files)} images, input.txt, and concatenated audio file {output_audio_path}")

# # FFmpeg command to generate the final video
# ffmpeg_cmd = [
#     "ffmpeg", "-f", "concat", "-safe", "0",
#     "-i", input_txt_path, "-i", output_audio_path,
#     "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
#     "-y", final_video_path
# ]

# # Run FFmpeg
# try:
#     subprocess.run(ffmpeg_cmd, check=True)
#     print(f"✅ Final video generated: {final_video_path}")
# except subprocess.CalledProcessError as e:
#     print(f"❌ FFmpeg failed: {e}")
