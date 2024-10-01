# Connect to the MIDI input 
# and use the received notes to generate a 
# continuation

import mido

# Set the backend to use 'portmidi'
mido.set_backend('mido.backends.rtmidi')

# List available MIDI input devices
print("Available MIDI input devices:")
midi_devices = mido.get_input_names()
for i, device in enumerate(midi_devices):
    print(f"{i}: {device}")

# Prompt user to select a MIDI device
device_index = int(input("Enter the number of the MIDI device you want to connect to: "))

# Validate the selected device
if device_index < 0 or device_index >= len(midi_devices):
    print("Invalid selection. Please try again.")
    exit()

selected_device = midi_devices[device_index]
print(f"Connecting to: {selected_device}")

# Callback function to handle incoming MIDI messages
def midi_callback(message:mido.Message):
    # print(f"Received message: {message}")
    if message.type == 'note_on' or message.type == 'note_off':
        print(message)

# Open the selected MIDI input port with a callback
# with mido.open_input(selected_device) as inport:

with mido.open_input(selected_device, callback=midi_callback) as inport:
    print(f"Listening for messages from {selected_device}...")

    try:
        # Wait for MIDI messages to come in via callback
        input("Press Enter to stop listening...\n")
    except KeyboardInterrupt:
        print("MIDI listener stopped.")
