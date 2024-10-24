import os
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
from gen_utils import MidiDeviceHandler
import mido
import time

# Example usage:
# if __name__ == "__main__":
#     scheduler = MIDIScheduler()

#     # Test MIDI message, for example using mido.Message
#     msg1 = mido.Message('note_on', note=64, velocity=64)
#     msg2 = mido.Message('note_off', note=64, velocity=64)

#     scheduler.addMIDIMsg(msg1, 500)  # Send after 500 ms
#     scheduler.addMIDIMsg(msg2, 2000)  # Send after 1000 ms

#     time.sleep(10)  # Wait enough time for the messages to be sent
#     scheduler.stop()



if __name__ == "__main__":


    def got_midi(msg:mido.Message):
        print(msg)

    midiHandler = MidiDeviceHandler(got_midi)
    midiHandler.getMIDIDevicesFromUser()
    midiHandler.initMIDI()


    try:
        # Block main thread, simulate waiting for user input or other tasks
        input("Press Enter to stop the model runner...\n")
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
        
    midiHandler.stop()
  
