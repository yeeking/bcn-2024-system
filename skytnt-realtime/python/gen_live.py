import os
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
from gen_utils import ImproviserAgent, MIDIScheduler, ModelHandler
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
    # ckpt = "small.ckpt"
    ckpt = "../../trained-models/skytnt/version_703-la-hawthorne-finetune.ckpt"
    # midi_file = 'input.mid'

    assert os.path.exists(ckpt), "Cannot find checkpoint file " + ckpt
    # assert os.path.exists(midi_file), "Cannot find MIDI file " + midi_file

    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device='cuda')
    ModelHandler.load_model(ckpt, model)
    
    # print(model)

    improviser = ImproviserAgent(memory_length=32, model=model, tokenizer=tokenizer, test_mode=False) 
    
    improviser.initMIDI() # select MIDI inputs and outputs
    improviser.run() # start responding to MIDI input 

    try:
        # Block main thread, simulate waiting for user input or other tasks
        input("Press Enter to stop the model runner...\n")
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
        
    improviser.stop()
  
