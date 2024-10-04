import os
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
from gen_utils import ImproviserAgent, load_model

if __name__ == "__main__":
    # ckpt = "small.ckpt"
    ckpt = "../models/small.ckpt"
    midi_file = 'input.mid'

    assert os.path.exists(ckpt), "Cannot find checkpoint file " + ckpt
    assert os.path.exists(midi_file), "Cannot find MIDI file " + midi_file

    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device='cpu')
    load_model(ckpt, model)
    print(model)

    improviser = ImproviserAgent(memory_length=128, model=model, tokenizer=tokenizer, test_mode=False) 
    
    improviser.initMIDI()
    improviser.run()

    try:
        # Block main thread, simulate waiting for user input or other tasks
        input("Press Enter to stop the model runner...\n")
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
        
    improviser.stop()
  
