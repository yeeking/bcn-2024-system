import os
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
from gen_utils import ImproviserAgent, MIDIScheduler, ModelHandler
import mido
import time


if __name__ == "__main__":
    ckpt = "../../trained-models/skytnt/version_703-la-hawthorne-finetune.ckpt"
    # ckpt = "../../trained-models/skytnt/skytnt-hf-model-la-dataset.ckpt"
    assert os.path.exists(ckpt), "Cannot find checkpoint file " + ckpt

    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device='cuda')
    #model = MIDIModel(tokenizer).to(device='cpu')
    
    ModelHandler.load_model(ckpt, model)
    
    improviser = ImproviserAgent(input_time_ms=2000, 
                                 output_length=32, 
                                 feedback_mode=False, 
                                model=model, 
                                tokenizer=tokenizer,
                                allow_gen_overlap=False,  
                                test_mode=False) 
    
    improviser.initMIDIFromCLI() # select MIDI inputs and outputs
    improviser.start() # start responding to MIDI input 

    try:
        # Block main thread, simulate waiting for user input or other tasks
        input("Press Enter to stop the model runner...\n")
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
        
    improviser.stop()
  
