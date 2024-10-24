## Script to muck around with the generator

import os
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
from gen_utils import ImproviserAgent, MIDIScheduler, ModelHandler
import mido
import time
import numpy as np 
import MIDI

if __name__ == "__main__":
    ckpt = "../../trained-models/skytnt/version_703-la-hawthorne-finetune.ckpt"
    # ckpt = "../../trained-models/skytnt/skytnt-hf-model-la-dataset.ckpt"
    assert os.path.exists(ckpt), "Cannot find checkpoint file " + ckpt

    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device='cuda')
    print("Loading model weights")
    ModelHandler.load_model(ckpt, model)
    # print("Sending input to model")

    mid,disable_channels, disable_patch_change = ModelHandler.get_random_input(tokenizer)
    print(disable_channels)

    print(tokenizer.detokenize(mid))

    gen_events = ModelHandler.generate_midi_seq(model, tokenizer, 
                            tokenizer.detokenize(mid),
                            output_len=128, # generate as much as we give you
                            temp=0.7, 
                            top_p=0.5, #0.1 to 1.0
                            top_k=1, #1 to 20 
                            allow_cc=False, # True or False
                            amp=True, use_model=True, show_bar=False) 
    print("Generated events", len(gen_events))
    # mid = MIDI.score2midi(gen_events)
    with open(f"output.mid", 'wb') as f:
        f.write(MIDI.score2midi(gen_events))


