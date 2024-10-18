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
    print("Sending input to model")
    notes = [['note', i*8, 120, 0, np.random.randint(32, 96), 60] for i in range(64)] # start_time, duration, channel, note, velocity

    input_events = [480,
                     # channel 0
                     notes
    ]

    # try loading a midi file
    midif = 'mark_intro.md.MID'
    assert os.path.exists(midif)

    with open(midif, 'rb') as file:
        # Read the binary data from the file
        mid = file.read()
    input_events = MIDI.midi2score(mid)
    print(f"input channels: {len(input_events)}")
    
    # # max_len = len(input_events[1])
    # for max_len in [8, 16, 32, 64, 128]:
    #     # max_len = 64
    #     gen_events = ModelHandler.generate_midi_seq(model, tokenizer, 
    #                         input_events,
    #                         output_len=max_len, # generate as much as we give you
    #                         temp=0.7, 
    #                         top_p=0.5, #0.1 to 1.0
    #                         top_k=1, #1 to 20 
    #                         allow_cc=False, # True or False
    #                         amp=True, use_model=True, show_bar=False) 
    #     # print(f"Done with gen. Len of output {len(gen_events[1])} but max len {max_len}")
    #     print(f" at 120 BPM max len {max_len}, last beat at this offset in seconds {gen_events[1][-1][1] / 480 / 120 * 60}")