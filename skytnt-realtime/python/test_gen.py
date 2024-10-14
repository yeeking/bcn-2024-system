## Script to muck around with the generator

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
    print("Loading model weights")
    ModelHandler.load_model(ckpt, model)
    print("Sending input to model")
    notes = [['note', i*120, 120, 0, 60, 60] for i in range(128)] # start_time, duration, channel, note, velocity
    input_events = [480,
                     # channel 0
                     [
                        ['note', 0, 120, 0, 60, 60]# start_time, duration, channel, note, velocity
                     ]
    ]
    input_events = [480,
                     # channel 0
                     notes
    ]
    max_len = 32
    gen_events = ModelHandler.generate_midi_seq(model, tokenizer, 
                        input_events,
                        output_len=max_len, # generate as much as we give you
                        temp=0.7, 
                        top_p=0.5, #0.1 to 1.0
                        top_k=1, #1 to 20 
                        allow_cc=False, # True or False
                        amp=True, use_model=True) 
    real_data = gen_events[1][1]
    print(f"Done with gen. Len of output {len(real_data)} but max len {max_len}")
    