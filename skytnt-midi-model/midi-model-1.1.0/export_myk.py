## This script takes a trained model checkpoint and
## converts it to a smaller, non-trainable one (I think!)

import torch
import argparse
import torch.nn as nn
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
import sys

assert len(sys.argv) == 3, "Need two args: input model and output model file"

in_ckpt = sys.argv[1]
out_ckpt = sys.argv[2]


tokenizer = MIDITokenizer()
model = MIDIModel(tokenizer).to(device="cpu")
ckpt = torch.load(in_ckpt, map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt)
model.load_state_dict(state_dict, strict=False)
model.eval()
torch.save(model, out_ckpt)

