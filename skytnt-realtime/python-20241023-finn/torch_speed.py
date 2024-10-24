from midi_model import MIDIModel,MIDITokenizer
import torch 
import numpy as np 
import time

frames = 16

ckpt = "../models/small.ckpt"
tokenizer = MIDITokenizer()

def load_model(ckpt, device):
    model = MIDIModel(tokenizer).to(device=device)
    print("Loading torch lightning model")
    ckpt = torch.load(ckpt, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

print("Testing inference on the torch lightning model")

model = load_model(ckpt, 'cpu')

start_time = time.time()
for _ in range(frames):
    input = torch.randint(tokenizer.vocab_size, (1, 16, tokenizer.max_token_seq), dtype=torch.int64, device="cpu")
    model.forward(input)
end_time = time.time()
torch_cpu_elapsed_time = end_time - start_time

model = load_model(ckpt, 'cuda')

start_time = time.time()
for _ in range(frames):
    input = torch.randint(tokenizer.vocab_size, (1, 16, tokenizer.max_token_seq), dtype=torch.int64, device="cuda")
    model.forward(input)
end_time = time.time()
torch_gpu_elapsed_time = end_time - start_time

speed_boost = torch_cpu_elapsed_time / torch_gpu_elapsed_time

print(f"Torch took CPU: {torch_cpu_elapsed_time} GPU: {torch_gpu_elapsed_time} speed boost: {round(speed_boost, 2)}X for {frames} frames ")
