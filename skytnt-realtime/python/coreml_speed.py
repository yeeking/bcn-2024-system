import coremltools as ct
from midi_model import MIDIModel,MIDITokenizer
import torch 
import numpy as np 
import time


ckpt = "/Users/matthewyk/src/ai-music/bcn-2024-system/skytnt-realtime/models/skytnt-pre-trained-la-dataset.ckpt"
tokenizer = MIDITokenizer()
model = MIDIModel(tokenizer).to(device="cpu")
print("Loading torch lightning model")
ckpt = torch.load(ckpt, map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt)
model.load_state_dict(state_dict, strict=False)
model.eval()

print("Testing inference on the torch lightning model")

start_time = time.time()
for _ in range(1024):
    input = torch.randint(tokenizer.vocab_size, (1, 16, tokenizer.max_token_seq), dtype=torch.int64, device="cpu")
    model.forward(input)
end_time = time.time()
torch_elapsed_time = end_time - start_time

print("Testing load of coreml model")
cml_model = ct.models.MLModel("newmodel.mlpackage")

print("Testing inference on the coreml model")
start_time = time.time()
for _ in range(1024):
    random_input = np.random.rand(1, 16, 8).astype(np.float32)
    input_data = {"x_1": random_input}
    prediction = cml_model.predict(input_data)
end_time = time.time()
coreml_elapsed_time = end_time - start_time

print(f"Torch took {torch_elapsed_time} coreml took {coreml_elapsed_time}")