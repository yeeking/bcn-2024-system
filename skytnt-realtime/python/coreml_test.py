import coremltools as ct
from midi_model import MIDIModel,MIDITokenizer
import torch 
import numpy as np 

ckpt = "/Users/matthewyk/src/ai-music/bcn-2024-system/skytnt-realtime/models/skytnt-pre-trained-la-dataset.ckpt"
tokenizer = MIDITokenizer()
model = MIDIModel(tokenizer).to(device="cpu")
print("Loading torch lightning model")
ckpt = torch.load(ckpt, map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt)
model.load_state_dict(state_dict, strict=False)
model.eval()

print("Testing inference on the torch lightning model")
input = torch.randint(tokenizer.vocab_size, (1, 16, tokenizer.max_token_seq), dtype=torch.int64, device="cpu")
model.forward(input)

print("Converting to torchscript model")
model.to_torchscript("test.ts", method="trace", example_inputs=input)
ts_model = torch.jit.load('test.ts')

print("Saving coreml model")
cml_model = ct.convert(ts_model,convert_to="mlprogram", inputs=[ct.TensorType(shape=input.shape)])
cml_model.save("newmodel.mlpackage")

print("Testing load of coreml model")
cml_model = ct.models.MLModel("newmodel.mlpackage")

print("Running inference on the coreml model")
random_input = np.random.rand(1, 16, 8).astype(np.float32)
input_data = {"x_1": random_input}
prediction = cml_model.predict(input_data)