# experimenting with export to coreml in case that runs faster
# since the onnx models are not compatible with the coreml onnx runtime
# and are therefore slower than I'd like 

import coremltools as ct
from midi_model import MIDIModel,MIDITokenizer
import torch 

ckpt = "/Users/matthewyk/src/ai-music/bcn-2024-system/skytnt-realtime/models/skytnt-pre-trained-la-dataset.ckpt"
tokenizer = MIDITokenizer()
model = MIDIModel(tokenizer).to(device="cpu")
ckpt = torch.load(ckpt, map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt)
model.load_state_dict(state_dict, strict=False)
model.eval()
input = torch.randint(tokenizer.vocab_size, (1, 16, tokenizer.max_token_seq), dtype=torch.int64, device="cpu")
print(f"\n\ninput shape for model is {input.shape}\n\n")
model.to_torchscript("test.ts", method="trace", example_inputs=input)
ts_model = torch.jit.load('test.ts')
cml_model = ct.convert(ts_model,convert_to="mlprogram", inputs=[ct.TensorType(shape=input.shape)])
cml_model.save("newmodel.mlpackage")
