# experimenting with export to coreml in case that runs faster
# since the onnx models are not compatible with the coreml onnx runtime
# and are therefore slower than I'd like 

import coremltools as ct
from midi_model import MIDIModel,MIDITokenizer
import torch
from torch import nn 


# Special classes to allow seperate export of tokenizer and main base model 
class MIDIModelBase(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = model.net

MIDIModelBase.forward = MIDIModel.forward

class MIDIModelToken(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net_token = model.net_token
        self.lm_head = model.lm_head


MIDIModelToken.forward = MIDIModel.forward_token


ckpt = "/Users/matthewyk/src/ai-music/bcn-2024-system/skytnt-realtime/models/skytnt-pre-trained-la-dataset.ckpt"
tokenizer = MIDITokenizer()
model = MIDIModel(tokenizer).to(device="cpu")
ckpt = torch.load(ckpt, map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt)
model.load_state_dict(state_dict, strict=False)
model.eval()
# now break it into two models 
model_base = MIDIModelBase(model).eval()
model_token = MIDIModelToken(model).eval()

print("Exporting base model")
x = torch.randint(tokenizer.vocab_size, (1, 16, tokenizer.max_token_seq), dtype=torch.int64, device="cpu")
print("Saving and loading from script")
ts_model = torch.jit.trace(model_base, x)
ts_model.save('base.ts')
ts_model = torch.jit.load('base.ts')
print("Saving to coreml")
cml_model = ct.convert(ts_model,convert_to="mlprogram", inputs=[ct.TensorType(shape=x.shape)])
cml_model.save("base.mlpackage")

print("Exporting the tokenizer model")

hidden = torch.randn(1, 1024, device="cpu")
x = torch.randint(tokenizer.vocab_size, (1, tokenizer.max_token_seq), dtype=torch.int64, device="cpu")

print("Saving and loading from script")
ts_model = torch.jit.trace(model_token, (hidden, x))
ts_model.save('tokenizer.ts')
ts_model = torch.jit.load('tokenizer.ts')
print("Saving to coreml")
cml_model = ct.convert(ts_model,convert_to="mlprogram", inputs=[ct.TensorType(shape=hidden.shape), ct.TensorType(shape=x.shape)])
cml_model.save("tokenizer.mlpackage")
