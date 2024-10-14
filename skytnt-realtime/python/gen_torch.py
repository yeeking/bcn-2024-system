import torch
import MIDI
import numpy as np
import os 
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
import tqdm
import torch.nn.functional as F


def load_model(path, model:MIDIModel):
    """
    load a python model from the sent checkpoint
    """
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return "success"


@torch.inference_mode()
def generate(model:MIDIModel, tokenizer:MIDITokenizer, prompt=None, max_len=512, temp=1.0, top_p=0.98, top_k=20,
             disable_patch_change=False, disable_control_change=False, disable_channels=None, amp=True):
    if disable_channels is not None:
        disable_channels = [tokenizer.parameter_ids["channel"][c] for c in disable_channels]
    else:
        disable_channels = []
    max_token_seq = tokenizer.max_token_seq
    if prompt is None:
        input_tensor = torch.full((1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=model.device)
        input_tensor[0, 0] = tokenizer.bos_id  # bos
    else:
        prompt = prompt[:, :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = np.pad(prompt, ((0, 0), (0, max_token_seq - prompt.shape[-1])),
                            mode="constant", constant_values=tokenizer.pad_id)
        input_tensor = torch.from_numpy(prompt).to(dtype=torch.long, device=model.device)
    input_tensor = input_tensor.unsqueeze(0)
    cur_len = input_tensor.shape[1]
    bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
    if model.device == 'cuda':
        autocast = torch.cuda.amp.autocast  # Use CUDA autocast for GPU
    else:
        autocast = torch.cpu.amp.autocast  # Use CPU autocast for CPU

    with bar, autocast(enabled=amp):
        while cur_len < max_len:
            # print(f"Calling forward length is {cur_len} of {max_len} input shape is {input_tensor.shape} ")
            end = False
            hidden = model.forward(input_tensor)[0, -1].unsqueeze(0)
            next_token_seq = None
            event_name = ""
            for i in range(max_token_seq):
                mask = torch.zeros(tokenizer.vocab_size, dtype=torch.int64, device=model.device)
                if i == 0:
                    mask_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                    if disable_patch_change:
                        mask_ids.remove(tokenizer.event_ids["patch_change"])
                    if disable_control_change:
                        mask_ids.remove(tokenizer.event_ids["control_change"])
                    mask[mask_ids] = 1
                else:
                    param_name = tokenizer.events[event_name][i - 1]
                    mask_ids = tokenizer.parameter_ids[param_name]
                    if param_name == "channel":
                        mask_ids = [i for i in mask_ids if i not in disable_channels]
                    mask[mask_ids] = 1
                logits = model.forward_token(hidden, next_token_seq)[:, -1:]
                scores = torch.softmax(logits / temp, dim=-1) * mask
                sample = model.sample_top_p_k(scores, top_p, top_k)
                if i == 0:
                    next_token_seq = sample
                    eid = sample.item()
                    if eid == tokenizer.eos_id:
                        end = True
                        break
                    event_name = tokenizer.id_events[eid]
                else:
                    next_token_seq = torch.cat([next_token_seq, sample], dim=1)
                    if len(tokenizer.events[event_name]) == i:
                        break
            if next_token_seq.shape[1] < max_token_seq:
                next_token_seq = F.pad(next_token_seq, (0, max_token_seq - next_token_seq.shape[1]),
                                       "constant", value=tokenizer.pad_id)
            next_token_seq = next_token_seq.unsqueeze(1)
            # print(next_token_seq)
            input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
            cur_len += 1
            bar.update(1)
            yield next_token_seq.reshape(-1).cpu().numpy()
            if end:
                break


def generate_midi_seq(model:MIDIModel, tokenizer:MIDITokenizer, midi_filename, n_events_from_file, gen_events, temp, top_p, top_k, allow_cc, amp):
    """
    generate a midi sequence using the sent params 
    """
    # prepare variables
    mid_seq = []
    gen_events = int(gen_events)
    max_len = gen_events
    disable_patch_change = False
    disable_channels = None

    # generate from a midi file
    assert os.path.exists(midi_filename)
    # make an input token sequence from the 
    # midi in the file
    with open(midi_filename, 'rb') as file:
        # Read the binary data from the file
        mid = file.read()
    mid = tokenizer.tokenize(MIDI.midi2score(mid))
    mid = np.asarray(mid, dtype=np.int64)
    mid = mid[:int(n_events_from_file)]
    max_len += len(mid)
    for token_seq in mid:
        mid_seq.append(token_seq.tolist())
    

    generator = generate(model, tokenizer, mid, max_len=max_len, 
                        temp=temp, top_p=top_p, top_k=top_k,
                        disable_patch_change=disable_patch_change, disable_control_change=not allow_cc,
                        disable_channels=disable_channels, amp=amp)
    for i, token_seq in enumerate(generator):
        mid_seq.append(token_seq)
        ## this bit is for outputting as you generate 
        # event = tokenizer.tokens2event(token_seq.tolist())
    mid = tokenizer.detokenize(mid_seq)
    with open(f"output.mid", 'wb') as f:
        f.write(MIDI.score2midi(mid))
    

def run():
    # ckpt = "small.ckpt"
    ckpt = "../models/small.ckpt"
    midi_file = 'input.mid'

    assert os.path.exists(ckpt), "Cannot find checkpoint file " + ckpt
    assert os.path.exists(midi_file), "Cannot find MIDI file " + midi_file
    print(f"Generating from file {midi_file} with ckpt {ckpt}")

    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device='cuda')
    load_model(ckpt, model)
    print(model)
     
    generate_midi_seq(model, tokenizer, midi_file, 
                      n_events_from_file=128, 
                      gen_events=1024, 
                      temp=0.7, 
                      top_p=0.5, #0.1 to 1.0
                      top_k=1, #1 to 20 
                      allow_cc=False, # True or False
                      amp=True) # True or False
    
if __name__ == "__main__":
    run()
