### MYK hack that just generates a one-off randomly initialised sequence
### with all instruments going nuts


import argparse
import glob
import json
import uuid
import wave

# import gradio as gr
import numpy as np
import torch

import torch.nn.functional as F
import tqdm

import MIDI
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
from midi_synthesizer import synthesis
# from huggingface_hub import hf_hub_download


def write_audio_to_file(audio, sample_rate):
    # Open a wave file to write the audio
    with wave.open('output.wav', 'w') as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(2)  # 16-bit samples
        wf.setframerate(sample_rate)
        # int_audio = np.int16(audio * 32767)
        wf.writeframes(audio.tobytes())

@torch.inference_mode()
def generate(prompt=None, max_len=512, temp=1.0, top_p=0.98, top_k=20,
             disable_patch_change=False, disable_control_change=False, disable_channels=None, amp=True):
    if disable_channels is not None:
        disable_channels = [tokenizer.parameter_ids["channel"][c] for c in disable_channels]
    else:
        disable_channels = []
    max_token_seq = tokenizer.max_token_seq
    if prompt is None:
        print("No prompt... making one ")
        input_tensor = torch.full((1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=model.device)
        print(input_tensor)
        input_tensor[0, 0] = tokenizer.bos_id  # bos
    else:
        print(f"Got a prompt. truncating to {max_token_seq}")
        prompt = prompt[:, :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = np.pad(prompt, ((0, 0), (0, max_token_seq - prompt.shape[-1])),
                            mode="constant", constant_values=tokenizer.pad_id)
        input_tensor = torch.from_numpy(prompt).to(dtype=torch.long, device=model.device)
        print(input_tensor.to('cpu'))
    input_tensor = input_tensor.unsqueeze(0)
    cur_len = input_tensor.shape[1]
    bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
    with bar, torch.cuda.amp.autocast(enabled=amp):
        while cur_len < max_len:
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
            input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
            cur_len += 1
            bar.update(1)
            yield next_token_seq.reshape(-1).cpu().numpy()
            if end:
                break


def create_msg(name, data):
    return {"name": name, "data": data, "uuid": uuid.uuid4().hex}


def send_msgs(msgs, msgs_history):
    msgs_history.append(msgs)
    if len(msgs_history) > 50:
        msgs_history.pop(0)
    return json.dumps(msgs_history)


def run(instruments, drum_kit, mid, midi_events, gen_events, temp, top_p, top_k, allow_cc, amp):
    print("Here goes run")
    msgs_history = []
    mid_seq = []
    gen_events = int(gen_events)
    max_len = gen_events

    disable_patch_change = False
    disable_channels = None
    print("Run from zero input")
    i = 0
    mid = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
    patches = {}
    if instruments is None:
        instruments = []
    for instr in instruments:
        print(instr)
        patches[i] = patch2number[instr]
        i = (i + 1) if i != 8 else 10
    # if drum_kit != "None":
    #     patches[9] = drum_kits2number[drum_kit]
    for i, (c, p) in enumerate(patches.items()):
        mid.append(tokenizer.event2tokens(["patch_change", 0, 0, i, c, p]))
    
    # here ius where you can insert 
    # some more MIDI messages
    

    mid_seq = mid
    mid = np.asarray(mid, dtype=np.int64)
    if len(instruments) > 0:
        disable_patch_change = True
        disable_channels = [i for i in range(16) if i not in patches]

    print("Calling generate")
    generator = generate(mid, max_len=max_len, temp=temp, top_p=top_p, top_k=top_k,
                         disable_patch_change=disable_patch_change, disable_control_change=not allow_cc,
                         disable_channels=disable_channels, amp=amp)
    print("Got response. iterating it")
    for i, token_seq in enumerate(generator):
        mid_seq.append(token_seq)
        # event = tokenizer.tokens2event(token_seq.tolist())
        # yield mid_seq, None, None, send_msgs([create_msg("visualizer_append", event), create_msg("progress", [i + 1, gen_events])], msgs_history), msgs_history
    mid = tokenizer.detokenize(mid_seq)
    print("Writing to output.md")
    with open(f"output.mid", 'wb') as f:
        f.write(MIDI.score2midi(mid))
    soundfont_path = "../../../soundfonts/SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2"
    audio = synthesis(MIDI.score2opus(mid), soundfont_path)
    write_audio_to_file(audio, 44100)
    print("Got audio of length ", audio.shape)
    return mid_seq, "output.mid", (44100, audio), send_msgs([create_msg("visualizer_end", None)], msgs_history), msgs_history


def cancel_run(mid_seq, msgs_history):
    if mid_seq is None:
        return None, None, []
    mid = tokenizer.detokenize(mid_seq)
    with open(f"output.mid", 'wb') as f:
        f.write(MIDI.score2midi(mid))
    # audio = synthesis(MIDI.score2opus(mid), soundfont_path)
    return "output.mid", (44100, audio), send_msgs([create_msg("visualizer_end", None)], msgs_history)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return "success"


def get_model_path():
    model_paths = sorted(glob.glob("**/*.ckpt", recursive=True))
    return gr.Dropdown(choices=model_paths)


# def load_javascript(dir="javascript"):
#     scripts_list = glob.glob(f"{dir}/*.js")
#     javascript = ""
#     for path in scripts_list:
#         with open(path, "r", encoding="utf8") as jsfile:
#             javascript += f"\n<!-- {path} --><script>{jsfile.read()}</script>"
#     template_response_ori = gr.routes.templates.TemplateResponse

#     def template_response(*args, **kwargs):
#         res = template_response_ori(*args, **kwargs)
#         res.body = res.body.replace(
#             b'</head>', f'{javascript}</head>'.encode("utf8"))
#         res.init_headers()
#         return res

#     gr.routes.templates.TemplateResponse = template_response


number2drum_kits = {-1: "None", 0: "Standard", 8: "Room", 16: "Power", 24: "Electric", 25: "TR-808", 32: "Jazz",
                    40: "Blush", 48: "Orchestra"}
patch2number = {v: k for k, v in MIDI.Number2patch.items()}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}

if __name__ == "__main__":
  
    ckpt = "../../trained-models/skytnt/version_703-la-hawthorne-finetune.ckpt"

    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device='cuda')
    print(f"loading model from  {ckpt}")
    load_model(ckpt)


    input_midi = "input mid"
    input_midi_events = 128

    ins_options = [
                    [[], "None"],
                    [["Acoustic Grand"], "None"],
                    [["Acoustic Grand", "Violin", "Viola", "Cello", "Contrabass"], "Orchestra"],
                    [["Flute", "Cello", "Bassoon", "Tuba"], "None"],
                    [["Violin", "Viola", "Cello", "Contrabass", "Trumpet", "French Horn", "Brass Section",
                      "Flute", "Piccolo", "Tuba", "Trombone", "Timpani"], "Orchestra"],
                    [["Acoustic Guitar(nylon)", "Acoustic Guitar(steel)", "Electric Guitar(jazz)",
                      "Electric Guitar(clean)", "Electric Guitar(muted)", "Overdriven Guitar", "Distortion Guitar",
                      "Electric Bass(finger)"], "Standard"], 
                    [["Acoustic Grand", "Vibraphone", "Electric Guitar(jazz)",
                      "Electric Guitar(clean)", "Electric Guitar(muted)", "Overdriven Guitar", "Distortion Guitar",
                      "Electric Bass(finger)"], "Standard"]
    ]
    input_instruments = ins_options[-1][0]
    # input_drum_kit = list(drum_kits2number.values())[0]            
    input_drum_kit = None
    input_gen_events = 256
    input_temp = 1.0
    input_top_p = 0.98
    input_top_k = 12
    input_allow_cc = True
    input_amp = True
    print("calling run")
    run(input_instruments, input_drum_kit,
        input_midi, input_midi_events,
        input_gen_events, input_temp, input_top_p, input_top_k,
        input_allow_cc, input_amp)

