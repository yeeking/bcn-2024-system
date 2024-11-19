## Script to muck around with the generator

import os
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
from gen_utils import ImproviserAgent, MIDIScheduler, ModelHandler
import mido
import time
import numpy as np 
import MIDI
import torch
import tqdm

number2drum_kits = {-1: "None", 0: "Standard", 8: "Room", 16: "Power", 24: "Electric", 25: "TR-808", 32: "Jazz",
                    40: "Blush", 48: "Orchestra"}
patch2number = {v: k for k, v in MIDI.Number2patch.items()}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}


@torch.inference_mode()
def generate(model, tokenizer, 
            prompt=None, max_len=512, temp=1.0, top_p=0.98, top_k=20,
             disable_patch_change=False, disable_control_change=False, disable_channels=None, amp=True):
    
    print(f"generate (skytnt version) with  disable_patch_change {disable_patch_change}, disable_control_change  {disable_control_change}, disable_channels {disable_channels}")
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



def run(model:MIDIModel, tokenizer:MIDITokenizer, instruments, drum_kit, score_format_prompt, max_len, temp, top_p, top_k, allow_cc, amp):
    print("Here goes run")
    msgs_history = []
    output_sequence = []
    max_len = int(max_len)

    disable_patch_change = False
    disable_channels = None
    print("Run from zero input")
    i = 0
    input_sequence = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
    patches = {}
    if instruments is None:
        instruments = []
    for instr in instruments:
        print(instr)
        patches[i] = patch2number[instr]
        i = (i + 1) if i != 8 else 10
    if drum_kit != None:
        patches[9] = drum_kits2number[drum_kit]
    for i, (c, p) in enumerate(patches.items()):
        input_sequence.append(tokenizer.event2tokens(["patch_change", 0, 0, i, c, p]))
    
    # here ius where you can insert 
    # some more MIDI messages
    print(f"Input sequence detokenised\n{tokenizer.detokenize(input_sequence)}")
    print(f"Raw input sequence {input_sequence}")

    # first trick tokenizer into tokenizing a list of note events from a score
    score_format_prompt = [220, score_format_prompt]
    token_format_prompt = tokenizer.tokenize(score_format_prompt)

    print(f"Token format prompt {token_format_prompt}")
    input_sequence.extend(token_format_prompt[1:-1]) # skip the tempo message if there, remove the weird 'no.2' at the end, Yes no.2[   2,    0,    0,    0,    0,    0,    0,    0]])
    print(f"Token format input and prompt {input_sequence}")

    output_sequence = input_sequence
    input_sequence = np.asarray(input_sequence, dtype=np.int64)


    if len(instruments) > 0:
        disable_patch_change = True
        disable_channels = [i for i in range(16) if i not in patches]


    print("Calling generate")
    generator = generate(model, tokenizer, input_sequence, max_len=max_len, temp=temp, top_p=top_p, top_k=top_k,
                         disable_patch_change=disable_patch_change, disable_control_change=not allow_cc,
                         disable_channels=disable_channels, amp=amp)
    print("Got response. iterating it")
    for i, token_seq in enumerate(generator):
        output_sequence.append(token_seq)
        # event = tokenizer.tokens2event(token_seq.tolist())
        # yield mid_seq, None, None, send_msgs([create_msg("visualizer_append", event), create_msg("progress", [i + 1, gen_events])], msgs_history), msgs_history
    input_sequence = tokenizer.detokenize(output_sequence)
    print("Writing to output.md")
    with open(f"output.mid", 'wb') as f:
        f.write(MIDI.score2midi(input_sequence))
    # soundfont_path = "../../../soundfonts/SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2"
    # audio = synthesis(MIDI.score2opus(input_sequence), soundfont_path)
    # write_audio_to_file(audio, 44100)
    # print("Got audio of length ", audio.shape)
    # return output_sequence, "output.mid", (44100, audio), send_msgs([create_msg("visualizer_end", None)], msgs_history), msgs_history



# def run((model_, tokenizer_, instruments, drum_kit, input_sequence, midi_events, gen_events, temp, top_p, top_k, allow_cc, amp):
def gen_at_random(model_, tokenizer_, score_format_prompt, instruments,max_events_to_generate):
  

    # instruments = ins_options[-1][0]
    # input_drum_kit = list(drum_kits2number.values())[0]            
    input_drum_kit = None
    input_temp = 1.0
    input_top_p = 0.98
    input_top_k = 12
    input_allow_cc = True
    input_amp = True
    print("calling run")
    run(model_, tokenizer_, 
        # instruments=input_instruments, # selected from default sets above
        instruments=instruments, 
        drum_kit=input_drum_kit, # set to none 
        score_format_prompt=score_format_prompt,
        max_len=max_events_to_generate, 
        temp=input_temp, top_p=input_top_p, top_k=input_top_k,
        allow_cc=input_allow_cc, amp=input_amp)

    # run(instruments, drum_kit, input_sequence, midi_events, gen_events, temp, top_p, top_k, allow_cc, amp):

    # print("Calling generate")
    # gen_events = ModelHandler.generate_midi_seq_multi(model_, tokenizer_, 
    #                 score_format_notes=[], 
    #                 output_len=max_events_to_generate) 

    # print("Writing to output.mid")
    # with open(f"output.mid", 'wb') as f:
    #     f.write(MIDI.score2midi(gen_events))

def test_different_lengths(model_, tokenizer_, input_events_):
    # max_len = len(input_events[1])
    for max_len in [8, 16, 32, 64, 128]:
        # max_len = 64
        gen_events = ModelHandler.generate_midi_seq(model_, tokenizer_, 
                            input_events_,
                            output_len=max_len, # generate as much as we give you
                            temp=0.7, 
                            top_p=0.5, #0.1 to 1.0
                            top_k=1, #1 to 20 
                            allow_cc=False, # True or False
                            amp=True, use_model=True, show_bar=False) 
        # print(f"Done with gen. Len of output {len(gen_events[1])} but max len {max_len}")
        print(f" at 120 BPM max len {max_len}, last beat at this offset in seconds {gen_events[1][-1][1] / 480 / 120 * 60}")

def gen_continuation(model_, tokenizer_, input_events_, max_len_):
    """
    generate a continuation of the sent input_events
    """
    # event_count = np.sum([c for c in input_events])
    print(f"gen_midi_file calling model... with {len(input_events_[1])} events")
    gen_events = ModelHandler.generate_midi_seq_multi(model_, tokenizer_, 
                            input_events_,
                            output_len=max_len_, # generate as much as we give you
                            temp=0.7, 
                            top_p=0.5, #0.1 to 1.0
                            top_k=1, #1 to 20 
                            allow_cc=True, # True or False
                            amp=True, use_model=True, show_bar=False) 
    out_file = 'output.mid'
    # print(f"Saving {len(gen_events[1])} events to {out_file}")
    with open(out_file, 'wb') as f:
        f.write(MIDI.score2midi(gen_events))

def get_notes_and_patch_changes(midi_file, length):
    """
    read midi from sent file and filter it to note and patch change messages
    then return 'length' events. 
    """
    assert os.path.exists(midi_file)
    print(f"Reading input events from {midi_file}")
    with open(midi_file, 'rb') as file:
        # Read the binary data from the file
        raw_midi = file.read()
    print(f"Converting {len(raw_midi)} bytes to score format...")
    events = MIDI.midi2score(raw_midi)
    # input_events[0] = input_events[0]
    events[1] = [i for i in events[1] if (i[0] == 'patch_change') or (i[0] == 'note')]
    events[1] = events[1][0:length] # trim 
    return events

def get_events_of_type(midi_file, event_type):
    """
    returns all events of the sent type from the sent midi_file
    """
    assert os.path.exists(midi_file)
    print(f"Reading {event_type} events from {midi_file}")
    with open(midi_file, 'rb') as file:
        # Read the binary data from the file
        raw_midi = file.read()
    events = MIDI.midi2score(raw_midi)
    events = [i for i in events[1] if i[0] == event_type]
    return events

    

def get_random_patch_changes(tpb):
    """
    Get a list of random patch change events across all MIDI channels
    """
    events = []
    # now some patch changes, one on each channel, random values
    # for i, (c, p) in enumerate(patches.items()):
    #     mid.append(tokenizer.event2tokens(["patch_change", 0, 0, i, c, p]))
    
    for ch in range (1, 16): events.append([['patch_change', 0, ch, np.random.randint(0, 120)]])
    return events

def get_copy_events():
    """
    events as they were generated by the 'gen_ranom' script which generates great stuff!
    """
    events = [480, [['patch_change', 0, 0, 0]], [['patch_change', 0, 1, 11]], [['patch_change', 0, 2, 26]], [['patch_change', 0, 3, 27]], [['patch_change', 0, 4, 28]], [['patch_change', 0, 5, 29]], [['patch_change', 0, 6, 30]], [['patch_change', 0, 7, 33]]]
    return events


if __name__ == "__main__":
    ckpt = "../../trained-models/skytnt/version_703-la-hawthorne-finetune.ckpt"
    # ckpt = "../../trained-models/skytnt/skytnt-hf-model-la-dataset.ckpt"
    assert os.path.exists(ckpt), "Cannot find checkpoint file " + ckpt
    print("Creating model and loading to GPU")
    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device='cuda')
    print("Loading model weights")
    ModelHandler.load_model(ckpt, model)
    multitrack_midi_file = 'multitrack-midi.mid'
    piano_midi = 'mark_intro.mid'
    # midif = 'input_long.mid'
    # patch_events = get_events_of_type(multitrack_midi_file, "patch_change")
    tpb = 384
    patch_events = get_random_patch_changes(tpb) # [[pc1]], [[pc2]]]
    piano_notes = get_events_of_type(piano_midi, "note")  # [[note1], [note2]]
    input_events = [960]
    

    print(f"Sending input to model {input_events} ")



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
                     [["Acoustic Grand", "Vibraphone"], "Standard"],
                    [["Acoustic Grand", "Vibraphone", "Electric Guitar(jazz)",
                      "Electric Guitar(clean)", "Electric Guitar(muted)", "Overdriven Guitar", "Distortion Guitar",
                      "Electric Bass(finger)"], "Standard"]
    ]
    # # input_events[1] = input_events[0:256] # trim 
    # instruments = ["Acoustic Grand", "Vibraphone", "Electric Guitar(jazz)",
    #                   "Electric Guitar(clean)", "Electric Guitar(muted)", "Overdriven Guitar", "Distortion Guitar",
    #                   "Electric Bass(finger)"]
    # instruments = ["Acoustic Grand", "Vibraphone", "Electric Guitar(jazz)",
    #                   "Electric Guitar(clean)", "Electric Guitar(muted)",
    #                   "Electric Bass(finger)"]
    # test_different_lengths(model, tokenizer, input_events)
    # gen_continuation(model, tokenizer, input_events, 256)

    instruments = ins_options[6][0]
    gen_at_random(model_=model, tokenizer_=tokenizer, 
                  score_format_prompt=piano_notes,
                  instruments=instruments, max_events_to_generate=512)
