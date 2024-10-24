# import torch
import MIDI
import numpy as np
import os 
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
import tqdm
import onnxruntime as rt

number2drum_kits = {-1: "None", 0: "Standard", 8: "Room", 16: "Power", 24: "Electric", 25: "TR-808", 32: "Jazz",
                    40: "Blush", 48: "Orchestra"}
patch2number = {v: k for k, v in MIDI.Number2patch.items()}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}


def softmax(x, axis):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def sample_top_p_k(probs, p, k):
    probs_idx = np.argsort(-probs, axis=-1)
    probs_sort = np.take_along_axis(probs, probs_idx, -1)
    probs_sum = np.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    mask = np.zeros(probs_sort.shape[-1])
    mask[:k] = 1
    probs_sort = probs_sort * mask
    probs_sort /= np.sum(probs_sort, axis=-1, keepdims=True)
    shape = probs_sort.shape
    probs_sort_flat = probs_sort.reshape(-1, shape[-1])
    probs_idx_flat = probs_idx.reshape(-1, shape[-1])
    next_token = np.stack([np.random.choice(idxs, p=pvals) for pvals, idxs in zip(probs_sort_flat, probs_idx_flat)])
    next_token = next_token.reshape(*shape[:-1])
    return next_token


def generate(onnx_model:rt.capi.onnxruntime_inference_collection.InferenceSession, 
            onnx_tokenizer:rt.capi.onnxruntime_inference_collection.InferenceSession, 
            py_tokenizer:MIDITokenizer,
        prompt=None, max_len=512, temp=1.0, top_p=0.98, top_k=20,
             disable_patch_change=False, disable_control_change=False, disable_channels=None):
    if disable_channels is not None:
        disable_channels = [py_tokenizer.parameter_ids["channel"][c] for c in disable_channels]
    else:
        disable_channels = []
    max_token_seq = py_tokenizer.max_token_seq
    if prompt is None:
        input_tensor = np.full((1, max_token_seq), py_tokenizer.pad_id, dtype=np.int64)
        input_tensor[0, 0] = py_tokenizer.bos_id  # bos
    else:
        prompt = prompt[:, :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = np.pad(prompt, ((0, 0), (0, max_token_seq - prompt.shape[-1])),
                            mode="constant", constant_values=py_tokenizer.pad_id)
        input_tensor = prompt
    input_tensor = input_tensor[None, :, :]
    cur_len = input_tensor.shape[1]
    bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
    with bar:
        while cur_len < max_len:
            end = False
            hidden = onnx_model.run(None, {'x': input_tensor})[0][:, -1]
            next_token_seq = np.empty((1, 0), dtype=np.int64)
            event_name = ""
            for i in range(max_token_seq):
                mask = np.zeros(py_tokenizer.vocab_size, dtype=np.int64)
                if i == 0:
                    mask_ids = list(py_tokenizer.event_ids.values()) + [py_tokenizer.eos_id]
                    if disable_patch_change:
                        mask_ids.remove(py_tokenizer.event_ids["patch_change"])
                    if disable_control_change:
                        mask_ids.remove(py_tokenizer.event_ids["control_change"])
                    mask[mask_ids] = 1
                else:
                    param_name = py_tokenizer.events[event_name][i - 1]
                    mask_ids = py_tokenizer.parameter_ids[param_name]
                    if param_name == "channel":
                        mask_ids = [i for i in mask_ids if i not in disable_channels]
                    mask[mask_ids] = 1
                logits = onnx_tokenizer.run(None, {'x': next_token_seq, "hidden": hidden})[0][:, -1:]
                scores = softmax(logits / temp, -1) * mask
                sample = sample_top_p_k(scores, top_p, top_k)
                if i == 0:
                    next_token_seq = sample
                    eid = sample.item()
                    if eid == py_tokenizer.eos_id:
                        end = True
                        break
                    event_name = py_tokenizer.id_events[eid]
                else:
                    next_token_seq = np.concatenate([next_token_seq, sample], axis=1)
                    if len(py_tokenizer.events[event_name]) == i:
                        break
            if next_token_seq.shape[1] < max_token_seq:
                next_token_seq = np.pad(next_token_seq, ((0, 0), (0, max_token_seq - next_token_seq.shape[-1])),
                                        mode="constant", constant_values=py_tokenizer.pad_id)
            next_token_seq = next_token_seq[None, :, :]
            input_tensor = np.concatenate([input_tensor, next_token_seq], axis=1)
            cur_len += 1
            bar.update(1)
            yield next_token_seq.reshape(-1)
            if end:
                break


# def generate_midi_seq(model, tokenizer, midi_filename, n_events_from_file, instruments, drum_kit, mid, midi_events, gen_events, temp, top_p, top_k, allow_cc):
def generate_midi_seq(onnx_model:rt.capi.onnxruntime_inference_collection.InferenceSession, 
                      onnx_tokenizer:rt.capi.onnxruntime_inference_collection.InferenceSession, 
                      py_tokenizer:MIDITokenizer, 
                      midi_filename, n_events_from_file, max_len, temp, top_p, top_k, allow_cc):

# def generate_midi_seq(model:MIDIModel, tokenizer:MIDITokenizer, midi_filename, n_events_from_file, gen_events, temp, top_p, top_k, allow_cc, amp):
    """
    generate a midi sequence using the sent params 
    """
    mid_seq = []
    max_len = int(max_len)

    disable_patch_change = False
    disable_channels = None

    assert os.path.exists(midi_filename)

    with open(midi_filename, 'rb') as file:
        # Read the binary data from the file
        mid = file.read()

    mid = py_tokenizer.tokenize(MIDI.midi2score(mid))
    mid = np.asarray(mid, dtype=np.int64)
    mid = mid[:int(n_events_from_file)]
    max_len += len(mid)
    for token_seq in mid:
        mid_seq.append(token_seq.tolist())
    
    # yield mid_seq, None, None, send_msgs(init_msgs, msgs_history), msgs_history    
    generator = generate(onnx_model, onnx_tokenizer, py_tokenizer, mid, 
                         max_len=max_len, temp=temp, top_p=top_p, top_k=top_k,
                         disable_patch_change=disable_patch_change, 
                         disable_control_change=not allow_cc,
                         disable_channels=disable_channels)
    
    for i, token_seq in enumerate(generator):
        mid_seq.append(token_seq)
        # event = tokenizer.tokens2event(token_seq.tolist())
    
    mid = py_tokenizer.detokenize(mid_seq)
    with open(f"output.mid", 'wb') as f:
        f.write(MIDI.score2midi(mid))
    
def run():
    model_base_path = 'model_base.onnx'
    model_token_path = 'model_token.onnx'
    midi_file = 'input.mid'

    assert os.path.exists(model_base_path), "Cannot find checkpoint file " + model_base_path
    assert os.path.exists(model_token_path), "Cannot find checkpoint file " + model_token_path
    
    assert os.path.exists(midi_file), "Cannot find MIDI file " + midi_file
    print(f"Generating from file {midi_file} with ckpt {model_base_path}")

    py_tokenizer = MIDITokenizer()
    
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    
    try:
        model_base = rt.InferenceSession(model_base_path, providers=providers)
        model_token = rt.InferenceSession(model_token_path, providers=providers)
    except Exception as e:
        print(e)
        input("Failed to load models, maybe you need to delete them and re-download it.\nPress any key to continue...")
        exit(-1)

    print("tokenizer is", type(model_token))  
    generate_midi_seq(model_base, model_token, py_tokenizer, midi_file, 
                      n_events_from_file=128, 
                      max_len=128, 
                      temp=0.7, 
                      top_p=0.5, #0.1 to 1.0
                      top_k=1, #1 to 20 
                      allow_cc=False) # True or False
    
if __name__ == "__main__":
    run()
