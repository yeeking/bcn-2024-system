import threading
import mido 
import midi_tokenizer
import midi_model
import time 
import numpy as np
import torch 
from midi_model import MIDIModel, MIDITokenizer
import tqdm
import os
import copy 

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

    print(f"Entering model forard call loop. in tensor shape: {input_tensor.shape}")
    with bar, autocast(enabled=amp):
        while cur_len < max_len:
            print(f"Calling forward length is {cur_len} of {max_len} input shape is {input_tensor.shape} ")
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


def generate_midi_seq(model:MIDIModel, tokenizer:MIDITokenizer, score_format_input, gen_events, temp, top_p, top_k, allow_cc, amp):
    """
    generate a midi sequence using the sent models and  params 
    """
    # prepare variables
    mid_seq = []
    gen_events = int(gen_events)
    max_len = gen_events
    disable_patch_change = False
    disable_channels = None

    tokens = tokenizer.tokenize(score_format_input)

    mid = np.asarray(tokens, dtype=np.int64)
    print(f"Final midi format for model. Shape: {mid.shape}")
    # mid = mid[:int(max_input_len)] # if want to use a subset of the inputs 

    ## this bit adds the input sequence to the start of the output sequence
    ## don't want that so commented out 
    # max_len += len(mid)
    # for token_seq in mid:
    #     mid_seq.append(token_seq.tolist())
    
    generator = generate(model, tokenizer, mid, max_len=max_len, 
                        temp=temp, top_p=top_p, top_k=top_k,
                        disable_patch_change=disable_patch_change, disable_control_change=not allow_cc,
                        disable_channels=disable_channels, amp=amp)
    for i, token_seq in enumerate(generator):
        print(f"Gen step {i} of {len(token_seq)}")
        mid_seq.append(token_seq)
        ## this bit is for outputting as you generate ... might be useful! 
        # event = tokenizer.tokens2event(token_seq.tolist())

    score_format_data = tokenizer.detokenize(mid_seq)
    return score_format_data


class RingBuffer:
    """
    circular / ring buffer
    """
    def __init__(self, size):
        self.size = size
        self.lock = threading.Lock()
        self.reset()

    def addEvent(self, event):
        with self.lock:
            self.array[self.index] = event
            self.index = (self.index + 1) % self.size

    def getEvents(self):
        with self.lock:
            return list(self.array)
    
    def isFull(self):
        """
        return true if the index is pointing to the last position
        """
        if self.index == len(self.array) - 1:
            return True
        else:
            return False
        
    def reset(self):
        with self.lock:
            self.index = 0
            self.array = [None] * self.size

class MidoWrapper():
    def __init__(self, midiCallback):
        self.midiCallback = midiCallback
        self.input_device = None
        self.output_device = None
        self.output_port = None
        self.output_lock = threading.Lock()  # Lock to ensure thread safety for output

    def initMIDI(self):
        mido.set_backend('mido.backends.rtmidi')
        
        # Get and choose MIDI input device
        print("Available MIDI input devices:")
        midi_input_devices = mido.get_input_names()
        for i, device in enumerate(midi_input_devices):
            print(f"{i}: {device}")

        input_device_index = int(input("Enter the number of the MIDI input device you want to connect to: "))
        assert (input_device_index >= 0) and (input_device_index < len(midi_input_devices)), "Invalid MIDI input device chosen"
        self.input_device = midi_input_devices[input_device_index]
        print(f"Connecting to MIDI input: {self.input_device}")

        # Get and choose MIDI output device
        print("\nAvailable MIDI output devices:")
        midi_output_devices = mido.get_output_names()
        for i, device in enumerate(midi_output_devices):
            print(f"{i}: {device}")

        output_device_index = int(input("Enter the number of the MIDI output device you want to connect to: "))
        assert (output_device_index >= 0) and (output_device_index < len(midi_output_devices)), "Invalid MIDI output device chosen"
        self.output_device = midi_output_devices[output_device_index]
        print(f"Connecting to MIDI output: {self.output_device}")

    def start(self):
        # Start MIDI input thread
        input_thread = threading.Thread(target=self.midi_input_thread_function)
        input_thread.start()

        # Start MIDI output thread (to handle output connection)
        output_thread = threading.Thread(target=self.midi_output_thread_function)
        output_thread.start()

    def midi_input_thread_function(self):
        with mido.open_input(self.input_device, callback=self.midiCallback) as inport:
            # input("MIDI input open! Press Enter to stop listening...\n")
            while True:
                time.sleep(1)  # Keeps the thread running

    def midi_output_thread_function(self):
        # Open MIDI output
        with mido.open_output(self.output_device) as outport:
            self.output_port = outport
            print("MIDI output open! You can now send messages to the output device.")

            # Keep the thread alive to allow sending messages
            while True:
                time.sleep(1)  # Keeps the thread running

    def sendMIDI(self, message: mido.Message):
        """Send a mido.Message to the output device."""
        with self.output_lock:
            if self.output_port:
                self.output_port.send(message)
                print(f"Sent MIDI message: {message}")
            else:
                print("MIDI output port is not initialized.")

# class MidoWrapper():
#     def __init__(self, midiCallback):
#         self.midiCallback = midiCallback
   
#     def initMIDI(self):
#         mido.set_backend('mido.backends.rtmidi')
#         print("Available MIDI input devices:")
#         midi_devices = mido.get_input_names()
#         for i, device in enumerate(midi_devices):
#             print(f"{i}: {device}")

#         device_index = int(input("Enter the number of the MIDI device you want to connect to: "))
#         # could do something more elegant where they get another chance... 
#         assert (device_index >= 0) and (device_index < len(midi_devices)), "Invalid MIDI device chosen"
#         selected_device = midi_devices[device_index]
#         print(f"Connecting to: {selected_device} callback {self.midiCallback}")

#         def midi_thread_function():
#             with mido.open_input(selected_device, callback=self.midiCallback) as inport:
#                 input("MIDI open! Press Enter to stop listening...\n")

#         # Start a new thread to run the thread_function
#         thread = threading.Thread(target=midi_thread_function)
#         thread.start()


class ImproviserAgent():
    def __init__(self, memory_length:int, model:MIDIModel, tokenizer:MIDITokenizer):
        self.midoWrapper = MidoWrapper(self.receiveMIDI)
        self.noteBuffer = RingBuffer(memory_length)
        self.start_time_s = time.time()
        self.bpm = 120 # not sure what to do with this one! 
        self.ticks_per_beat = 480 # 96 is a typical old-skool MIDI file tpb. 480 is more modern.
        self.setModel(model)
        self.tokenizer = tokenizer
        self.lock = threading.Lock()
        
    def setModel(self, model:midi_model.MIDIModel):
        self.model = model 

    def initMIDI(self):
        self.midoWrapper.initMIDI()
        self.midoWrapper.start()


    def receiveMIDI(self, msg:mido.Message):
        if msg.type == "note_on":
            print(f"ImproviserAgent receiveMIDI note on {msg}")
            # ['note', start_time, duration, channel, note, velocity]
            # ["note", offset_in_ticks, duration_in_ticks, channel, note, velocity ]
            # time since the start of this generation cycle 
            offset_secs = time.time() - self.start_time_s
            offset_in_ticks = (offset_secs / (60/self.bpm)) * self.ticks_per_beat
            event = ['note', int(offset_in_ticks), self.ticks_per_beat, 0, msg.note, msg.velocity]
            self.noteBuffer.addEvent(event)
            if self.noteBuffer.isFull():# generate when the buffer is full
                self.generate()


    def analyse_output(self, output):
        """
        print some info about the de-tokenized output of the model
        """
        if len(output) < 2:
            print(f"Bad output {output}")
            return 
        tpb = output[0]
        ch_count = len(output) - 1 # first entry is not a channel it is tpb
        ch1_events = len(output[1])
        beat_offsets = [round(eve[1]/tpb, 2) for eve in output[1] if eve[0] == 'note']

        print(f"Received {ch_count} channels on ch1 {ch1_events} notes, tpb: {tpb} offsets {beat_offsets}")
        

    def generate(self):
        """
        convert the current contents of the ring buffer into 
        a score format event list as in MIDI.py score 
        then tokenize it and send it to the model 
        """
        with self.lock:
        # remove 'None' type events and pre-pend ticks per beat
        # note that I call copy.copy as the underlying lists might get modified 
        # by incoming midi events on a separate thread.
            input_events = [copy.copy(e) for e in self.noteBuffer.array if e != None]
            if len(input_events) == 0:
                print("No context yet...")
                return  
            # now reset the ring buffer so we can capture more events whilst they continue to play
            self.noteBuffer.reset()
            # and reset the start time for the events we will now capture 
            self.start_time_s = time.time() # reset start time for next frame 
            print(f"Sending {len(input_events)} to the model")
            input_events = [self.ticks_per_beat] + [input_events]
            # output_events = self.generate(input_events)
            gen_events = generate_midi_seq(self.model, tokenizer, 
                        input_events,
                        gen_events=self.noteBuffer.size, 
                        temp=0.7, 
                        top_p=0.5, #0.1 to 1.0
                        top_k=1, #1 to 20 
                        allow_cc=False, # True or False
                        amp=True) # True or False
            # now can generate some data
            return gen_events

        
    def run(self):
        # Define the function that will run in the thread
        def thread_function():
            while True:
                time.sleep(5)  # Wait for note collection
                self.generate() # try to generate every x seconds regardless of what has come in    
        # Start a new thread to run the thread_function
        thread = threading.Thread(target=thread_function)
        thread.start()

if __name__ == "__main__":
    # ckpt = "small.ckpt"
    ckpt = "../models/small.ckpt"
    midi_file = 'input.mid'

    assert os.path.exists(ckpt), "Cannot find checkpoint file " + ckpt
    assert os.path.exists(midi_file), "Cannot find MIDI file " + midi_file

    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device='cpu')
    load_model(ckpt, model)
    print(model)

    improviser = ImproviserAgent(memory_length=128, model=model, tokenizer=tokenizer) 
    improviser.initMIDI()
    improviser.run()
# improviserGlobal.initMIDI()

