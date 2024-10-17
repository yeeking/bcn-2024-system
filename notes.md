# Notes on building Barcelona system

# 17/10/2024

Working on new input output length logic. 
DONE * Add a function to the ring buffer so it always sends the n most recent items, even when it wraps around, noting that it writes new items at index and increments index. So most recent n items = 

* verify can adjust input and output with the UI

* Mix in for auto-regression... so it has memory of its own notes as well as human's - better to continue ideas?



# 16/10/2024

## Finalising mark's system for Sat

X - sort out the double midi init thing on live_web script
  -> couldn't quite figure out why it calls that code twice, but pushed initMIDI to the start button callback
- sort out the input controls so can vary length of input (and output?)
x- try audio interface on the other USB socket (right) for a laugh tried a thunderbolt cable. Seemed to be solid @441k 256,n=3 
- make sure markov system plugins available
- make some notes about the model structure

## Moving on from the ring buffer

* Ring buffer thing: 
  - maybe just have a long event memory (4096 since that's the max. context) and then choose how much of it to pass in as context
  - so the memory can have a 'getMemory(length_in_seconds)'
    -> that passes back the last 'length_in_seconds' seconds of notes
  * then as a counterpart, when generating we might figure out how to stop when we get a certain length
     into the future. But the input is the first and easiest one to crack

=======
>>>>>>> 2fc72fd3a4baa6df9ef120176af00c6d4d61514b
# 14/10/2024

Working on improved generator and midi note control. Main issues:

- MIDI messages in context are stored in a ring buffer that does not know about time
  -> add a mode to the ring buffer where it only wraps after a certain time has passed
- generation always triggered every x seconds
DONE  -> at least add a 'do not regen if you are still playing out the previous output'
- output from gen seems to last quite a long time
  -> if waiting for end before re-gen, might get a bit tedious. 
DONE   -> maybe two modes... overlap or not overlap

DONE - coupling between input lengtha and output length
  -> at least add a 'do not regen if you are still playing out the previous output'
- output from gen seems to last quite a long time
  -> if waiting for end before re-gen, might get a bit tedious. 
   -> maybe two modes... overlap or not overlap

- coupling between input lengtha and output length
  -> input length seems to be tied to output length
   -> ideally, I would want them independent. E.g. having input context be long enough for the last 5 seconds of notes 
   -> I fixed this with the import torch.functional as F call which previously crashed it (duh)

## Notes on test_gen.py

I wrote a small test script to generate from the model in various ways. I totally broke the sytem at one point after some careless hacking around in the generator code but i did also idenify some weirdness there and renamed some variables to make it more readable. Basically, the length of time notes go into the future is proportional to the number of times you call forward, ultimately. Which kinda makes sense. Here's some figures:

 at 120 BPM max len 8, last beat at this offset in seconds 1.96875
 at 120 BPM max len 16, last beat at this offset in seconds 3.5625
 at 120 BPM max len 32, last beat at this offset in seconds 7.4375
 at 120 BPM max len 64, last beat at this offset in seconds 16.09375
 at 120 BPM max len 128, last beat at this offset in seconds 35.78125

So in an improvisation context, you might want to vary the output length between 8 and 32 to get a 1->7 second range of output. 

Then you can vary the input length as well, e.g. have very short input and very short output for a highly interactive mode 


# 12/10/2024

## Removing overlap events

I think overlapping events are a problem. In fact the midi and the way it manages midi events is a problem too. Fix those! But keep the golden version oIs it f the system as a branch in case it was a freakish hit :) 

## Audio quality

Was hearing too-frequent zipper noises in the audio. Tried stripping out all the plugins (esp. yabridge windows ones). I /think/ that running jackd from the command line instead of qjackctl improved things - presume the default settings on CLI work better? Tried 44k, 88k, 96k. Did not hear it on 44 or 88 but did on 96, though I was abusing MIDI device in the 96k test. In fact, it seems the a2j thing i use to get MIDI into reaper is complaining at the end of the xrun blitz. Very difficult to debug since I cannot reproduce the issue with any consistency. Not great. Is it yabridge? Is it heavy and iffy MIDI data from the model? 

## Black screening

Black screen happened twice or three times. Not good but the screen mode option seems to fix it,though not clear if that'll work when only on laptop screen. Could it be the auto VRAM thing in the bios? Maybe just set it to 2gig and see if it happens? 

## Some quick tests with the multi-track model

Tried the non-finetuned model. Behaviour obviously very different. Probably not as good - tends to go for lots of dramatic scales in response to freestyle sax.

# 11/10/2024

## Pitch to MIDI

After convincing proof of concept session with Mark, I now am happy to go live for the Mark presentations with the system more or less 'as-is'. 

Then I switched focus to the Finn version of the algorithm. Did a lot of mucking the the aim of getting a monophonic audio->MIDI plugin going, to see how good it was. I could then make a decision about the approach to take for live audio->algorithm connection. I found a free windows plugin called DODO MIDI 2 which works pretty well with the 'pre-gain' dialed down on Finn's sax signals. Tricky bit was getting windows plugins to run well on Linux. Did that with a tool called yabridge which seems to work quite well. I am getting occasional audio glitches but only (I think) when I muck about with my plugin chains whilst playing back audio. It'll probably be ok. Worst case, I can just run the plugins in actual windows. I'll keep doing it in linux and see how it goes. 

I managed to get quite accurate and responsive transcription of finn's sax samples from dod midi with a slightly reduced 'pre-gain' value. 

## Better pianos

I also did a bit of searching for better piano sounds. Found what claimed to be a detailed sampling of a nice grand in sfz format. Then found a windows plugin called 'sforzando' that can load sfz format. Seemed to be really quiet though, so had to put a slightly scary volumne booster plugin in the chain. Does sound quite rich and dynamic though. Also found that the LA dataset has another piano sound font which is actually pretty good and plays in linux native juicy sound font. It has proper volume levels and cuts through a bit more, so i'll probably use that one. Might look for a linux native sfz plugin though.

Conclusion: the windows sfz player with the big piano sounds reallu good but is quiet/ posibly over-dynamic?? and I'm a bit dubious of that windows-linux bridge stuff. '

This one https://github.com/osxmidi/SFZero-X/releases/tag/0.6 is a native linux sfz player but it pre-loads all the samples, unlike the progressive loading sforzando meaning it makes the DAW project take forever to load, especially if you have two instance. 

The LA dataset has some decent pianos as sound fonts that work well in juicysf plugin, so probably just use those! 


# 10/10/2024

Preparing for first demo with Mark today. Want to work on some parameterisation of the behaviour of the system ideally controlled via a GUI, noting my list of random thoughts below after first successful test run of live improv. 

Things to do for Mark session:

DONE  * Fix stuck note-off in Q problem
DONE  * Build rudimentary UI for live_gen
  
NOT done  * GUI: Note density (% chance any given note is played)

DONE  * GUI: Time filter - only allow notes up to a certain time in the future 

NOT done but figured out solution  * GUI: time frame: do not use ring buffer, instead just capture everything in a time frame without wrapping. could just make ring buffer very long
  
## Side note

Since I started work on the system, SkyTNT suddenly started updating the code after a 1 year hiatus. There is a v2 midi tokenizer which seems to include something about tempo information (presume that requires a re-train) and some sort of caching (kv cache) that apparently improves performance. My main performance bottleneck is the inference so not sure how much i'd gain from that. Still, TODO: clone latest repo and do some speed tests on inference to see how it goes. There is also a citation bibtex on the github repo which is good. 

Also, I would like to try my own tokenizer eventually, more events and rests than notes and durations style. I wonder if the whole 'tokenize notes with offset and durations' thing is just a hack to solve time problem i.e. putting notes into the future is easier as the tokenisation explicitly encodes offset from now and duration. This is quite different than natural language modelling as that has no sense of how long a word lasts aside from the number of tokens that make up a word maybe? Interesting stuff there. Would be good to write about the different tokenization approaches as it has a 'history of music tech.' angle to it as well. 

# 08/10/2024

Today was a big day as I started testing the live interaction experience. Setup was as follows:

- nvidia machine runs the model - sends and receives MIDI
- model: la pre-trained model fine tuned on hawthorne
- memory length 32
- thinkpad runs reaper with two channels, plugin piano on each channel 

Results are really cool. But lots of things to dial in/ options to put on a 'user interface'

## Lots of thoughts about how to customise/ improve behaviour

- sense of me being in control 
  * lots of things to think about here. Too many notes, notes not in key, notes not clearly related to my notes, better control of what is sent as the input, better timing of when notes come back out. See below! 

- does not always stay in key, but it is quite good if you have 'advanced tonal acceptance' :)
  *  Solution: Need to have a key detector that checks the input prior to sending it to the model. Estimates the key + mode and limits output to notes in that mode (optionally). Could shift notes to nearest correct note. 

- sometimes plays a lot of notes.
  * Solution: note probability on output is very simple solution. 
  * Or some smarter way to filter? e.g. identify only chords and play those or non-chords and play those

- how far do notes go into the future? and how many notes come out each 'forward'
  * seems to place notes quite far into the future and to generate quite a lot of notes each 'forward'

- how often to retrigger inference
  - think that also relates to comments below about clearing the input buffer and sending notes out during inference. 

- when to clear the input buffer
  * input buffer is based on number of notes, not time. Might make more sense to time slice the input buffer so to start overwriting when a length of time has passed as opposed to when a number of notes have come in. Should be a case of newest note time - oldest note time > max time? set write index to zero each time a new note comes in 

- can you send it a longer input sequence and only generate a short output sequence?
  - see comment below about sending notes as soon as first 'forward' call returns.

- can you start sending notes as soon as they come out during sliding inference? 
  * Right now it generates 128 or whatever frames of inference then sends all notes, ok as notes are placed into the future, but can it send notes as they come out? That way you just need to manage the auto-regressive input frame quite well, possibly mixing in not just the output but also new inputs received from MIDI

- it plays at the same time as you
  * note density metric for live input: decide when to play/ more subtle: auto-set output note density to complement input note density 

- different models. 
  * Really want to try the multichannel model!
  * Check out how it goes about selecting channels in the generator app code. 

# 06/10/2024

## MIDI Q

Did a quick bit of GPT + hacking to get a midi scheduler working that can Q MIDI events coming out of the model and send them out in the future at the correct time offset. 

Works ok but note off seems to get stuck in the Q. 


## A bit of performance testing

I gots to know how fast my current gen_live script runs on my nvidia machine. 

Answer: 30 it/s, so 128 takes 4 seconds. 

On the same machine with CPU, seems to have a big lag kicking off inference each time, and then it goes at 8 it/s, with the delay, that goes to 4 it/s. So there you go. 

## Length of generation weirdness

Next I find that the length of stuff it generates is max length - length of input. Not sure where I got to with that so its one to keep digging into.


# 04/10/2024

Analysing the tokenizer removing notes problem.

Here is the input and output format for the tokenizer:
```
['note', 155, 465, 0, 55, 41] #input in score format ['note', start_time, duration, channel, note, velocity] 
['note', 0, 11, 0, 34, 0, 55, 41]#output ready for model
```

Then looking at the duplicates 

```
['note', 6, 11, 0, 16, 0, 60, 0] # this is the input
('note', 6, 11, 0, 0, 60)  # this is the lookup key
['note', 6, 11, 0, 16, 0, 60, 39] # this is the pre-existing item with that key.
```

The last digit of the 'matching' item is not the same. That appears to be the velocity, so according to this, I am sending it the same note at the same time but with a zero velocity and it is filtering it out. That makes sense as a filter, but why am i sending zero velocity notes? 

In the end, I did not really work out why reaper was sending zero velocity note ons, but my midi keyboard does not do that, so I am assuming it is something odd in the MIDI files I am playing out of Reaper. I made the tokenizer replace zero velocity note ons with > 0 ones if they come in at the same time with the same note. 

## MIDI state thing

Next step: deal with MIDI note offs and correct note durations? Or try to play the output of the model right now? 

- midi note offs: I'll need a stateful midistate thing that tells me when note offs come in. Like I had in the C++ improviser in the end. lets do that 

This is sorted now.

## Length of generation weirdness

Next I find that the length of stuff it generates is max length - length of input. Not sure where I got to with that so its one to keep digging into.

# 03/10/2024

## Expressing the runtime as an ImproviserAgent class. 

MIDI Callbacks are an issue. We specify a callback function that gets called when MIDI is received but if that callback function is a non-static member of a class, then how do we ensure that the instance object's state (self) is available to that callback? Probably we cannot, so we have to rely on global scope objects in the callback, presumbly. 

## Completed today:

* Got a basic improviser working with MIDI I/O, ring buffered MIDI memory and sending MIDI data to tokenizer, then to the model. Cool!
* TODO: work out why the tokenizer is rejecting some of the MIDI data from the buffer with its key setting, i.e. why do different MIDI events get the same key? Possibly to do with the time stamps being the same for same notes or something. Check! 

# 01/10/2024

## runtime for the model

Start by collecting notes from the player and using them to fill up slots a fixed size vector
[n1 -- props, n2 --- props, n3 --- props]

Each time a note comes in, we put it in the vector at a new slot. Once it is full, wrap around t othe start and start overwriting. 

In a separate thread, on a regular clock, send the vector into the model. Ensure it has blanks at the start. Read the output of the model. 

Take the output of the model and send that out for MIDI playback straight away
Also add the output into the slots of the vector. If 

On a regular clock ? we send that frame to the model. Probably, depending on how fast it can run. 

Model generates output. 



### Conclusion about input and output to the tokenizer

#### input 
Start time in ticks:

Assuming you have offset_in_seconds, bpm (sent as set_tempo message), ticks_per_beat (set as the first message in every frame you send to the model)
 
```
offset_in_ticks = (offset_in_seconds / (60/bpm)) * ticks_per_beat 
```
Duration in ticks: 
```
duration_in_ticks = duration_in_seconds * (60/bpm) * ticks_per_beat 
```
Gives you:
```
['note', offset_in_ticks, duration_in_ticks, channel, note, velocity]
```

So a complete frame of notes would be:

```
[ticks_per_beat, 
[["set_tempo", tempo in microseconds per quarter note]], 
[["note", offset_in_ticks, duration_in_ticks, channel, note, velocity ]]
]
```

tempo in microseconds per quarter note = 60,000,000 / bpm

#### sending it to the model
Once you have:
```
["note", offset_in_ticks, duration_in_ticks, channel, note, velocity ]
```
You can pass it to the tokenizer then into the model. But... you want to provide the model with a decent context, not just one frame. So you need to gather a frame of notes then pass that to the model. 

What I'm not sure about is what to put in the frame. You could wait for 2 seconds then send it to get another 2 seconds. But what if you only get one event - you are only giving it one event of context, but what about all the silence? The model doesn't model silence or does it pad with empty slots up to a fixed size? It must do! 



#### output 

Then coming out of the model we get:

```
#[type, bar_offset_from_now, beat_offset_in_bar, track_id, duration, channel, note, velocity]

```
which in order to play it, you want to convert to:

```
#[type, seconds_offset, track_id, duration, channel, note, velocity]
```

### Moving on from coreml for now

Need to make some progress with the live MIDI input today. 

Progress made on coreml yesterday: 

* I explored coreML and it seems to work well, going 5-10x as fast as CPU pytorch on my various macs.
* I created a basic C++ project that is able to load a coreml model and run some numbers through it. 
* I nearly completed a C++ project that loads in two separate models (tokenizer and base model) like the app_onnx python example and passes data from one to the other. Perhaps complete this, then switch back to live MIDI input?

-> do a quick sanity check: how fast does the inference go on my 2070? Ok Then back to live MIDI if that'll work.'. Did that. coreml vs. torch on mac mini is 4x boost. cpu vs cuda on 2070 is 6x boost. Both good for realtime generation. 


### Back to realtime gen

Today's main task: format incoming MIDI messages to score format, as that is what the tokenized wants. First:

* Check if the generator code is autoregressive. If so, means I am ready to be mixing in extra events to the context as it goes. How to do that? Look for multiple calls to model forward in the generator script with a longer length of output requested. Done. It calls it for each frame. But what is the input?

* Investigated the impact of the input frame on the generation. The torch models allow for a flexible input length, up to the max 'context' of 2048+. If you feed it 2048 as an input (that is some minutes of piano playing) then it trims it to 2048 and does that. On the 2070, feeding it 2048 slows it to 2 events a second. 

### Timing

Does the model produce events in the past of previous earliest event? Check start times on events. This would be a problem! So:
  - of the tokenised events coming out of the model, do any of them appear earlier than any of the previous notes. 
  - keep a tally of onset times
  - each time a new onset time comes out of the model check if it is earlier than any of the previous onset times. 

Ok - time values coming out of the model come in as deltas on the current time point. They are converted into absolute offsets within the generated frame like this:

```
# key
#[type, bar_offset,beat_offset_in_bar, track_id, duration, channel, note, velocity]

['note', 0, 11, 0, 1020, 0, 55, 41]
Global offset 0 event offset 0 Time raw: 11 into 330
['note', 0, 11, 0, 1110, 0, 43, 40]
Global offset 0 event offset 0 Time raw: 11 into 330
['note', 0, 11, 0, 1110, 0, 59, 48]
Global offset 0 event offset 0 Time raw: 11 into 330
['note', 2, 12, 0, 900, 0, 60, 53]  ### Step forward two steps in time
Global offset 2 event offset 2 Time raw: 12 into 1320
['note', 0, 13, 0, 780, 0, 55, 43]
Global offset 2 event offset 0 Time raw: 13 into 1350
['note', 0, 13, 0, 870, 0, 45, 43]
Global offset 2 event offset 0 Time raw: 13 into 1350
---
['note', 0, 0, 0, 750, 0, 48, 45]
Global offset 88 event offset 0 Time raw: 0 into 42240
['note', 0, 0, 0, 750, 0, 64, 56]
Global offset 88 event offset 0 Time raw: 0 into 42240
['note', 0, 11, 0, 300, 0, 66, 56]
Global offset 88 event offset 0 Time raw: 11 into 42570
['note', 1, 5, 0, 300, 0, 67, 56]
Global offset 89 event offset 1 Time raw: 5 into 42870
```

Ok that's all clear - no stepping back in time, Just events happening either now or in the future come out of the model, where the time is an accumulation of event[1] values.

So I think I could tokenise output from the model

### Passing events into the model

Next question - how do we convert incoming MIDI data for the model?

The following shows what the midi tokenizer does to 'MIDI.score' formatted note data. 

```
['patch_change', 0, 0, 0] #input  
['patch_change', 0, 0, 0, 0, 0]#output
['note', 155, 465, 0, 55, 41] #input in score format ['note', start_time, duration, channel, note, velocity] 
['note', 0, 11, 0, 34, 0, 55, 41]#output ready for mode 
['note', 155, 507, 0, 43, 40] #input 
['note', 0, 11, 0, 37, 0, 43, 40]#output - for the model
['note', 155, 507, 0, 59, 48] #input 
['note', 0, 11, 0, 37, 0, 59, 48]#output
['note', 620, 352, 0, 55, 43] #input 
['note', 2, 13, 0, 26, 0, 55, 43]#output
['note', 620, 394, 0, 45, 43] #input 
['note', 2, 13, 0, 29, 0, 45, 43]#output
['note', 605, 409, 0, 60, 53] #input 
['note', 2, 12, 0, 30, 0, 60, 53]#output
['note', 972, 520, 0, 55, 38] #input 
['note', 4, 7, 0, 38, 0, 55, 38]#output
['note', 972, 563, 0, 47, 37] #input 
['note', 4, 7, 0, 41, 0, 47, 37]#output

```

```
event is: 
['note', start_time, duration, channel, note, velocity]
new_event is:
t = round(16 * event[1] / ticks_per_beat) ## i.e. adjusted start_time
[event[0], t // 16, t % 16, track_idx] + event[2:]
where

[type, bar_offset,beat_offset_in_bar, track_id, duration, channel, note, velocity]
```

Right so the event data sent into the tokenizer uses ticks for the start_time.

The absolute length of a tick in seconds depends on the BPM of the piece and the ticks per beat. 

E.g. if BPM is 120, one beat lasts 60/120 s == 0.5 seconds
if ticks per beat is 500, one tick lasts 0.5 / 500 seconds or 0.001 seconds probably

So if you have tick of 2568, just multiply that by the length of a tick to get the offset in seconds. Say you are capturing frames of MIDI from a live performer:

start time [frame] end time 

you want to convert the times in the frame into ticks. You are just receiving MIDI whenever - there is no MIDI clock here. 

So, choose a ticks per beat value at the start, make sure you send that with the data, also maybe send a set_tempo message
then for each offset in seconds in the frame, convert it to ticks using your chosen ticks per beat value and your chosen tempo. 



## 30/09/2024

### Experimenting with coreml

Tried running the onnx models with the coreml version of the onnxruntime on my mac studio, after observing that the CPU version is only slightly quicker than my AMD machine. The LLama model won't run on coreml in onnx runtime as the runtime does not support dynamic input sizes (I think). So I tried working directly with the coreml packages in Python. I was able to export the torch_lightning model to torchscript then to export the torchscript model to coreml format. I could them generate random numbers and pass them to the model by calling predict. 

Torch infers 128 times in 8.8 seconds, coreml in 1 

The inference looks really quick. This is good and means I might be able to make a mac version of a plugin in the future that uses coreml to run the model. Tempted to continue with this and to try to talk to the model from C++ but want to get live MIDI input going so more on that later. 

### MIDI messages -> tokenizer input

The MIDI.py file contains various functions to convert between representations of note data. The tokenizer receives 'score' format data as its input, then converts that for the model. 

### What is score format? 

Quoting from the MIDI.py docs:

``The "opus" is a direct translation of the midi-file-events, where
the times are delta-times, in ticks, since the previous event.

The "score" is more human-centric; it uses absolute times, and
combines the separate note_on and note_off events into one "note"
event, with a duration."

['note', start_time, duration,channel, note, velocity] # in a "score"

E.g. 

```
my_score = [
    96,
    [   # track 0:
        ['patch_change', 0, 1, 8],
        ['note', 5, 96, 1, 25, 96],
        ['note', 101, 96, 1, 29, 96]
    ],   # end of track 0
]
```



Score data can be parsed like this:
```
channels = {2,3,5,8,13}
itrack = 1   # skip 1st element which is ticks
while itrack < len(score):
    for event in score[itrack]:
        if event[0] == 'note':   # for example,
            pass  # do something to all notes
        # or, to work on events in only particular channels...
        channel_index = MIDI.Event2channelindex.get(event[0], False)
        if channel_index and (event[channel_index] in channels):
            pass  # do something to channels 2,3,5,8 and 13
    itrack += 1
```


### Events in the score format

There are lots of events aside from 'note' in score format. Perhaps most interesting for now: 

patch_change, 
['patch_change', dtime, channel, patch]

pitch_wheel_change, 
['pitch_wheel_change', dtime, channel, pitch_wheel]

set_tempo, 
['set_tempo', dtime, tempo]

key_signature:
['key_signature', dtime, sf, mi]

## 27/09/2024

### Back to realtime MIDI input

After wasting a few hours on ROCM yesteday, I am cracking on with the realtime MIDI input -> continuator implementation. 

I looked into the MIDI tokenizer input before. So I am going to start in Python then work out how to tokenize some MIDI grabbed from a MIDI device into the correct format, which is whatever you get from the MIDI.py library's score format. I think. 

I managed to get a minimal version of the python gen script that uses onnx models instead of torch. It seems to run a bit quicker than the torch models too.

### What actually is the control flow here?

Assuming the improviser context, there are various interaction structures you could use

* human solos, AI plays chords
 -> input to the model is human solo, output is filtered to just chords (i.e. events occuring close to eachother and infrequently or something)
* AI solos, human plays chords
 -> input is human chords plus AI solo, output is filtered to just standalone notes
* Both play together
 -> input is human playing + AI playing, output is not filtered
* trading fours
 -> human plays something, AI plays something 

Need a ring buffer for the short term memory of what is going on
-> it should contain embeddings for token frames??

UI:
  Toggle AI listens to you -> ring buffer input sequence
  Toggle AI listens to itself -> put AI output into the ring buffer	
  Play last 
  Output filter controls
  

## 26/09/2024

### Making SkyTNT work in realtime

- trimmed the gradio app to a minimal setup that can do a continuation on a midi file
- now want to receive MIDI in realtime
- the MIDI.py thing works with raw binary MIDI file data! need to work around that somehow

Need to convert the incoming MIDI into the appropriate encoding for the tokenizer.
Reverse engineering the skytnt code, that appears to be the 'MIDI.score' format.
This looks as follows:

s[0] = 220 # the 'ticks' parameter
s[1..n] # the tracks of data, where each track is a list of events
# e.g.: s[1] (track 1) might be a load of special time config messages  
s[1] = [['set_tempo', 0, 500000],
 ['time_signature', 0, 4, 2, 24, 8],
 ['text_event', 1, '']]
# then s[2] (track 2) might be the first track of actual midi note events in the file
s[2][1:5] =
[['patch_change', 0, 0, 0],
 ['note', 183, 99, 0, 50, 67],
 ['note', 436, 43, 0, 35, 67],
 ['note', 675, 71, 0, 48, 69],
 ['note', 928, 71, 0, 47, 68]]
 
So I need to format my incoming midi into that structure. Ideally, I can look at the output used in auto-regressive mode to see how that is. 

### ROCm roll (ugh)

Did some mucking about with ROCm, since I now have all that stuff installed on my laptop. Was hoping to see things running faster. 

I now have a cuda device registering and visible from pytorch, after installing a custom build of pytorch from here:

https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/

rocminfo says that ''ROCk module version 6.8.5 is loaded'. I had to add myself to the video group and reboot. 

I tried running the basic app.py for skytnt using cuda as the device but no dice there. I crash out with a RuntimeError: HIP error: invalid device function. Looks kinda insurmountable, implies my graphics card can't do whatever that is. Maybe a different driver? who knows. Need to move on. 

Then tried running the onnx runtime version of the app to see if onnx can talk better to my ROCM stuff than pytorch. There is a line in app_onnx.py where the runtime is set:

```
providers = ['MIGraphXExecutionProvider', 'ROCMExecutionProvider']

# and since:
import onnxruntime as ort
ort.get_device()
# prints
 'GPU-MIGRAPHX'
```
I went for MIGraphXExecutionProvider. It takes a long time to load the mode, and then crashes out on some sort of compile step. Some value is set to zero:
```
2024-09-27 12:05:06.450081551 [E:onnxruntime:, sequential_executor.cc:514 ExecuteKernel] Non-zero status code returned while running MGXKernel_graph_torch-jit-export_4651530453327880677_0 node. Name:'MIGraphXExecutionProvider_MGXKernel_graph_torch-jit-export_4651530453327880677_0_0' Status Message: Failed to call function
```

yeah some missing function again. Maybe that's my cheapo ROCM! 

Giving up and moving on. ps I also tried 


### Next steps:

- need to operationalise the skytnt model in a realtime context
- has good potential because the repo comes with onnx exports and an onnx exporter script. 
- then can try some different training styles as per tegrity/ conditioning

- possibly also operationalise the tegrity models ... is a simple export possible here?
- then train some monophonic models with chord conditioning if at all possible
- then onto timbre of course.


### Some notes about different transformer architectures

Looked deeper into the SkyTNT transformer model, following the info about the LA dataset and tegridy. Found this repo:

https://github.com/asigalov61

@inproceedings{lev2024tegridytools,
    title       = {tegridy-tools: Symbolic Music NLP Artificial Intelligence Toolkit},
    author      = {Aleksandr Lev},
    booktitle   = {GitHub},
    year        = {2024},
}


Lev's repo' contains all kinds of pre-trained music transformers. It is all based around the x-transformers model from lucidrains, but modified slightly. The repos for the various models contain this:

https://github.com/asigalov61/Ultimate-Accompaniment-Transformer/blob/main/x_transformer_1_23_2.py

Then it is created something like this:

```
model = TransformerWrapper(
    num_tokens = PAD_IDX+1,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 2048, depth = 4, heads = 16, attn_flash = True)
    )

model = AutoregressiveWrapper(model, ignore_index = PAD_IDX, pad_value=PAD_IDX)
```

So it is essentially a basic transformer model plus flash attention, but without positional encoding. Compared to the SkyTNT model, that uses the class 'LlamaModel' from huggingface transformers:

https://github.com/huggingface/transformers/tree/main/docs/source/en/model_doc
(search for llama, llama2, llama3)

Apparently, LlamaModel is ''The bare LLaMA Model outputting raw hidden-states without any specific head on top'. Huggingface use that model to wrap all llama versions (1,2,3). You get a different version of llama depending on which parameters you send in when you create it. SkyTNT creates it like this:

```
self.net = LlamaModel(LlamaConfig(vocab_size=tokenizer.vocab_size,
                                          hidden_size=n_embd, num_attention_heads=n_head,
                                          num_hidden_layers=n_layer, intermediate_size=n_inner,
                                          pad_token_id=tokenizer.pad_id, max_position_embeddings=4096))
```

This is not llama2, as that adds 'Grouped Query Attention' with the 'num_key_value_heads (`int`, *optional*)' param which SkyTNT does not use. For info, the HF info about llama3 says: 'The architecture is exactly the same as Llama2.'. 

The other model I have been looking at and training is the multitrack music transformer:

```bibtex
@inproceedings{dong2023mmt,
    author = {Hao-Wen Dong and Ke Chen and Shlomo Dubnov and Julian McAuley and Taylor Berg-Kirkpatrick},
    title = {Multitrack Music Transformer},
    booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    year = 2023,
}
```

The transformer model in this one is:

```
       transformer_kwargs = {
            "max_seq_len": kwargs.pop("max_seq_len"),
            "max_beat": kwargs.pop("max_beat"),
            "emb_dropout": kwargs.pop("emb_dropout", 0),
            "use_abs_pos_emb": kwargs.pop("use_abs_pos_emb", True),
        }
        self.decoder = MusicTransformerWrapper(
            encoding=encoding,
            attn_layers=Decoder(dim=dim, **kwargs),
            **transformer_kwargs,
        )
        self.decoder = MusicAutoregressiveWrapper(
            self.decoder, encoding=encoding
        )
```
The important bits are the 'attn_layers' which is a Decoder from lucidrains x_transformers library:

```
from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Decoder,
    TokenEmbedding,
    always,
    default,
    exists,
)
```

These are the args:

```
dim=args.dim, (512 default)
encoding=encoding,
depth=args.layers, (6)
heads=args.heads,  (8)
max_seq_len=args.max_seq_len (1024),
max_beat=args.max_beat, (256)
rotary_pos_emb=args.rel_pos_emb, (yes/no)
use_abs_pos_emb=args.abs_pos_emb, (yes/no)
emb_dropout=args.dropout,
attn_dropout=args.dropout,
ff_dropout=args.dropout,
```

So not sure if they are using flash attention there, but the paper seems to focus on the idea of position embeddings which the other systems do not use, as far as I can tell, though need to check the llama implementation from HF to confirm.

### Tokenisers/ tokenizers

Another point where the SkyTNT model vs. Lev tegridy models diverge is probably the tokenizer. That is something I will need to understand in order to make a realtime version, since I'll need to encode a MIDI stream into the correct format for the model.

## 25/09/2024

### Next steps:

- RUNNING train from scratch with the LA formatted, transposed data 
- RUNNING try a fine tune from trained model using LA formatted data. I /think/ tranposing is not needed for that. 

### Progress today:

- got SkyTNT llama music model to train from scratch on the jazz piano dataset
- after 7000 steps the first few seconds of output are quite good! 
- after 25,000 steps, sounding quite good too
- also figured out how to transpose all the pieces to c min/maj which will give it more tonal information from the limited dataset
- further, worked out how to convert the dataset into 'LA' format, which is supposedly better for training


### Sort out the dataset for skytnt model training:

- next step there is to transpose all the pieces to the same key and train on that
- and to sort out the timing so it is not just cecil taylor all the way

-> timing
Check out how the timing processing works on the la dataset
Convert my data into LA format? 


### Sound font plugin

Waste loads of time trying to build this soundfont vst: https://github.com/Birch-san/juicysfplugin
Then return to the original one here: 

This fork worked with a bit of editing on the call from processBlock to fluidsynth to map the buffer into it:

https://github.com/studiorack/juicysf

Also got my original back. All good now. 

### Training stuff:

- Get the SkyTNT model to train on igor
- Get those soundfonts working in reaper on my new OS install
- Make the realtime inference system
- Investigate training on jazz piano dataset
  - is the loss function appropriate for mono
  - format for MMT
  - format for SkyTNT
- investigate training with weimar dataset

- what about the realtime inference/orchestration idea
   * ok this is a goer with the models I have 
   -> just need to implement realtime and autoregression data flow
   * best model for this is to play all the parts except the one they are playing (a la reflexive looper)

- what about the 'filling in the gap' model? 
  * v1: reflexive looper model: play all the parts except the one 

## 24/09/2024

(rewriting as I lost my notes by not saving)

Today I was experimenting with training on the GS GPU server. I managed to get MMT to train on the SOD data set using the RPE and APE encoding models. Had to do a few edits on scripts to make it work on Python3.8 which is what is on that machine:

- changed file extensions in one of the data prep scripts to '.csv' '.npy' instead of 'csv'
- chnaged the argparser setup to deal with missing BooleanValue thing in 3.8 
- I think that was about it

I left a training run going for 10 hours to see what the result is. I set the save interval at 10,000 so I wouldn't get too many saved models. 

Next steps here:
- noting that they already provide pre-trained models for all their datasets and settings, not much point in me doing any more training runs on their data/ settings. But I should compare my trained model to their trained model with the SOD/ RPE setup to verify training is working as expected....
- Download the best model, check loss, compare the model to their version of that model
* Possibly see about training on that LA dataset that SkyTNT is trained on - get the data at least
- What about training on same key, single instrument data?

## 17/09/2024

### Progress at the end of the day

I was able to run all the scripts except speed_test for the MMT transformer model, at least on its
SOD dataset. And on my local machine. 

I had to apply some code fixes to libraries, even muspy to make it work
I also had to fandangle the python setup. Ultimately, a specific version of
pytorch (not 2) and a specific version of x-transformers made it work
It seems x-transformers is not compatible with some modules in pytorch 2.

It generates MIDI of a decent quality in response to other MIDI

I also obtained a load of different sound fonts and a fluidsynth VST wrapper.

So now I can easily play MMT-generated MIDI files in realtime in a decent environment.

MMT also has an evaluation script that applies metrics around rhythmic consistency 
and note pitch correctness to outputs. 

Next steps:
* I have two models that I can infer with and train: MMT and music-lm
* MMT is some sort of x-transformers thing, music-lm is a llama model (using huggingface)

### Going back to Multitrack  Music Transformer (2023)

https://github.com/salu133445/mmt

Based on the x-transformers package which implements all kinds of funky transformer stuff that might be useful here.

https://github.com/lucidrains/x-transformers

Next steps: 
* I have downloaded their pre-trained models. They are small! midi-lm is nearly 1gig. These are 79 mb.
* It can train, yes. 
* Does it infer? Trying that now
* Can you fine tune it? Hopefully on my laptop as the models are small.

### SkyTNT

I did a finetuning run from SkyTNT's checkpoint with the jazz piano dataset.

Next steps:

* ** Try to operationalise the code in a realtime improv context: MIDI in, MIDI out**
* Consider compositional ideas: piano in, full backing track out etc. 
* Possibly check out the fintunes: https://huggingface.co/skytnt/midi-model-ft/blob/main/README.md
* Investigate the MIDI data format, e.g. can we transpose everything to one key etc.
  Apparently it is trained on this: https://huggingface.co/datasets/projectlosangeles/Los-Angeles-MIDI-Dataset#make-your-own-los-angeles-midi-dataset-from-any-midi-scrape
* Investigate cheaper GPU options than colab 

### 

### Music-LM

I found this 2023/ 2024 repository which implements a MIDI to MIDI model:

https://github.com/jeremyjordan/midi-lm

It provides a training script but I can't see any pre-trained models.

So this might be one to compare to skytnt, if I can find a pre-trained model or train one myself.	

## 16/09/2024

### SkyTNT's MIDI transforner

This is the real shizzle. Works very well.

LLama model LlamaModel

https://github.com/SkyTNT/midi-model?tab=readme-ov-file
https://huggingface.co/skytnt/midi-model

* Active development in 2024
* Windows app
* Python training scripts

Next steps:
 * Run it [DONE]
 * Train it [DOING IT ... can do on colab with A100 40gb GPU + batch size 6]
   -> https://colab.research.google.com/drive/15rxjokSWakwPUUOFdmQDnsmJSpH99m-7#scrollTo=hc1Nbkz-wJVP
 * Try to fine tune the pre-trained on the jazz MIDI
 * Try to train it with everything in the same key, on just jazz MIDI

### Musicautobot

This is a few years old, so maybe not so hot:
https://github.com/bearpelican/musicautobot

but might be interesting and efficient. 


### Trained music transformers/Tristan Behrens 

Did a bit of searching for trained music transformers and found this, trained on lakh dataset:

https://www.youtube.com/watch?v=LtCMQ6R5bsk
https://huggingface.co/ai-guru/lakhclean_mmmtrack_4bars_d-2048

He has a load of other models that he has not released. Some of them do in-painting. 

But - his notebook looks good as a demo of using his gpt2/lak generator

He has another repo here which shows a basic version of music transformer
along with training data:

https://github.com/AI-Guru/musictransformer2023

So next step is to:

* Try out his colab 
* if good, try to run it locally
* have a read of his musictranformer2023 code
## 13/09/2024

### Musiclang

Experimented with musiclang: https://github.com/musiclang/musiclang_predict?tab=readme-ov-files

It can do a continuation task - so to predict the next n tokens given a midi file as input

Experimented minimally with it and made the following obseverations:

* Technically, under the hood they have a hacked llamacpp c library that python talks to
* They have a pre-trained model trained on lakh
* It aims to generate multi-instrument files by default 
* It can be conditioned with a chord sequenbce
* Might be realtime capable - can make 23 second audio file in way less than 23 seconds
* Tried feeding it some chick corea and yeah it kinda works
* I think the big thing with this model is that it is conditioned on chords:
  - A bit like the MMT-Bert model: https://arxiv.org/abs/2409.00919
  - that is big as can lead to a more satisfying compositional structure
    if I compose a chord structure for the performance.
* Can I fine tune it somehow? Probably not without their original training code
* Can I use it in realtime, yeah probably. 
* Does it sound good: meh but let's try it

### Treating the improvising problem as a masking problem

The masking task involves a language model guessing how to fill in a blank. 

There are lots of examples on huggingface showing how to train models for masking tasks: 

https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling

This paper used it to create good-sounding fill-ins for piano material:

https://aimc2024.pubpub.org/pub/2e7q3cpr/release/1

They claim this:
- they outperform music transformer for phrase level synthesis with small training data
- new tokenisation method REMI Lite which improves rhythmic structure (my encoding might do that too)
- they segment the midi files into phrases by rendering to audio then using an mfcc similarity metric plus clustering to identify phrase boundaries.

I have contacted the authors, one of whom is Michael Casey so hopefully can get some code there.

### Going back to basics: x-transformers

Lots of info on this page about various attempts to improve sequence modelling with transformers by adding positional information in different ways:

https://github.com/lucidrains/x-transformers

Could consider training a little one on some sort of text-like music format.

### Another model with code is Multitrack  Music Transformer (2023)

https://github.com/salu133445/mmt

Based on the x-transformers package which implements all kinds of funky transformer stuff that might be useful here.

https://github.com/lucidrains/x-transformers

Next steps: 
* Does it work? 
* Can you fine tune it?

### DDSP-MIDI

This is a potentially interesting system https://github.com/magenta/midi-ddsp

It attempts to address the problem of playing DDSP instruments from MIDI
wherein you do not have continuous pitch and amplitude input (as you do in timbre transfer)
So it synthesizes expression contours per note (i.e. continuous +/- values for pitch and amplitude)
I mucked about with this in the DDSP work I did. 


### Evaluation metrics 

https://aimc2024.pubpub.org/pub/m2ub7gup/release/1

