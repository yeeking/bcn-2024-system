# Notes on building Barcelona system

## 25/09/2024

Progress today:

- got SkyTNT llama music model to train from scratch on the jazz piano dataset
- after 7000 steps the first few seconds of output are quite good! 
- next step there is to transpose all the pieces to the same key and train on that
- also to op


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

