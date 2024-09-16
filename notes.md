# Some notes

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

### DDSP-MIDI

This is a potentially interesting system https://github.com/magenta/midi-ddsp

It attempts to address the problem of playing DDSP instruments from MIDI
wherein you do not have continuous pitch and amplitude input (as you do in timbre transfer)
So it synthesizes expression contours per note (i.e. continuous +/- values for pitch and amplitude)
I mucked about with this in the DDSP work I did. 


### Evaluation metrics 

https://aimc2024.pubpub.org/pub/m2ub7gup/release/1

