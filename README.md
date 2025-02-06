# bcn-2024-system
AI music improviser system used in Barcelona 2024 event

This code is adapted from the SkyTNT llama-based music model

https://github.com/SkyTNT/midi-model

Changes made by MYK:

* Removed Flash=True as this stopped it from working due to some rot on the python packages
 (https://github.com/SkyTNT/midi-model/issues/14)
* make it train on my mac
* built an agent wrapper around the model and associated MIDI processing stuff to allow it to run as a realtime improviser
* fine tuned it on the QMUL pijama dataset


