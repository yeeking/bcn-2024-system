### This script takes as an input
### a folder tree full of midi files
### and converts it to the 'LA format'
### 

import os

import math
import statistics
import random
from collections import Counter
import shutil
import hashlib
from tqdm import tqdm
import sys
import TMIDIX

def get_dataset_files(in_folder):
    # os.chdir(dataset_addr)
    files = list()
    for (dirpath, dirnames, filenames) in os.walk(in_folder):
        files += [os.path.join(dirpath, file) for file in filenames]
    # print('=' * 70)
    return files

def process_midi_file(filename, out_folder):
    """
    Checks if the MIDI in the file is ok, and if so, copies it anonymously to out_folder
     --> not excessive length
     --> minimum number of notes (256) 
     --> non-negative onset times and durations
    """
    # input_files_count += 1
    fn = os.path.basename(f)
    # Filtering out giant MIDIs
    file_size = os.path.getsize(f)
    if file_size >= 1000000: # ignore monster files
        return   
    fdata = open(filename, 'rb').read()
    md5sum = hashlib.md5(fdata).hexdigest()
    md5name = str(md5sum) + '.mid'
    score = TMIDIX.midi2score(fdata)
    events_matrix = []
    itrack = 1
    while itrack < len(score):
        for event in score[itrack]:
            events_matrix.append(event)
        itrack += 1

    events_matrix.sort(key=lambda x: x[1])
    notes = [y for y in events_matrix if y[0] == 'note']
    if len(notes) >= 256: # want at least 256 notes played
        times = [n[1] for n in notes]
        durs = [n[2] for n in notes]
        if min(times) >= 0 and min(durs) >= 0: # sane onset times and durations 
            if len(times) > len(set(times)): # not sure?? set(times) is unique items in times, times, is everything
                shutil.copy2(filename, out_folder+str(md5name[0])+'/'+md5name) # ok anonymise it 

if __name__ == '__main__':
    assert len(sys.argv) == 3, "Usage python script.py MIDI_in_folder LAMIDI_out_folder"
    ### verify this folder exists 
    # !git clone https://github.com/asigalov61/tegridy-tool

    # assert os.path.exists('tegridy-tool'), f"You need to git clone https://github.com/asigalov61/tegridy-tool"

    in_folder = sys.argv[1]
    assert os.path.exists(in_folder), f"In folder does not exist {in_folder}"

    out_folder = sys.argv[2]
    assert not os.path.exists(out_folder), f"Out folder already exists {out_folder} - need a non-existent one"

    os.makedirs(out_folder)
    output_dirs_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    for o in output_dirs_list:
        p = out_folder + "/" + str(o)+'/'
        os.makedirs(p)
    
    files = get_dataset_files(in_folder)
    random.shuffle(files)
    TMIDIX.Tegridy_Any_Pickle_File_Writer(files, out_folder + 'files')
    for f in tqdm(files):
        process_midi_file(f, out_folder)


