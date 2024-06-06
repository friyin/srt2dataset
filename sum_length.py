#!/usr/bin/env python

import os
import sys
import re
#import torch
import librosa
import num2words
import tempfile
import shutil
import demucs.api
import string
import random

from pydub import AudioSegment, effects  
from pydub.silence import split_on_silence



def length2str(length_i):
    millis  = length_i % 1000
    secs = int(length_i/1000) % 60
    mins = int(length_i/60000)

    if mins:
        result = f"{mins}:{secs:02}.{millis:03}"
    else:
        result = f"{secs}.{millis:03}"

    return result

def main():
    if len(sys.argv) < 3:
        print(f"Usage {sys.argv[0]} metadata_file.csv base_dir")
        sys.exit(1)

    basedir=sys.argv[2]
    metadata_filename=sys.argv[1]

    with open(metadata_filename, "r") as f:

        lines = f.readlines()
        total_length = 0
        for line in lines:
            audio_filename = line.split("|")[0]
            try:
                audio_obj = AudioSegment.from_file(os.path.join(basedir, audio_filename))
                length = len(audio_obj)
                total_length += length
                print(f"Audio: {audio_filename} length: {length2str(length)}")
            except FileNotFoundError:
                continue

        str_total_length = length2str(total_length)
        print(f"Total length: {str_total_length}")
        

main()
