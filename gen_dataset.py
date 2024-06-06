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
import argparse

from tacotron2 import layers
import numpy as np
from pydub import AudioSegment, effects  
from pydub.silence import split_on_silence
from scipy.io.wavfile import read as read_wav, write as write_wav
from tacotron2 import hparams as t_hparams

hparams = None
stft = None


def init_globals():
    global hparams, stft

    hparams = t_hparams.create_hparams()
    hparams.sampling_rate = 44010
    #hparams.sampling_rate = 48000
    stft = layers.TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                               hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                               hparams.mel_fmax)

def preprocess_audio(wav_file, silence_audio_size):
    sr = hparams.sampling_rate
    max_wav_value = 32768.0
    trim_fft_size = 1024
    trim_hop_size = 256
    trim_top_db   = 23

    
    data, sampling_rate = librosa.core.load(wav_file, sr)
    silence_data = [0.] * silence_audio_size
    data_tmp = data / np.abs(data).max() * 0.999
    data_tmp = np.append(silence_data, data_tmp)
    data_tmp = librosa.effects.trim(data_tmp, top_db=trim_top_db, frame_length=trim_fft_size, hop_length=trim_hop_size)[0]
    data_tmp = data_tmp * max_wav_value
    data_tmp = np.append(data_tmp, silence_data)
    data_tmp = data_tmp.astype(dtype=np.int16)
    write_wav(wav_file, sr, data_tmp)
    #print(len(data),len(data_))
    #if(i%100 == 0):
    #    print (i)


#def load_wav_to_torch(full_path):
#    sampling_rate, data = read_wav(full_path)
#    return torch.FloatTensor(data.astype(np.float32)), sampling_rate
#
#def wav2mel(input_filename, output_filename):
#    #print("Generating mel")
#
#    audio, sampling_rate = load_wav_to_torch(input_filename)
#    if sampling_rate != stft.sampling_rate:
#        raise ValueError("{} {} SR does not match the objective {} SR".format(filename,
#            sampling_rate, stft.sampling_rate))
#    audio_norm = audio / hparams.max_wav_value
#    audio_norm = audio_norm.unsqueeze(0)
#    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
#    melspec = stft.mel_spectrogram(audio_norm)
#    melspec = torch.squeeze(melspec, 0).cpu().numpy()
#    np.save(output_filename, melspec)


def range_srt2millis(audio_txt):
    time_arr = audio_txt.split(":")
    sec_mil_arr = time_arr[-1].split(",")
    millis = int(sec_mil_arr[1])
    secs = int(sec_mil_arr[0])
    mins = int(time_arr[-2])
    hours = int(time_arr[-3])

    return millis + secs * 1000 + mins * 60000 + hours * 3600000


def extract_text(text, speaker_str):
    speaker_idx = text.index(":")
    speaker_tgt = int(text[:speaker_idx].split(" ")[1])

    if int(speaker_str) != speaker_tgt:
        return None

    return text[speaker_idx+2:]


def change_extension(filename, new_ext):
    ext_idx = filename[::-1].find(".")
    if ext_idx < 0:
        result_filename = filename + "." + new_ext
    else:
        result_filename = filename[:len(filename) - ext_idx] + new_ext
    return result_filename


def convert_num_to_words(utterance, lang="es"):
    utterance = ' '.join([num2words.num2words(i, lang=lang) if i.isdigit() else i for i in utterance.split()])
    return utterance

def extract_nums_from_text(input_text, lang):
    return convert_num_to_words(re.sub('(\d+(\.\d+)?)', r' \1 ', input_text), lang=lang).strip()

def extract_vocals(input_audio_file, output_audio_file):

    separator = demucs.api.Separator(model="mdx_extra", segment=12, progress=True)
    orig, separated = separator.separate_audio_file(input_audio_file)
    demucs.api.save_audio(separated["vocals"], output_audio_file, samplerate=separator.samplerate)

def audio_srt_to_chunks(speaker, audio_filename_in, lang, outdir, noise_remove=False):
    audio_subdir = "audio"
    #dataset_path = os.path.join(outdir, dataset_name)
    dataset_path = outdir
    audio_out_dir = os.path.join(dataset_path, audio_subdir)
    dataset_filename_wav = os.path.join(dataset_path, "dataset_wav.txt")
    #dataset_filename_npy = os.path.join(dataset_path, "dataset_npy.txt")
    index_filename       = os.path.join(dataset_path, "index.txt")
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(audio_out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file_name = ''.join(random.choices(string.ascii_lowercase, k=4)) + ".wav"

        if noise_remove:
            audio_filename = os.path.join(tmp_dir, tmp_file_name)
            print("Isolating vocal part using demucs...")
            extract_vocals(audio_filename_in, audio_filename)
        else:
            audio_filename = audio_filename_in

        audio_obj = AudioSegment.from_file(audio_filename).set_frame_rate(hparams.sampling_rate).set_channels(1)
    
        srt_filename = change_extension(audio_filename_in, "srt")

        with open(srt_filename, "r") as f:
            srt_lines = f.readlines()

        out_lines_wav = list()
        #out_lines_npy = list()
        out_wav = list()

        if os.path.exists(index_filename):
            with open(index_filename, "r") as f:
                sequence = int(f.read())
        else:
            sequence = 0

        print(f"Begin index: {sequence}")
        total_lines = len(srt_lines)
        

        for i in range(0, total_lines, 4):
            idx            = srt_lines[i]
            time_range_srt = srt_lines[i + 1]
            text_raw_srt   = srt_lines[i + 2]
            blank_srt      = srt_lines[i + 3]

            #print(f"I: {idx} TR: {time_range_srt} T: {text_raw_srt}")

            range_srt  = time_range_srt.split(" --> ")
            start_time = range_srt2millis(range_srt[0])
            end_time   = range_srt2millis(range_srt[1])

            text = extract_text(text_raw_srt, speaker)
            if text:
                text = text.strip()

            #print(f"Text: {text}")
            if not text:
                continue

            #print(f"ST: {start_time} - {end_time}: {text}")
            #audio_chunk = audio_obj[start_time - 100 : end_time + 100]
            audio_chunk = audio_obj[start_time - 50 : end_time + 150]
            audio_chunk = effects.normalize(audio_chunk) 


            audio_silence = AudioSegment.silent(duration=500)
            audio_chunk = audio_silence + audio_chunk + audio_silence

            audio_chunk_filename_wav = f"chunk_{sequence:05}.wav"
            #audio_chunk_filename_npy = f"chunk_{sequence:05}.npy"
            audio_chunk_full_path_wav = os.path.join(audio_out_dir, audio_chunk_filename_wav)
            #audio_chunk_full_path_npy = os.path.join(audio_out_dir, audio_chunk_filename_npy)
            audio_chunk_rel_path_wav = os.path.join(audio_subdir, audio_chunk_filename_wav)
            #audio_chunk_rel_path_npy = os.path.join(audio_subdir, audio_chunk_filename_npy)
            audio_chunk.export(audio_chunk_full_path_wav, format="wav")

            #print(f"Writing: {audio_chunk_rel_path_wav} {audio_chunk_rel_path_npy}")
            #with tempfile.TemporaryDirectory() as tmp_dir:
            #    print(f"Writing: {audio_chunk_rel_path_wav}")
                
            #    tmp_filemame_wav = os.path.join(tmp_dir, audio_chunk_filename_wav)
            #    audio_chunk.export(tmp_filemame_wav, format="wav")
                
            #    print(f"Processing: {audio_chunk_rel_path_wav}")
            #    preprocess_audio(tmp_filemame_wav, 50000)
            #    shutil.copy(tmp_filemame_wav, audio_chunk_full_path_wav)

            #wav2mel(audio_chunk_full_path_wav, audio_chunk_full_path_npy)
    
            text_wo_nums = extract_nums_from_text(text, lang)
            print(f"Text w/o nums: {text_wo_nums}")
            out_lines_wav.append(f"{audio_chunk_rel_path_wav}|{text}|{text_wo_nums}\n")
            out_wav.append(audio_chunk_rel_path_wav)
            #out_lines_npy.append(f"{audio_chunk_rel_path_npy}|{text}")
            sequence += 1

    

    with open(dataset_filename_wav, "a") as f:
        f.writelines(out_lines_wav)

    #with open(dataset_filename_npy, "a") as f:
    #    f.writelines(out_lines_npy)

    with open(index_filename, "w") as f:
        f.write(str(sequence))

    print(f"Total {len(out_lines_wav)} chunks, next index: {sequence}")

def get_parser():
    parser = argparse.ArgumentParser("gen_dataset",
                                     description="Generate TTS datasets from audios and srt")
    parser.add_argument("-i",
                        "--audio_path",
                        help="Input audio file")
    parser.add_argument("-n",
                        "--speaker_num",
                        type=int,
                        help="Speaker number selected")
    parser.add_argument("-l",
                        "--lang",
                        default="es",
                        help="Language selected")
    parser.add_argument("-o",
                        "--outdir",
                        default="result",
                        help="Dataset output directory")
    parser.add_argument("-r",
                        "--noise_remove",
                        type=bool,
                        default=False,
                        help="Isolate vocal part of audio using demucs neural filter")

    return parser

def main(opts=None):
    parser = get_parser()
    args = parser.parse_args(opts)
    audio_srt_to_chunks(args.speaker_num, args.audio_path, args.lang, args.outdir, noise_remove=args.noise_remove)    



#if len(sys.argv) < 3:
#    print(f"Usage: {sys.argv[0]} num_speaker audio_path [lang] [outdir]")
#    sys.exit(1)

init_globals()
main()

#lang   = sys.argv[3] if len(sys.argv)>=4 else "es"
#outdir = sys.argv[4] if len(sys.argv)>=5 else "result"

#audio_srt_to_chunks(sys.argv[1], sys.argv[2], lang, outdir)


