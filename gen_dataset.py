#!/usr/bin/env python

import os
import sys
import torch
from tacotron2 import layers
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.io.wavfile import read as read_wav
from tacotron2 import hparams as t_hparams

hparams = None
stft = None


def init_globals():
    global hparams, stft

    hparams = t_hparams.create_hparams()
    stft = layers.TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                               hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                               hparams.mel_fmax)

def load_wav_to_torch(full_path):
    sampling_rate, data = read_wav(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def wav2mel(input_filename, output_filename):
    print("Generating mel")

    audio, sampling_rate = load_wav_to_torch(input_filename)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} {} SR does not match the objective {} SR".format(filename,
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0).cpu().numpy()
    np.save(output_filename, melspec)


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


def audio_srt_to_chunks(speaker, audio_filename, outdir):
    audio_subdir = "audio"
    #dataset_path = os.path.join(outdir, dataset_name)
    dataset_path = outdir
    audio_out_dir = os.path.join(dataset_path, audio_subdir)
    dataset_filename_wav = os.path.join(dataset_path, "dataset_wav.txt")
    dataset_filename_npy = os.path.join(dataset_path, "dataset_npy.txt")
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(audio_out_dir, exist_ok=True)
    audio_obj = AudioSegment.from_file(audio_filename).set_frame_rate(hparams.sampling_rate).set_channels(1)
   
    srt_filename = change_extension(audio_filename, "srt")

    with open(srt_filename, "r") as f:
        srt_lines = f.readlines()

    out_lines_wav = list()
    out_lines_npy = list()

    sequence = 0
    total_lines = len(srt_lines)
    for i in range(0, total_lines, 4):
       idx            = srt_lines[i]
       time_range_srt = srt_lines[i + 1]
       text_raw_srt   = srt_lines[i + 2]
       blank_srt      = srt_lines[i + 3]

       print(f"I: {idx} TR: {time_range_srt} T: {text_raw_srt}")

       range_srt = time_range_srt.split(" --> ")
       start_time = range_srt2millis(range_srt[0])
       end_time = range_srt2millis(range_srt[1])

       text = extract_text(text_raw_srt, speaker)
       #print(f"Text: {text}")
       if not text:
           continue

       #print(f"ST: {start_time} - {end_time}: {text}")
       audio_chunk = audio_obj[start_time:end_time]
       audio_chunk_filename_wav = f"chunk_{sequence:05}.wav"
       audio_chunk_filename_npy = f"chunk_{sequence:05}.npy"
       audio_chunk_full_path_wav = os.path.join(audio_out_dir, audio_chunk_filename_wav)
       audio_chunk_full_path_npy = os.path.join(audio_out_dir, audio_chunk_filename_npy)
       audio_chunk_rel_path_wav = os.path.join(audio_subdir, audio_chunk_filename_wav)
       audio_chunk_rel_path_npy = os.path.join(audio_subdir, audio_chunk_filename_npy)
       audio_chunk.export(audio_chunk_full_path_wav, format="wav")
       wav2mel(audio_chunk_full_path_wav, audio_chunk_full_path_npy)

       out_lines_wav.append(f"{audio_chunk_rel_path_wav}| {text}")
       out_lines_npy.append(f"{audio_chunk_rel_path_npy}| {text}")
       sequence += 1

    with open(dataset_filename_wav, "w") as f:
        f.writelines(out_lines_wav)

    with open(dataset_filename_npy, "w") as f:
        f.writelines(out_lines_npy)


if len(sys.argv)!=4:
    print(f"Usage: {sys.argv[0]} num_speaker audio_path outdir")
    sys.exit(1)

init_globals()

audio_srt_to_chunks(sys.argv[1], sys.argv[2], sys.argv[3])


