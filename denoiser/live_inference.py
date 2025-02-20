# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import argparse
import sys

import torch
import torchaudio
import os


import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

from demucs import DemucsStreamer
from pretrained import add_model_flags, get_model
from utils import bold


def get_parser():
    parser = argparse.ArgumentParser(
        "denoiser.live",
        description="Performs live speech enhancement, reading audio from "
                    "the default mic (or interface specified by --in) and "
                    "writing the enhanced version to 'Soundflower (2ch)' "
                    "(or the interface specified by --out)."
        )
    parser.add_argument(
        "-i", "--in", dest="in_",
        help="name or index of input interface.")
    parser.add_argument(
        "-o", "--out", default="Soundflower (2ch)",
        help="name or index of output interface.")
    add_model_flags(parser)
    parser.add_argument(
        "--no_compressor", action="store_false", dest="compressor",
        help="Deactivate compressor on output, might lead to clipping.")
    parser.add_argument(
        "--device", default="cpu")
    parser.add_argument(
        "--dry", type=float, default=0.04,
        help="Dry/wet knob, between 0 and 1. 0=maximum noise removal "
             "but it might cause distortions. Default is 0.04")
    parser.add_argument(
        "-t", "--num_threads", type=int,
        help="Number of threads. If you have DDR3 RAM, setting -t 1 can "
             "improve performance.")
    parser.add_argument(
        "-f", "--num_frames", type=int, default=1,
        help="Number of frames to process at once. Larger values increase "
             "the overall lag, but will improve speed.")
    return parser


def parse_audio_device(device):
    if device is None:
        return device
    try:
        return int(device)
    except ValueError:
        return device


def query_devices(device, kind):
    try:
        caps = sd.query_devices(device, kind=kind)
    except ValueError:
        message = bold(f"Invalid {kind} audio interface {device}.\n")
        message += (
            "If you are on Mac OS X, try installing Soundflower "
            "(https://github.com/mattingalls/Soundflower).\n"
            "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
            "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
            "audio interface to use this.")
        print(message, file=sys.stderr)
        sys.exit(1)
    return caps


def main():
    args = get_parser().parse_args()
    args.model_path = "/home/kp/Remote/speech_references/denoiser/outputs/exp_/best.th"
    args.num_frames = 1
    model = get_model(args)
    model.eval()
    print("Model loaded.")
    streamer = DemucsStreamer(model, dry=args.dry, num_frames=args.num_frames)

    input_noisy_wav = "/home/kp/Remote/speech_references/denoiser/dataset/debug/noisy/p287_001.wav"
    in_frames, sr = torchaudio.load(input_noisy_wav)
    in_frames = in_frames.numpy()

    output_wav = []
    channels_out = 1
    first = True
    current_time = 0
    last_log_time = 0
    last_error_time = 0
    cooldown_time = 2
    log_delta = 10
    sr_ms = model.sample_rate / 1000
    stride_ms = streamer.stride / sr_ms
    print(f"Ready to process audio, total lag: {streamer.total_length / sr_ms:.1f}ms.")
    while True:
        try:
            if current_time > last_log_time + log_delta:
                last_log_time = current_time
                tpf = streamer.time_per_frame * 1000
                rtf = tpf / stride_ms
                print(f"time per frame: {tpf:.1f}ms, ", end='')
                print(f"RTF: {rtf:.1f}")
                streamer.reset_time_per_frame()

            length = streamer.total_length if first else streamer.stride
            first = False
            current_time += length / model.sample_rate
            frame = in_frames[:length]
            frame = torch.from_numpy(frame).mean(dim=1).to(args.device)
            with torch.no_grad():
                out = streamer.feed(frame[None])[0]
            if not out.numel():
                continue
            if args.compressor:
                out = 0.99 * torch.tanh(out)
            out = out[:, None].repeat(1, channels_out)
            mx = out.abs().max().item()
            if mx > 1:
                print("Clipping!!")
            out.clamp_(-1, 1)
            out = out.cpu().numpy()
            output_wav.append(out)
            # if overflow or underflow:
            #     if current_time >= last_error_time + cooldown_time:
            #         last_error_time = current_time
            #         tpf = 1000 * streamer.time_per_frame
            #         print(f"Not processing audio fast enough, time per frame is {tpf:.1f}ms "
            #               f"(should be less than {stride_ms:.1f}ms).")
        except KeyboardInterrupt:
            print("Stopping")
            break
    print(f"len(output) = {len(output_wav)}")


if __name__ == "__main__":
    main()
