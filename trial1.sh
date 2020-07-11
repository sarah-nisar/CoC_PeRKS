#!/bin/sh
#!/usr/bin/env bash
eval "$(conda shell.bash hook)" #properly initialise non-interactive shell
conda activate newenv && cd emotion-recognition-using-speech-master && python deep_emotion_recognition.py && python mytrain.py 