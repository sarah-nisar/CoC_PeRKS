#!/bin/sh
#!/usr/bin/env bash
eval "$(conda shell.bash hook)" # properly initialise non-interactive shell
conda activate newenv && cd Facial-Emotion-detection && cd src && python emotions.py --mode display

