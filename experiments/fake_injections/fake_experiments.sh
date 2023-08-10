#!/bin/bash

#python fake_injection.py --tweak_factor 4 --layer_number 8 --model gpt2-small --dataset 2wmh
python fake_injection.py --tweak_factor 8 --layer_number 4 --model gpt2-large --dataset 2wmh
