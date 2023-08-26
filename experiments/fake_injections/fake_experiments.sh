#!/bin/bash

#python fake_injection.py --tweak_factor 4 --layer_number 8 --model gpt2-small --dataset 2wmh
#python fake_injection.py --tweak_factor 8 --layer_number 4 --model gpt2-large --dataset 2wmh

python fake_injection.py --tweak_factor 4 --layer_number 2 --model gpt2-small --dataset hand
python fake_injection.py --tweak_factor 13 --layer_number 11 --model gpt2-large --dataset hand
python fake_injection.py --tweak_factor 6 --layer_number 7 --model gpt2-small --dataset 2wmh
python fake_injection.py --tweak_factor 9 --layer_number 8 --model gpt2-large --dataset 2wmh
