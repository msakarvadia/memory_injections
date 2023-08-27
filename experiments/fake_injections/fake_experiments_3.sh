#!/bin/bash

python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type nouns
python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type verbs
python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type adjectives
python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type adverbs
python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type conjunctions
python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type top_5000
