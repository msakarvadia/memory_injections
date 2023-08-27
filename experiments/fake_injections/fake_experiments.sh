#!/bin/bash

#python fake_injection.py --tweak_factor 4 --layer_number 8 --model gpt2-small --dataset 2wmh
#python fake_injection.py --tweak_factor 8 --layer_number 4 --model gpt2-large --dataset 2wmh

python fake_injection.py --tweak_factor 3 --layer_number 7 --model gpt2-small --dataset hand --fake_data_type nouns
python fake_injection.py --tweak_factor 3 --layer_number 7 --model gpt2-small --dataset hand --fake_data_type verbs
python fake_injection.py --tweak_factor 3 --layer_number 7 --model gpt2-small --dataset hand --fake_data_type adjectives
python fake_injection.py --tweak_factor 3 --layer_number 7 --model gpt2-small --dataset hand --fake_data_type adverbs
python fake_injection.py --tweak_factor 3 --layer_number 7 --model gpt2-small --dataset hand --fake_data_type conjunctions
python fake_injection.py --tweak_factor 3 --layer_number 7 --model gpt2-small --dataset hand --fake_data_type top_5000

python fake_injection.py --tweak_factor 10 --layer_number 14 --model gpt2-large --dataset hand --fake_data_type nouns
python fake_injection.py --tweak_factor 10 --layer_number 14 --model gpt2-large --dataset hand --fake_data_type verbs
python fake_injection.py --tweak_factor 10 --layer_number 14 --model gpt2-large --dataset hand --fake_data_type adjectives
python fake_injection.py --tweak_factor 10 --layer_number 14 --model gpt2-large --dataset hand --fake_data_type adverbs
python fake_injection.py --tweak_factor 10 --layer_number 14 --model gpt2-large --dataset hand --fake_data_type conjunctions
python fake_injection.py --tweak_factor 10 --layer_number 14 --model gpt2-large --dataset hand --fake_data_type top_5000

python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type nouns
python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type verbs
python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type adjectives
python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type adverbs
python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type conjunctions
python fake_injection.py --tweak_factor 5 --layer_number 6 --model gpt2-small --dataset 2wmh --fake_data_type top_5000

python fake_injection.py --tweak_factor 9 --layer_number 8 --model gpt2-large --dataset 2wmh --fake_data_type nouns
python fake_injection.py --tweak_factor 9 --layer_number 8 --model gpt2-large --dataset 2wmh --fake_data_type verbs
python fake_injection.py --tweak_factor 9 --layer_number 8 --model gpt2-large --dataset 2wmh --fake_data_type adjectives
python fake_injection.py --tweak_factor 9 --layer_number 8 --model gpt2-large --dataset 2wmh --fake_data_type adverbs
python fake_injection.py --tweak_factor 9 --layer_number 8 --model gpt2-large --dataset 2wmh --fake_data_type conjunctions
python fake_injection.py --tweak_factor 9 --layer_number 8 --model gpt2-large --dataset 2wmh --fake_data_type top_5000
