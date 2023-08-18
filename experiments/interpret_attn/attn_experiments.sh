#!/bin/bash

python interpret_attn.py --dataset hand --model_name gpt2-small
python interpret_attn.py --dataset hand --model_name gpt2-large
python interpret_attn.py --dataset 2wmh --model_name gpt2-small
python interpret_attn.py --dataset 2wmh --model_name gpt2-large
