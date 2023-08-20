#!/bin/bash

#iterate through models
for model in gpt2-small gpt2-large
do
    #iterate through dataset
    for data in hand 2wmh
    do
        #iterate thru pos data
        for pos in top_5000 nouns verbs adjectives adverbs conjunctions
        do
            python pos_tweaK_factor_analysis.py --model $model --dataset $data --memory_dataset $pos 
        done
    done
done
