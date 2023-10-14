# !/bin/bash

learning_rate=("1e-4" "4e-4" "8e-4" "1e-3" "4e-3")
encoded_layer=(2 4 6 8 10)

for l in ${encoded_layer[@]}; do
    for lr in ${learning_rate[@]}; do
        python3 main.py -l=$l --lr=$lr 
    done
done