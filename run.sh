# !/bin/bash
encoded_layer=(2 4 6 8 10 12)
task="dan"

for l in ${encoded_layer[@]}; do
    python3 main.py -l=$l -t=${task}
done