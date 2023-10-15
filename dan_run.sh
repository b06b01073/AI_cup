# !/bin/bash
encoded_layer=(4 8 10)
patch_size=(7 3)
task="dan"

for l in ${encoded_layer[@]}; do
    for p in ${patch_size[@]}; do
        python3 main.py -l=$l -p=${p} -t=${task}
    done
done