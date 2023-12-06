# How to run the program
* Please first read the requirement.txt, install the required packages, and read the ArgumentParser for each files.
* In order to run `main.py`, you are required to modify the PyTorch source code so that the vision transformer can recieve inputs that have more than 3 channels (use the `diff` command to see the modification of `modification/modified_vision_transformer` and `modification/ori_vision_transformer`), where the original path of the source code is `https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py`.

```python=  
$ python main.py # 棋力模仿
$ python result_collector.py -t=<path_to_dataset *.csv> -m=<path_to_your_model> -o=<path_to_store_the_result> 


$ python train_blender.py # 棋風辨識
$ python blending_collector.py -t=<path_to_dataset *.csv> -m=<path_to_your_model> --output=<path_to_store_the_result> 
```

# File discriptions

* baseline_model.py: implements the ResNet for blending
* blender.py: the models and training pipeline of blending
* blending_collector.py: collects the result for submission
* GoDataset: classes and functions for reading data and buildindg Pytorch Dataset
* GoEnv: the game of Go leveraging other low-level api
* gogame.py: originally from https://github.com/aigagror/GymGo (with modification), the low-level api for the game of Go
* GoParser: parse the csv files
* goutils.py: a lots of helper functions, such as formatting the moves, so that differents module can communicate.
* govars.py: global variables
* main.py: run and initialize models for task 1
* result_collector.py: collects the result for submission
* state_utils.py:originally from https://github.com/aigagror/GymGo (with modification), the low-level api to calculate the state of Go
* train_blender.py: start the training of blender.py
* trainer.py: the wrapper for training ViT for task 1.