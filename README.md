# Self driving winnie

## Prerequisites

- [Python 3](https://www.python.org/)

## How to use

This section provides a quick start guide on to use this application.

### Requirements 

Make sure by running `pip install -r requirements.txt` that all the requirements are installed. 

### Create your model

1. You can define your own model in `create_model.py`.
2. You can then generate the model as a json file `python3 create_model.py`.

We are using [NVIDIA model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) for self driving cars.

### Record samples in order to train your model

By launching `python3 record.py`, you can control cozmo using a controler and record some image samples.

You can also modify `record.py` to change the controls.

### Training

After recording, you can train your model using `python3 train.py`.

### Autopilot

When training is done, you can let cozmo impress you by running `python3 auto_driving.py`.

![Winnie](https://media.giphy.com/media/13R4gPwPpLHQwU/giphy.gif)
