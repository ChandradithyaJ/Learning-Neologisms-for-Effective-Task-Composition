# Multitask-Image-Editing-via-Neologisms-and-Textual-Inversion
We propose leveraging neologisms to create a “new” word to represent composing tasks.

## Installation
### Set up a conda env
```
conda create -n mtask python==3.11
conda activate mtask
```

### Install the packages in editable model from the root
```
pip install -e .
```

## Get a HuggingFace access token
Create a `.env` and paste your HuggingFace token and the checkpoint directory location
```
HF_ACCESS_TOKEN=<token>
```

## Test
The images and prompt inside the `test` folder can be used for testing.