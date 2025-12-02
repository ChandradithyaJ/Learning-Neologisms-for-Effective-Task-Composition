# Multitask-Image-Editing-via-Neologisms-and-Textual-Inversion
We propose leveraging neologisms to create a “new” word to represent composing tasks.

## Installation
### Set up a conda env
```
conda create -n mtask python==3.11
conda activate mtask
```

<!-- ### Djrango/Qwen2vl-Flux
Skip this code block if the `qwen2vl_flux/` folder is already present.
```
git clone https://github.com/erwold/qwen2vl-flux
cd qwen2vl-flux
```

Update the protobuf version. Replace `protobuf==4.23.4` with
```
protobuf==4.25.3,<5
```
and install the requirements
```
pip install -r requirements.txt
```

Add an `__init__.py` to allow imports from this folder and change the folder name
```
touch __init__.py
cd ..
mv qwen2vl-flux/ qwen2vl_flux/
``` -->

### Install the packages in editable model from the root
```
pip install -e .
```

## Get a HuggingFace access token
Create a `.env` and paste your HuggingFace token and the checkpoint directory location
```
HF_ACCESS_TOKEN=<token>
CHECKPOINT_DIR="../scratch/checkpoints/qwen2vl_flux"
```

## Test
The images and prompt inside the `test` folder can be used for testing.