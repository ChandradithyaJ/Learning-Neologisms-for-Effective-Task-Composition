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
## Train (Qwen-Image-Edit)
To run qwen-image-edit training:
First change the DATASET_PATH in /neologism_training/qwen_image_edit/train_cliploss.py, to a dataframe with columns "source_img", "target_img", and "instruction", ensure that every entry in "instruction" contains the word "and".
'''
python3 /Multitask-Image-Editing-via-Neologisms-and-Textual-Inversion/neologism_training/qwen_image_edit/train_cliploss.py
'''

## Eval (Qwen-Image-Edit)
Run the following command to eval, change path to neolgism_ckpt and dataset
'''
python /content/Multitask-Image-Editing-via-Neologisms-and-Textual-Inversion/neologism_training/qwen_image_edit/eval.py \
  --dataset /content/sample_data/eval_dataset.csv \
  --neologism_ckpt /content/Neologism_training/outputs_neologism_and/and_neologism_epoch_005.pt \
  --subdir eval \
  --max_images 50
'''

## Example Google Colab notebook 
The following link is an example of training and evaluating neologism on google colab (Requires A100 High-RAM)
https://colab.research.google.com/drive/1YhoqKIIAAe7QTITWB3XAVQ6ZpBImN9gi?usp=sharing


## Test
The images and prompt inside the `test` folder can be used for testing.