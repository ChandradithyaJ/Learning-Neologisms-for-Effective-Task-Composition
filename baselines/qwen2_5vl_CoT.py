from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import random

cache_directory = "../scratch/checkpoints/qwen2_5vl"

cot_system_prompt = """
    System Prompts:
    Q:= Let the cellphone screen be blue and let the laptop be black
    A:= 1) Let the cellphone screen be blue
        2) Let the laptop be black
    Q:= Add a bullet train and a bear and remove the boards
    A:= 1) Add a bullet train
        2) Add a bear
        3) Remove the boards
    Q:= Put a car on the screen of the laptop. It could be a hand next to the laptop. The hand could be holding a cup.
    A:= 1) Put a car on the screen of the laptop.
        2) It could be a hand next to the laptop.
        3) The hand could be holding a cup.
    Just give the points and do not include question prompt and A:= tag.
    """

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

def cot_subtasks(prompt):
    initial_prompt = f"""Main Task: {prompt}\nBreak this task into a list of smaller subtasks. Give only the names of the smaller subtasks and not any description of that subtask. Just give the subtasks in as least number of points as possible. Give the answer following the pattern given in System Prompts without the <A:=>.\n"""

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
        cache_dir=cache_directory
    )
    # default processor
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        cache_dir=cache_directory
    )

    modified_prompt = cot_system_prompt + "\n" + initial_prompt

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": modified_prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    subtasks = output_text[0].strip().split("\n")
    subtasks = [subtask.strip().lower() for subtask in subtasks if subtask.strip()]
    list_length = len(subtasks)

    # free CUDA memory
    del model, processor, inputs, generated_ids, generated_ids_trimmed, output_text, image_inputs, video_inputs
    torch.cuda.empty_cache()

    return subtasks

if __name__ == "__main__":

    use_test = True # use the test folder examples

    if use_test:
        file_name = "test_image_input"
        with open(f'./test/prompts/{file_name}.txt', 'r') as f:
            prompt = f.read()

        subtasks = cot_subtasks(prompt)
        print(subtasks)