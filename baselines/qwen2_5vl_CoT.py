# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from qwen2vl_flux_query import generate_image, save_images
# import torch
# import os
# from PIL import Image
# import random

# cache_directory = "../scratch/checkpoints/qwen2_5vl"

# cot_system_prompt = """
#     System Prompts:
#     Q:= Let the cellphone screen be blue and let the laptop be black
#     A:= 1) Let the cellphone screen be blue
#         2) Let the laptop be black
#     Q:= Add a bullet train and a bear and remove the boards
#     A:= 1) Add a bullet train
#         2) Add a bear
#         3) Remove the boards
#     Q:= Put a car on the screen of the laptop. It could be a hand next to the laptop. The hand could be holding a cup.
#     A:= 1) Put a car on the screen of the laptop.
#         2) It could be a hand next to the laptop.
#         3) The hand could be holding a cup.
#     Just give the points and do not include question prompt and A:= tag.
#     """

# # The default range for the number of visual tokens per image in the model is 4-16384.
# # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# # min_pixels = 256*28*28
# # max_pixels = 1280*28*28
# # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
#     # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     #     "Qwen/Qwen2.5-VL-7B-Instruct",
#     #     torch_dtype=torch.bfloat16,
#     #     attn_implementation="flash_attention_2",
#     #     device_map="auto",
#     # )

# def cot_subtasks(prompt):
#     initial_prompt = f"""Main Task: {prompt}\nBreak this task into a list of smaller subtasks. Give only the names of the smaller subtasks and not any description of that subtask. Just give the subtasks in as least number of points as possible. Give the answer following the pattern given in System Prompts without the <A:=>.\n"""

#     # default: Load the model on the available device(s)
#     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         "Qwen/Qwen2.5-VL-7B-Instruct",
#         torch_dtype="auto",
#         device_map="auto",
#         cache_dir=cache_directory
#     )
#     # default processor
#     processor = AutoProcessor.from_pretrained(
#         "Qwen/Qwen2.5-VL-7B-Instruct",
#         cache_dir=cache_directory
#     )

#     modified_prompt = cot_system_prompt + "\n" + initial_prompt

#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": modified_prompt},
#             ],
#         }
#     ]

#     # Preparation for inference
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")

#     # Inference: Generation of the output
#     generated_ids = model.generate(**inputs, max_new_tokens=128)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )

#     subtasks = output_text[0].strip().split("\n")
#     subtasks = [subtask.strip().lower() for subtask in subtasks if subtask.strip()]
#     list_length = len(subtasks)

#     # free CUDA memory
#     del model, processor, inputs, generated_ids, generated_ids_trimmed, output_text, image_inputs, video_inputs
#     torch.cuda.empty_cache()

#     return subtasks

# def answer(image_path, prompt):
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": image_path,
#                 },
#                 {"type": "text", "text": prompt},
#             ],
#         }
#     ]

#     # Preparation for inference
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")

#     # Inference: Generation of the output
#     generated_ids = model.generate(**inputs, max_new_tokens=128)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )

#     output = output_text[0].strip().split("\n")

#     # free CUDA memory
#     del inputs, generated_ids, generated_ids_trimmed
#     torch.cuda.empty_cache()

#     return output

# if __name__ == "__main__":

#     use_test = True # use the test folder examples

#     if use_test:
#         file_name = "test_image_input"
#         input_image = Image.open(f'./test/images/{file_name}.jpg')
#         with open(f'./test/prompts/{file_name}.txt', 'r') as f:
#             prompt = f.read()
#         output_dir = f"./test/cot_output"
#         os.makedirs(output_dir, exist_ok=True)

#         subtasks = cot_subtasks(prompt)
#         print(subtasks)
    
#         # Qwen2VL-Flux
#         for i, subtask in enumerate(subtasks):
#             output_images = generate_image(input_image, subtask)

#             random_idx = random.randint(0, len(output_images)-1)
#             input_image = output_images[random_idx]
#             torch.cuda.empty_cache()

#             # save intermediate images
#             save_images(f'{file_name}_step{i+1}', output_images, output_dir)

#         # Qwen2.5VL
#         # outputs = []
#         # for subtask in subtasks:
#         #     subtask_output = answer(input_image, subtask)
#         #     outputs.append(subtask_output)

#         # save output
#         # Qwen2.5VL
#         # with open(f'{output_dir}/{file_name}.txt', 'w') as f:
#         #     f.write(output)


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import random
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
cache_directory = "../scratch/checkpoints/qwen2_5vl"

# MODE:
#   "test" -> run on ./test/images/test_image_input.jpg + prompt
#   "csv"  -> run COT over data/filtered_dataset.csv
MODE = "csv"

CSV_INPUT = "../data/filtered_dataset.csv"
CSV_OUTPUT = "../data/filtered_dataset_with_cot.csv"
INSTRUCTION_COLUMN = "instruction"   # change if your column name differs

# ---------------------------------------------------------------------
# System prompt for COT
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Load model + processor ONCE
# ---------------------------------------------------------------------
print("Loading Qwen2.5-VL model and processor...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",          # let HF shard across devices if needed
    cache_dir=cache_directory,
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir=cache_directory,
)

# ---------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------
def cot_subtasks(prompt: str):
    """
    Take a natural-language instruction and return a list of COT subtasks.
    Uses global `model` and `processor`.
    """
    initial_prompt = (
        f"Main Task: {prompt}\n"
        "Break this task into a list of smaller subtasks. "
        "Give only the names of the smaller subtasks and not any description of that subtask. "
        "Just give the subtasks in as least number of points as possible. "
        "Give the answer following the pattern given in System Prompts without the <A:=>.\n"
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

    # Prep for inference (text-only, but we still call the same helpers)
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
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Trim off the prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    lines = output_text[0].strip().split("\n")
    subtasks = [line.strip() for line in lines if line.strip()]

    # free only big temps
    del inputs, generated_ids, generated_ids_trimmed, output_text, image_inputs, video_inputs
    torch.cuda.empty_cache()

    return subtasks


def answer(image_path_or_pil, prompt: str):
    """
    Optional: use Qwen2.5-VL to answer a question about an image + text.
    """
    # allow either a path or an already-loaded PIL image
    if isinstance(image_path_or_pil, str):
        image_obj = image_path_or_pil
    else:
        image_obj = image_path_or_pil

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": prompt},
            ],
        }
    ]

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
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    output_lines = [line.strip() for line in output_text[0].split("\n") if line.strip()]

    del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
    torch.cuda.empty_cache()

    return output_lines

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":

    if MODE == "test":
        from qwen2vl_flux_query import generate_image, save_images
        # === Original test pipeline: COT + Flux editing ===
        file_name = "test_image_input"
        input_image = Image.open(f"./test/images/{file_name}.jpg")
        with open(f"./test/prompts/{file_name}.txt", "r") as f:
            prompt = f.read().strip()

        output_dir = "./test/cot_output"
        os.makedirs(output_dir, exist_ok=True)

        print("Running COT on test prompt...")
        subtasks = cot_subtasks(prompt)
        print("Subtasks:", subtasks)

        # Qwen2VL-Flux loop
        for i, subtask in enumerate(subtasks):
            print(f"[{i+1}/{len(subtasks)}] Generating images for subtask: {subtask!r}")
            output_images = generate_image(input_image, subtask)

            random_idx = random.randint(0, len(output_images) - 1)
            input_image = output_images[random_idx]
            torch.cuda.empty_cache()

            # save intermediate images
            save_images(f"{file_name}_step{i+1}", output_images, output_dir)

        print("Done test mode.")

    elif MODE == "csv":
        # === CSV mode: run COT over all rows and save a new CSV ===
        print(f"Reading CSV from {CSV_INPUT} ...")
        df = pd.read_csv(CSV_INPUT)

        if INSTRUCTION_COLUMN not in df.columns:
            raise ValueError(
                f"Column '{INSTRUCTION_COLUMN}' not found in {CSV_INPUT}. "
                f"Available columns: {list(df.columns)}"
            )

        cot_results = []
        total = len(df)
        for i, instr in enumerate(df[INSTRUCTION_COLUMN]):
            if isinstance(instr, float) and pd.isna(instr):
                cot_results.append("")
                continue

            print(f"[{i+1}/{total}] COT for instruction: {str(instr)[:80]!r}")
            subtasks = cot_subtasks(str(instr))
            cot_results.append(" || ".join(subtasks))

        df["cot_subtasks"] = cot_results

        print(f"Writing augmented CSV to {CSV_OUTPUT} ...")
        df.to_csv(CSV_OUTPUT, index=False)
        print("Done CSV mode.")

    else:
        raise ValueError("MODE must be either 'test' or 'csv'.")
