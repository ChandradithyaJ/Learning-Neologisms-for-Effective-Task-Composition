import pandas as pd
import os
import codecs
import ast

df = pd.read_csv('../scratch/DL_data/filtered_dataset_with_cot.csv')
print(df.columns)
images = df[['source_img', 'target_img', 'reject_img']]
del df

save_original_images_dir = '../scratch/DL_data/imgs/original'
save_final_images_dir = '../scratch/DL_data/imgs/final'
save_reject_images_dir = '../scratch/DL_data/imgs/reject'
for d in [save_original_images_dir, save_final_images_dir, save_reject_images_dir]:
    os.makedirs(d, exist_ok=True)

for idx, row in images.iterrows():
    # Write files
    with open(f'{save_original_images_dir}/{idx}.png', 'wb') as f:
        f.write(ast.literal_eval(row['source_img'])['bytes'])

    with open(f'{save_final_images_dir}/{idx}.png', 'wb') as f:
        f.write(ast.literal_eval(row['target_img'])['bytes'])

    with open(f'{save_reject_images_dir}/{idx}.png', 'wb') as f:
        f.write(ast.literal_eval(row['reject_img'])['bytes'])



# prompts = df[['instruction', 'cot_subtasks']]
# del df
# print(prompts.head())

# save_base_prompts_dir = '../scratch/DL_data/prompts/composite'
# save_cot_prompts_dir = '../scratch/DL_data/prompts/CoT'
# os.makedirs(save_base_prompts_dir, exist_ok=True)
# os.makedirs(save_cot_prompts_dir, exist_ok=True)

# for idx, row in prompts.iterrows():
#     with open(f'{save_base_prompts_dir}/{idx}.txt', 'w') as f:
#         f.write(str(row['instruction']))

#     with open(f'{save_cot_prompts_dir}/{idx}.txt', 'w') as f:
#         f.write(str(row['cot_subtasks']))