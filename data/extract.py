import os
import pandas as pd
from PIL import Image
from io import BytesIO
import ast

# === Step 1: Define paths relative to the script location (inside data/)
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'filtered_dataset_with_cot.csv')



# === Step 2: Setup image output directories
images_dir = os.path.join(base_dir, 'images')
original_dir = os.path.join(images_dir, 'original')
final_dir = os.path.join(images_dir, 'final')
reject_dir = os.path.join(images_dir, 'reject')

os.makedirs(original_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)
os.makedirs(reject_dir, exist_ok=True)

# === Step 3: Load CSV
df = pd.read_csv(csv_path)

# === Step 4: Extract and save images
for idx, row in df.iterrows():
    try:
        # Parse string into dict
        source_dict = ast.literal_eval(row['source_img'])
        target_dict = ast.literal_eval(row['target_img'])
        reject_dict = ast.literal_eval(row['reject_img'])

        # Extract raw bytes
        source_bytes = source_dict['bytes']
        target_bytes = target_dict['bytes']
        reject_bytes = reject_dict['bytes']

        # Decode to PIL images
        source_img = Image.open(BytesIO(source_bytes))
        target_img = Image.open(BytesIO(target_bytes))
        reject_img = Image.open(BytesIO(reject_bytes))

        # Save with index-based naming
        i = idx + 1
        source_img.save(os.path.join(original_dir, f"{i}.png"))
        target_img.save(os.path.join(final_dir, f"{i}.png"))
        reject_img.save(os.path.join(reject_dir, f"{i}.png"))

    except Exception as e:
        print(f"❌ Error at row {idx}: {e}")
        continue

print("✅ Done saving images under /data/images/")
