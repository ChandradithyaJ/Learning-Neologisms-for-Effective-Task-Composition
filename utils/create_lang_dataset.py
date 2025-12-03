from datasets import load_dataset
import os

ds = load_dataset("khaimaitien/qa-expert-multi-hop-qa-V1.0")
df = ds["train"].to_pandas()

# get multihop questions with multiple subquestions
df = df[(df["multihop"] == True) & (df["sub_questions"].apply(lambda x: len(x) > 1))]
# we only need the question, subquestion and final answer fields
df = df[["question", "sub_questions", "final_answer"]]

# Create a "|" joined list of the sub-questions
df["subquestions"] = df["sub_questions"].apply(
    lambda lst: "|".join([d["question"] for d in lst])
)

# Create a "|" joined list of the sub-answers (long_answer fields)
df["subanswers"] = df["sub_questions"].apply(
    lambda lst: "|".join([d["long_answer"] for d in lst])
)

# Drop the original column
df = df.drop(columns=["sub_questions"])

# save to scratch
save_dir = "../scratch/DL_data"
os.makedirs(save_dir, exist_ok=True)
df.to_parquet(f"{save_dir}/processed_lang_dataset.parquet")