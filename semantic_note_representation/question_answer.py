 # %% 
import transformers 
import torch
import pandas as pd
import numpy as np
#import xlsxwriter
import zipfile
import random 
import json
from tqdm import tqdm


# -------------------------
# Model setup
# -------------------------
model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
   "text-generation",
   model=model_id,
   model_kwargs={"torch_dtype": torch.bfloat16},
   device_map="auto",
   token=""
)


# -------------------------
# File paths
# -------------------------
note_file = "/workspace/data/Sudeshna/disease_progression/answer_generation/train_Notes.xlsx"
question_file = "/workspace/data/Sudeshna/disease_progression/answer_generation/Final_question_bank.xlsx"
candidate_file = "/workspace/data/Sudeshna/disease_progression/answer_generation/train_candidate_sets.xlsx"
output_file = "/workspace/data/Sudeshna/disease_progression/answer_generation/answer_train_Notes_5000_last.xlsx"


# -------------------------
# Load notes
# -------------------------
note_df = pd.read_excel(note_file)
note_dict = dict(zip(note_df["Note_id"], note_df["Note_clean"]))
# -------------------------
# Load questions
# -------------------------
questions_df = pd.read_excel(question_file)
question_dict = dict(zip(questions_df["canonical_qid"], questions_df["Question"]))

# -------------------------
# Load candidate sets
# candidate file:
# Note_id | q_ids (q1#q2#q3...)
# -------------------------
candidate_df = pd.read_excel(candidate_file)

# -------------------------
# Reproducibility
# -------------------------
random.seed(42)
torch.manual_seed(42)


# -------------------------
# Prompt template
# -------------------------
def build_prompt(note_text, qid_list):
   questions_block = "\n".join(
       [f"{i+1}. ({qid}) {question_dict[qid]}" for i, qid in enumerate(qid_list)]
   )
   return f"""
Clinical Note:
\"\"\"
{note_text}
\"\"\"
Below are yes/no clinical questions about the patient.
For EACH question, decide:
- YES: clearly supported by the note
- NO: clearly contradicted by the note
- CANNOT_ANSWER: not enough information or unclear
Rules:
- Use ONLY the information in the note
- Do NOT infer or assume
- If uncertain, choose CANNOT_ANSWER
Questions:
{questions_block}
Output format (STRICT JSON only):
[
 {{"question_id": "<ID>", "label": "YES|NO|CANNOT_ANSWER"}}
]
"""


# -------------------------
# Inference loop
# -------------------------
results = []
note_ids = []
for i in range(len(candidate_df)):
   print(i)
   note_id = candidate_df["Note_id"][i]
   note_ids.append(note_id) 
   note_text = note_dict[note_id]
   # Parse candidate questions
   qids = candidate_df["canonical_qids"][i].split("#")
   qids = [q for q in qids if q in question_dict]
   
   # Randomly sample up to 20 questions
   sampled_qids = random.sample(qids, min(20, len(qids)))
   prompt = build_prompt(note_text, sampled_qids)
   messages = [
       {
           "role": "system",
           "content": (
               "You are a clinical question answering system. "
               "Return only the required JSON output."
           )
       },
       {"role": "user", "content": prompt}
   ]
   outputs = pipeline(
       messages,
       max_new_tokens=800,
       do_sample=False,
       temperature=0.0,
       top_p=1.0
   )
   results.append(outputs[0]["generated_text"][-1]['content'])
   print(outputs[0]["generated_text"][-1]['content'])

   new_df = pd.DataFrame({"Note_id":[i for i in note_ids],"Results":[i for i in results]})
   new_df.to_excel(output_file)
