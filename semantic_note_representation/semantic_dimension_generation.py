 # %% 
import transformers 
import torch
import pandas as pd
import numpy as np
#import xlsxwriter
import zipfile
import random
import json
from pathlib import Path
from itertools import islice


INPUT_JSON = Path("/workspace/data/Sudeshna/disease_progression/contrastive_sets.jsonl") 
fin = INPUT_JSON.open()
#INPUT_JSON = "/workspace/data/Sudeshna/disease_progression/contrastive_sets.jsonl"
output_file = "/workspace/data/Sudeshna/disease_progression/generated_questions_1500_last.xlsx"

model_id = "meta-llama/Llama-3.1-8B-Instruct" 
#model_id = "meta-llama/Llama-3.2-1B-Instruct" 
#model_id = "meta-llama/Llama-3.2-3B-Instruct"

pipeline = transformers.pipeline( 
    "text-generation", 
    model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="auto", 
    token="" 
) 
# %% 

# Set random seed for reproducibility 
torch.manual_seed(42) 
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(42)

torch.backends.cuda.enable_mem_efficient_sdp(False) 
torch.backends.cuda.enable_flash_sdp(False)


def build_prompt(anchor_obj):
    # --------------------------
    # GROUP A (anchor + positives)
    # --------------------------
    anchor_docid = anchor_obj["anchor_doc_id"]
    anchor_note  = anchor_obj["anchor_Note"]

    groupA = []
    groupA.append({"doc_id": anchor_docid, "note": anchor_note})

    for p in anchor_obj.get("positives", []):
        groupA.append({
            "doc_id": p["doc_id"],
            "note": p["note"]
        })

    groupA = groupA[:6]   # 1 anchor + 5 positives

    # --------------------------
    # GROUP B (hard negatives)
    # --------------------------
    groupB = []
    for p in anchor_obj.get("hard_negatives", []):
        groupB.append({
            "doc_id": p["doc_id"],
            "note": p["note"]
        })
    groupB = groupB[:5]

    # --------------------------
    # GROUP C (easy negatives)
    # --------------------------
    groupC = []
    for p in anchor_obj.get("easy_negatives", []):
        groupC.append({
            "doc_id": p["doc_id"],
            "note": p["note"]
        })
    groupC = groupC[:5]

    # --------------------------
    # BUILD PROMPT
    # --------------------------
    lines = []
    lines.append(
        f" You are given three grouped NOTES below, "
        f"generate 10 most discriminative yes/no clinical questions (<=12 words) that distinguish GROUP A from GROUPS B and C."
    )

    lines.append(
        "Format the questions in a numbered list as shown below:\n"
        " 1. <first yes/no question>\n"
        " 2. <second yes/no question>\n"
        " 3. <third yes/no question>\n"
        "...\n"
        " 10. <tenth yes/no question>"
    )

    # ---- GROUP A ----
    lines.append("\nGROUP A (anchor + similar NOTES):")
    for i, d in enumerate(groupA, 1):
        safe_note = d["note"].replace('"','\\"')
        lines.append(f"A{i}: DOC_ID: {d['doc_id']}\nNOTE: \"{safe_note}\"")

    # ---- GROUP B ----
    lines.append("\nGROUP B (hard negatives):")
    for i, d in enumerate(groupB, 1):
        safe_note = d["note"].replace('"','\\"')
        lines.append(f"B{i}: DOC_ID: {d['doc_id']}\nNOTE: \"{safe_note}\"")

    # ---- GROUP C ----
    lines.append("\nGROUP C (easy negatives):")
    for i, d in enumerate(groupC, 1):
        safe_note = d["note"].replace('"','\\"')
        lines.append(f"C{i}: DOC_ID: {d['doc_id']}\nNOTE: \"{safe_note}\"")

    lines.append("\nNow produce ONLY the list of questions. Do not provide explanations, reasoning, or any additional text just the required output.")
    return "\n".join(lines)




anchor_numbers = []
solution = []

#k = 500
#for i, line in enumerate(fin):
#for i, item in enumerate(islice(fin, k)):
start = 1500  # starting line (inclusive)
end = 1848    # one past the last line (exclusive)
#for i, item in enumerate(islice(fin, k)):
for i, item in enumerate(islice(fin, start - 1, end), start=start):
#for i, item in enumerate(fin):
    print(i)
    #print(i)
    anchor_obj = json.loads(item)
    id_ = anchor_obj["anchor_doc_id"]
    anchor_numbers.append(id_)
    print(id_)
    user_prompt = build_prompt(anchor_obj)
    #print(user_prompt)
    messages = [ 
        {"role": "system","content": ("""You are a clinical question generator. You are given three groups of NOTES:
    GROUP A: Notes that are clinically similar to each other (anchor + positives).
    GROUP B: Hard negatives — notes that appear similar but differ in key clinical aspects.
    GROUP C: Easy negatives — notes that differ more obviously.

Your task is to generate 10 most discriminative yes/no clinical questions (each ≤12 words) that effectively distinguish GROUP A from GROUPS B and C.

INSTRUCTIONS:
    Use ONLY the information contained in the NOTES. Do NOT invent clinical facts.
    Every question must be answerable strictly from the text of the NOTES.
    Prefer questions focusing on diseases, symptoms, co-morbidities, clinical states, interventions, medications, or changes that clearly separate the clinical patterns seen in GROUP A from the negatives GROUP B and GROUP C.
    Each question should yield the SAME answer for ALL notes in GROUP A and the OPPOSITE answer for ALL notes in GROUPS B and C.
    Format the questions in a numbered list as shown below:
        1. <first yes/no question>
        2. <second yes/no question>
        3. <third yes/no question>
        ...
        10. <tenth yes/no question>
Do not provide explanations, reasoning, or any additional text just the required list of questions.""")},
         
        {"role": "user", 
         "content": (user_prompt)} 
        ] 
    # Optimize terminators 
    terminators = [ 
        pipeline.tokenizer.eos_token_id, 
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>") 
    ] 
    # Response generation with reduced randomness 
    outputs = pipeline( 
        messages, 
        max_new_tokens=10000,  # Adjust based on your needs 
        eos_token_id=terminators, 
        do_sample=False,  # Disable sampling for deterministic results 
        temperature=0.0,  # No randomness 
        top_p=1.0,       # Consider all tokens 
    ) 
    solution.append(outputs[0]["generated_text"][-1]['content'])
    print(outputs[0]["generated_text"][-1]['content'])

# %%
    new_df = pd.DataFrame({"Anchor_ID":[i for i in anchor_numbers], "Results":[i for i in solution]})
    new_df.to_excel(output_file)
