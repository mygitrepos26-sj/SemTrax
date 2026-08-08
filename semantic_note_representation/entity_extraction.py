# %% 
import transformers 
import torch
import pandas as pd
import numpy as np
#import xlsxwriter
import zipfile
import random 

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

input_file = "/workspace/data/Sudeshna/disease_progression/note_embeddings/val_Notes_parse_failures.xlsx"
output_file = "/workspace/data/Sudeshna/disease_progression/note_embeddings/entity_extraction_val_Notes_parse_failures.xlsx"

df=pd.read_excel(input_file)

# Set random seed for reproducibility 
torch.manual_seed(42) 
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(42)

torch.backends.cuda.enable_mem_efficient_sdp(False) 
torch.backends.cuda.enable_flash_sdp(False) 

note_ids = []
notes = []
solution = []

for i in range(len(df)):
    print(i)
    if(type(df['Summarized_note'][i]) != float):
        note_ids.append(df['Note_id'][i])
        note = df['Summarized_note'][i]
        notes.append(df['Summarized_note'][i])
    else:
        continue

    messages = [ 
        {"role": "system","content": ("""You are an expert clinical text analyzer. Extract ONLY the entities that are explicitly mentioned in the given nursing note. Do not infer or assume anything beyond what is written.

Entities to extract:

1. Diseases/Symptoms:
   - Include disease names, symptoms, and any abnormalities mentioned.
   - If a symptom or condition is explicitly stated as absent or negative for the patient, prefix it with "neg_".

2. Implemented Procedures:
   - Include only therapeutic or surgical procedures performed or planned (e.g., intubation, catheter insertion, surgery).
   - Do NOT include diagnostic tests like ECG, X-ray, MRI here.

3. Medications:
   - Include all medication names mentioned.

Return the output in three separate lists in JSON format as shown below:

{
  "diseases_symptoms": [ ... ],
  "procedures": [ ... ],
  "medications": [ ... ]
}

Example:
Input: "Patient reports chest pain and shortness of breath. No fever. Central line insertion performed. Prescribed aspirin."
Output:
{
  "diseases_symptoms": ["chest pain", "shortness of breath", "neg_fever"],
  "procedures": ["central line insertion"],
  "medications": ["aspirin"]
}

Do not provide explanations, reasoning, or any additional text just the required output.""")},
         
        {"role": "user", 
         "content": ( 
             f"""The given Nursing Note is: \n {note}"""
             """Extract ONLY the entities that are explicitly mentioned in the given nursing note. Do not infer or assume anything beyond what is written.

Entities to extract:

1. Diseases/Symptoms:
   - Include disease names, symptoms, and any abnormalities mentioned.
   - If a symptom or condition is explicitly stated as absent or negative for the patient, prefix it with "neg_".

2. Implemented Procedures:
   - Include only therapeutic or surgical procedures performed or planned (e.g., intubation, catheter insertion, surgery).
   - Do NOT include diagnostic tests like ECG, X-ray, MRI here.

3. Medications:
   - Include all medication names mentioned.

Return the output in three separate lists in JSON format as shown below:

{
  "diseases_symptoms": [ ... ],
  "procedures": [ ... ],
  "medications": [ ... ]
}

Example:
Input: "Patient reports chest pain and shortness of breath. No fever. Central line insertion performed. Prescribed aspirin."
Output:
{
  "diseases_symptoms": ["chest pain", "shortness of breath", "neg_fever"],
  "procedures": ["central line insertion"],
  "medications": ["aspirin"]
}
""" )} 
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
    print(note, outputs[0]["generated_text"][-1]['content'])

# %%
    new_df = pd.DataFrame({"Note_id":[i for i in note_ids],"Results":[i for i in solution]})
    new_df.to_excel(output_file)
