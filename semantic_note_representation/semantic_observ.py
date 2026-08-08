 import pandas as pd
import pickle
import torch
import os
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import numpy as np
from tqdm import tqdm


train_ans = pd.read_csv("/workspace/data/Sudeshna/disease_progression/QA_model/Train_answerability_data.csv")
val_ans   = pd.read_csv("/workspace/data/Sudeshna/disease_progression/QA_model/Val_answerability_data.csv")
#test_ans  = pd.read_csv("/workspace/data/Sudeshna/disease_progression/QA_model/Test_answerability_data.csv")

train_notes = pd.read_excel("/workspace/data/Sudeshna/disease_progression/QA_model/train_Notes.xlsx")
val_notes   = pd.read_excel("/workspace/data/Sudeshna/disease_progression/QA_model/val_Notes.xlsx")
#test_notes  = pd.read_excel("/workspace/data/Sudeshna/disease_progression/QA_model/test_Notes.xlsx")

questions_df = pd.read_excel("/workspace/data/Sudeshna/disease_progression/QA_model/Final_question_bank.xlsx")

# note_id -> note text
train_note_map = dict(zip(train_notes.Note_id, train_notes.Summarized_note))
val_note_map   = dict(zip(val_notes.Note_id, val_notes.Summarized_note))
#test_note_map  = dict(zip(test_notes.Note_id, test_notes.Summarized_note))


# q_id -> question text
qid_to_question = dict(zip(questions_df.canonical_qid, questions_df.Question))

def attach_texts(ans_df, note_map, q_map):
   df = ans_df.copy()
   df["note_text"] = df["Note_id"].map(note_map)
   df["question_text"] = df["canonical_qid"].map(q_map)
   # sanity: drop broken rows
   df = df.dropna(subset=["note_text", "question_text"])
   return df

train_ans_df = attach_texts(train_ans, train_note_map, qid_to_question)
val_ans_df   = attach_texts(val_ans, val_note_map, qid_to_question)
#test_ans_df  = attach_texts(test_ans, test_note_map, qid_to_question)

cols = ["Note_id", "canonical_qid", "note_text", "question_text", "label"]
train_ans_df = train_ans_df[cols]
val_ans_df   = val_ans_df[cols]
#test_ans_df  = test_ans_df[cols]

print("Train:", train_ans_df.shape)
print("Val:", val_ans_df.shape)
#print("Test:", test_ans_df.shape)

# def stratified_subset(df, frac=0.3, seed=42):
#    pos = df[df.label == 1]
#    neg = df[df.label == 0]
#    pos_sub = pos.sample(frac=frac, random_state=seed)
#    neg_sub = neg.sample(frac=frac, random_state=seed)
#    return pd.concat([pos_sub, neg_sub]).sample(frac=1, random_state=seed)

# train_df = stratified_subset(train_ans_df, frac=0.3)
# print("Subset size:", len(train_df))

def subset_keep_all_pos_sample_neg(
   df,
   neg_frac=0.1,   # keep only 10% of negatives
   seed=42
):
   pos = df[df.label == 1]          # keep ALL positives
   neg = df[df.label == 0]
   neg_sub = neg.sample(
       frac=neg_frac,
       random_state=seed
   )
   return (
       pd.concat([pos, neg_sub])
       .sample(frac=1, random_state=seed)
       .reset_index(drop=True)
   )

train_df = subset_keep_all_pos_sample_neg(
   train_ans_df,
   neg_frac=0.1
)
print("Train size:", len(train_df))


class AnswerabilityModel(nn.Module):
   def __init__(self, encoder_name="emilyalsentzer/Bio_ClinicalBERT"):
       super().__init__()
       self.encoder = AutoModel.from_pretrained(encoder_name)
       self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
   def forward(self, input_ids, attention_mask, token_type_ids=None):
       outputs = self.encoder(
           input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids
       )
       cls_emb = outputs.last_hidden_state[:, 0]
       logits = self.classifier(cls_emb)
       return logits.squeeze(-1)



tokenizer = AutoTokenizer.from_pretrained(
   "emilyalsentzer/Bio_ClinicalBERT"
)
def encode_batch(questions, notes, max_len=512):
   texts = [
       q + " [SEP] " + n
       for q, n in zip(questions, notes)
   ]
   return tokenizer(
       texts,
       truncation=True,
       padding=True,
       max_length=max_len,
       return_tensors="pt"
   )

# pos_ratio = train_ans_df["label"].mean()
# pos_weight = torch.tensor(
#    [(1 - pos_ratio) / pos_ratio],
#    dtype=torch.float
# ).cuda()

criterion = nn.BCEWithLogitsLoss()

model = AnswerabilityModel().cuda()
optimizer = AdamW(model.parameters(), lr=1e-5)

def train_epoch(df, batch_size=32, epoch_idx=0):
   model.train()
   df = df.sample(frac=1).reset_index(drop=True)
   total_loss = 0.0
   num_batches = 0
   print(f"\n[Epoch {epoch_idx}] Training started")
   for i in tqdm(
       range(0, len(df), batch_size),
       desc=f"Epoch {epoch_idx} | Training",
       leave=False
   ):
       batch = df.iloc[i:i + batch_size]
       enc = encode_batch(
           batch["question_text"].tolist(),
           batch["note_text"].tolist()
       )
       labels = torch.tensor(
           batch["label"].values,
           dtype=torch.float
       ).cuda()
       enc = {k: v.cuda() for k, v in enc.items()}
       logits = model(**enc)
       loss = criterion(logits, labels)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       total_loss += loss.item()
       num_batches += 1
   avg_loss = total_loss / max(1, num_batches)
   print(f"[Epoch {epoch_idx}] Training completed | Avg Loss: {avg_loss:.4f}")
   return avg_loss

@torch.no_grad()
def evaluate(df, batch_size=32, epoch_idx=0):
   model.eval()
   all_probs, all_labels = [], []
   print(f"[Epoch {epoch_idx}] Validation started")
   for i in tqdm(
       range(0, len(df), batch_size),
       desc=f"Epoch {epoch_idx} | Validation",
       leave=False
   ):
       batch = df.iloc[i:i + batch_size]
       enc = encode_batch(
           batch["question_text"].tolist(),
           batch["note_text"].tolist()
       )
       enc = {k: v.cuda() for k, v in enc.items()}
       logits = model(**enc)
       probs = torch.sigmoid(logits).cpu().numpy()
       all_probs.extend(probs)
       all_labels.extend(batch["label"].values)
   auc = roc_auc_score(all_labels, all_probs)
   print(f"[Epoch {epoch_idx}] Validation completed | AUC: {auc:.4f}")
   return auc, all_probs, all_labels

def save_checkpoint(epoch, model, optimizer, best_auc, path):
   torch.save({
       "epoch": epoch,
       "model": model.state_dict(),
       "optimizer": optimizer.state_dict(),
       "best_auc": best_auc
   }, path)


CHECKPOINT_DIR = "/workspace/data/Sudeshna/disease_progression/QA_model/finetune_checkpoints_Epoch5"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


NUM_EPOCHS = 1
best_auc = 0.0
start_epoch = 1

print("#########################################")
print("### Answerability Gating Model Training ###")
print("#########################################")
for epoch in range(start_epoch, NUM_EPOCHS + 1):
   train_loss = train_epoch(
       train_df,
       batch_size=32,
       epoch_idx=epoch
   )
   val_auc, probs, labels = evaluate(
       val_ans_df,
       batch_size=32,
       epoch_idx=epoch
   )
   print(
       f"\n===== Epoch {epoch} Summary =====\n"
       f"Train Loss : {train_loss:.4f}\n"
       f"Val AUC    : {val_auc:.4f}\n"
   )

   if val_auc > best_auc:
       best_auc = val_auc
       save_checkpoint(
           epoch,
           model,
           optimizer,
           best_auc,
           path=os.path.join(CHECKPOINT_DIR, "best.pt")
       )
       #print(f" New best model saved (Val AUC = {best_auc:.4f})")

   save_checkpoint(
       epoch,
       model,
       optimizer,
       best_auc,
       path=os.path.join(CHECKPOINT_DIR, "latest.pt")
   )
   print(f" New best model saved (Val AUC = {best_auc:.4f})")

SAVE_PATH = "/workspace/data/Sudeshna/disease_progression/QA_model/answerability_gating_model_finetune_Epoch5.pt"
torch.save(model.state_dict(), SAVE_PATH)
print(f"Answerability Gating Model saved to {SAVE_PATH}")

#auc, probs, labels = evaluate(val_ans_df)
precision, recall, thresholds = precision_recall_curve(labels, probs)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.argmax(f1[:-1])
best_threshold = thresholds[best_idx]
print("Best threshold τ =", best_threshold)

metadata = {
   "encoder": "emilyalsentzer/Bio_ClinicalBERT",
   "threshold": best_threshold,        # τ from validation
   "learning_rate": 1e-5,
   "epochs": 1,
}
torch.save(metadata, "/workspace/data/Sudeshna/disease_progression/QA_model/answerability_gating_metadata_finetune_Epoch5.pt")
