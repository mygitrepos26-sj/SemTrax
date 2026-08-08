 import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
# ==========================================
# Config
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 384
BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 15
CHECKPOINT_DIR = "/workspace/data/Sudeshna/disease_progression/answer_polarity/polarity_checkpoints_summary_Epoch15"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LABEL_MAP = {0: 0, 1: 1, -1: 2}
# ==========================================
# Load Data
# ==========================================
train_df = pd.read_excel("/workspace/data/Sudeshna/disease_progression/answer_polarity/polarity_train.xlsx")
val_df   = pd.read_excel("/workspace/data/Sudeshna/disease_progression/answer_polarity/polarity_val.xlsx")
test_df  = pd.read_excel("/workspace/data/Sudeshna/disease_progression/answer_polarity/polarity_test.xlsx")
notes_df = pd.read_excel("/workspace/data/Sudeshna/disease_progression/answer_polarity/notes1.xlsx")
ques_df  = pd.read_excel("/workspace/data/Sudeshna/disease_progression/answer_polarity/Final_question_bank.xlsx")
#USE SUMMARIZED NOTE
note_map = dict(zip(notes_df.Note_id, notes_df.Summarized_note))
ques_map = dict(zip(ques_df.canonical_qid, ques_df.Question))
# ==========================================
# Balance TRAIN DATA (ONLY TRAIN)
# ==========================================
train_0 = train_df[train_df.Label == 0]
train_1 = train_df[train_df.Label == 1]
train_2 = train_df[train_df.Label == -1]
target_size = int((len(train_1) + len(train_2)) / 2)
train_0_bal = train_0.sample(
   n=min(len(train_0), target_size),
   random_state=42
)
train_df_balanced = pd.concat(
   [train_0_bal, train_1, train_2]
).sample(frac=1, random_state=42).reset_index(drop=True)
print("Training label distribution AFTER balancing:")
print(train_df_balanced.Label.value_counts())
# ==========================================
# Tokenizer
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# ==========================================
# Dataset (NO CHUNKING)
# ==========================================
class PolarityDataset(Dataset):
   def __init__(self, df):
       self.df = df.reset_index(drop=True)
   def __len__(self):
       return len(self.df)
   def __getitem__(self, idx):
       row = self.df.iloc[idx]
       note = note_map[row.Note_id]
       question = ques_map[row.question_id]
       label = LABEL_MAP[row.Label]
       text = question + " [SEP] " + note
       enc = tokenizer(
           text,
           truncation=True,
           padding="max_length",
           max_length=MAX_LEN,
           return_tensors="pt"
       )
       return {
           "input_ids": enc["input_ids"].squeeze(0),
           "attention_mask": enc["attention_mask"].squeeze(0),
           "label": torch.tensor(label, dtype=torch.long)
       }
# ==========================================
# DataLoaders
# ==========================================
train_loader = DataLoader(
   PolarityDataset(train_df_balanced),
   batch_size=BATCH_SIZE,
   shuffle=True
)
val_loader = DataLoader(
   PolarityDataset(val_df),
   batch_size=BATCH_SIZE,
   shuffle=False
)
# ==========================================
# Model
# ==========================================
class PolarityModel(nn.Module):
   def __init__(self):
       super().__init__()
       self.encoder = AutoModel.from_pretrained(MODEL_NAME)
       self.dropout = nn.Dropout(0.1)
       self.classifier = nn.Linear(
           self.encoder.config.hidden_size, 3
       )
   def forward(self, input_ids, attention_mask):
       out = self.encoder(
           input_ids=input_ids,
           attention_mask=attention_mask
       )
       cls = out.last_hidden_state[:, 0]
       return self.classifier(self.dropout(cls))
# ==========================================
# Loss & Optimizer
# ==========================================
criterion = nn.CrossEntropyLoss()
model = PolarityModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# ==========================================
# Validation
# ==========================================
@torch.no_grad()
def evaluate(model, dataloader):
   model.eval()
   all_preds, all_labels = [], []
   for batch in tqdm(dataloader, desc="Validation", leave=False):
       input_ids = batch["input_ids"].to(DEVICE)
       attn = batch["attention_mask"].to(DEVICE)
       labels = batch["label"].to(DEVICE)
       logits = model(input_ids, attn)
       preds = torch.argmax(logits, dim=1)
       all_preds.extend(preds.cpu().numpy())
       all_labels.extend(labels.cpu().numpy())
   return f1_score(all_labels, all_preds, average="macro")
# ==========================================
# Training Loop
# ==========================================
best_val_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
   model.train()
   total_loss = 0.0
   for batch in tqdm(train_loader, desc=f"Epoch {epoch} | Train"):
       input_ids = batch["input_ids"].to(DEVICE)
       attn = batch["attention_mask"].to(DEVICE)
       labels = batch["label"].to(DEVICE)
       logits = model(input_ids, attn)
       loss = criterion(logits, labels)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       total_loss += loss.item()
   avg_loss = total_loss / len(train_loader)
   val_f1 = evaluate(model, val_loader)
   print(
       f"Epoch {epoch} | "
       f"Train Loss: {avg_loss:.4f} | "
       f"Val Macro-F1: {val_f1:.4f}"
   )
   # Save checkpoint
   torch.save(
       {
           "epoch": epoch,
           "model_state": model.state_dict(),
           "optimizer_state": optimizer.state_dict(),
           "val_f1": val_f1
       },
       os.path.join(CHECKPOINT_DIR, f"polarity_epoch_{epoch}.pt")
   )
   if val_f1 > best_val_f1:
       best_val_f1 = val_f1
       torch.save(
           model.state_dict(),
           os.path.join(CHECKPOINT_DIR, "polarity_best.pt")
       )
       print(f"Best model updated (Val F1 = {val_f1:.4f})")
