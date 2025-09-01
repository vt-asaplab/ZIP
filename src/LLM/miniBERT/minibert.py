import os
import json
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, logging

logging.set_verbosity_error()
torch.set_default_dtype(torch.float64)
optim.AdamW = partial(optim.AdamW, foreach=False)

BATCH_SIZE = 32
EPOCHS     = 2
LR         = 2e-5
MAX_LEN    = 64
MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "bert_mini_sst2_finetuned"

print("Downloading SST‑2 …")
raw_ds = load_dataset("glue", "sst2")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
ds = raw_ds.map(tokenize, batched=True)
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
train_loader = DataLoader(ds["train"],      batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(ds["validation"], batch_size=BATCH_SIZE)

class BertMiniSST2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            MODEL_NAME, add_pooling_layer=False
        ).to(torch.double)
        self.classifier = nn.Linear(256, 2, dtype=torch.double)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)

model     = BertMiniSST2().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

def train_epoch(ep):
    model.train()
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss   = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == len(train_loader):
            print(f"Epoch {ep} | Step {step}/{len(train_loader)} | Loss {loss.item():.4f}")

def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds  = logits.argmax(dim=1)
            correct += (preds == batch["label"]).sum().item()
            total   += preds.size(0)
    print(f"Validation accuracy: {100*correct/total:.2f}%")

for ep in range(1, EPOCHS+1):
    train_epoch(ep)
    evaluate()

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.bert.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

param_dict = {
    name: param.detach().cpu().tolist()
    for name, param in model.named_parameters()
}
with open("bert_mini_params.json", "w") as f:
    json.dump(param_dict, f, indent=2)
