import torch
from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ.get("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from train_val_split import train_ds, val_ds

checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
row = train_ds[0]
print(row)
def tokenization(example):
    return tokenizer(example["comment_text"])

print(tokenizer(row["comment_text"], padding=True, truncation=True))
train_ds = train_ds.map(tokenization, batched=True, remove_columns=["id"])
train_ds.set_format(type="torch", columns=['id', 'input_ids', 'token_type_ids', 'attention_mask', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
val_ds = val_ds.map(tokenization, batched=True, remove_columns=["id"])
val_ds.set_format(type="torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
