import torch
from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ.get("HF_TOKEN")
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from train_val_split import train_ds, val_ds

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
def tokenization(batch):
    #tokenize features
    token = tokenizer(batch["comment_text"], truncation=True)
    token["labels"] = [[float(batch[col][i]) for col in label_cols]
                       for i in range(len(batch["comment_text"]))
                       ]
    return token

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
unwanted_cols = label_cols + ["id", "comment_text"]

train_ds = train_ds.map(tokenization, batched=True, remove_columns=unwanted_cols)
train_ds.set_format(type="torch")
val_ds = val_ds.map(tokenization, batched=True, remove_columns=unwanted_cols)
val_ds.set_format(type="torch")

"""
training_params = TrainingArguments("tcm_trainer",logging_strategy="steps", eval_strategy="steps")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)

class WeightedClassTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fn = nn.CrossEntropyLoss(self.class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss





trainer = Trainer(
    model,
    training_params,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer
)
"""
