import torch
from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ.get("HF_TOKEN")
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from train_val_split import train_ds, val_ds
from eda import pos_weight_vals
from datasets import Dataset
import evaluate
import numpy as np

pos_weight_tensor = torch.Tensor(pos_weight_vals)
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

train_ds_test = Dataset.from_dict(train_ds[:1024])
val_ds_test = Dataset.from_dict(val_ds[:128])


training_params = TrainingArguments("tcm_trainer",
                                    logging_strategy="steps",
                                    eval_strategy="steps",
                                    dataloader_pin_memory=False,
                                    num_train_epochs=5,
                                    per_device_train_batch_size=32,
                                    per_device_eval_batch_size=32,
                                    )
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)

class WeightedClassTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fn = nn.BCEWithLogitsLoss(self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    f1_metric = evaluate.load("f1", "multilabel")
    roc_metric = evaluate.load("roc_auc", "multilabel")
    logits, labels = eval_pred
    probs = 1/(1+np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    f1_macro    = f1_metric.compute(predictions=preds,  references=labels, average="macro")
    f1_micro    = f1_metric.compute(predictions=preds,  references=labels, average="micro")
    f1_weighted = f1_metric.compute(predictions=preds,  references=labels, average="weighted")
    roc_auc     = roc_metric.compute(prediction_scores=probs, references=labels, average="macro")

    f1_per_label = f1_metric.compute(predictions=preds, references=labels, average=None)

    return {
        "f1_macro":         f1_macro["f1"],
        "f1_micro":         f1_micro["f1"],
        "f1_weighted":      f1_weighted["f1"],
        "roc_auc_macro":    roc_auc["roc_auc"],
        "f1_toxic":         f1_per_label["f1"][0],
        "f1_severe_toxic":  f1_per_label["f1"][1],
        "f1_obscene":       f1_per_label["f1"][2],
        "f1_threat":        f1_per_label["f1"][3],
        "f1_insult":        f1_per_label["f1"][4],
        "f1_identity_hate": f1_per_label["f1"][5],
    }

trainer = WeightedClassTrainer(
    class_weights=pos_weight_tensor,
    model=model,
    args=training_params,
    train_dataset=train_ds_test,
    eval_dataset=val_ds_test,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

print(torch.cuda.is_available())          # True if GPU is detected
print(torch.cuda.get_device_name(0))      # e.g. "Tesla T4"
print(f"Model device: {next(model.parameters()).device}")
trainer.train()
predictions = trainer.predict(val_ds_test)
print(predictions)
