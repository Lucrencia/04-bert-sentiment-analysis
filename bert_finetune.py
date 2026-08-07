"""
bert_finetune.py — Fine-tune DistilBERT for IMDB sentiment classification.

This script requires internet access (to download pretrained weights from
Hugging Face) and is best run on a GPU (Colab, Kaggle, or a local/cloud GPU).
It is not executed inside this notebook because the review environment used
to build this portfolio has no internet access to huggingface.co — the
TF-IDF baseline earlier in this notebook was trained and evaluated locally
on the same data split for a fair, fully-reproducible comparison.
"""
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer)
from datasets import Dataset

MODEL_NAME = "distilbert-base-uncased"

df = pd.read_csv("data/imdb_reviews.csv")
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["review"], padding="max_length", truncation=True, max_length=256)


train_ds = Dataset.from_pandas(train_df[["review", "label"]]).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df[["review", "label"]]).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}


args = TrainingArguments(
    output_dir="./bert_ckpt",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)

preds = trainer.predict(test_ds)
y_pred = np.argmax(preds.predictions, axis=1)
print(classification_report(test_df["label"], y_pred, target_names=["negative", "positive"]))

trainer.save_model("./bert_ckpt/final")
tokenizer.save_pretrained("./bert_ckpt/final")
