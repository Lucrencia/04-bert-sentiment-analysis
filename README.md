# Sentiment Analysis: Classical Baseline + Transformer Fine-Tuning

Classifying movie reviews as positive/negative, comparing a fast classical baseline against a transformer fine-tuning approach — and being explicit about the cost/accuracy tradeoff between them, which is the actual decision a team has to make in production.

## Problem
Build a sentiment classifier and make an honest case for which approach to ship: a cheap, interpretable, CPU-friendly model, or a more accurate but heavier transformer.

## Dataset
[IMDB 50K Movie Reviews](https://github.com/Ankit152/IMDB-sentiment-analysis) (balanced positive/negative). A 2,500-review balanced subset is used here for fast iteration; the same code runs unmodified on the full 50K set.

## Approach
- Cleaned HTML artifacts and punctuation from raw review text
- Built a **TF-IDF (1-2 grams) + Logistic Regression** baseline, tuned via grid search — fully executed and evaluated in `notebook.ipynb`
- Wrote a complete **DistilBERT fine-tuning script** (`bert_finetune.py`) using Hugging Face `transformers`/`datasets`/`Trainer` — this requires downloading pretrained weights and a GPU for practical training time, so it's included as a standalone, ready-to-run script rather than executed inline (see note in the notebook)

## Results
- **TF-IDF + Logistic Regression: 0.928 ROC-AUC, 85% accuracy/F1** on held-out test data
- Most predictive words are intuitive and interpretable (e.g. "waste", "worst" vs. "excellent", "best") — useful for stakeholder trust and debugging
- DistilBERT fine-tuning (`bert_finetune.py`) typically adds ~3-6 points of F1 over this baseline on similar tasks, at substantially higher training/serving cost

## Tech Stack
`pandas` · `scikit-learn` (TF-IDF baseline) · `transformers` / `datasets` / `torch` (transformer script) · `matplotlib` / `seaborn`

## Files
- `notebook.ipynb` — EDA, baseline model, full evaluation, and the transformer code walkthrough
- `bert_finetune.py` — standalone DistilBERT fine-tuning script (needs internet + ideally a GPU)
- `data/imdb_reviews.csv` — balanced 2,500-review working subset (the full 50K IMDB CSV was omitted here for repo size, but the notebook code works unmodified on the full dataset)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb          # baseline, fully offline
python bert_finetune.py                  # optional, needs internet + GPU recommended
```
