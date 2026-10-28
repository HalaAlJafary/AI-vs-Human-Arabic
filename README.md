# Arabic AI vs. Human Text Detection

> Classifying Arabic text as **AI-generated** or **human-written** using classical machine learning, neural networks, and a fine-tuned AraBERT transformer.

**Course:** Natural Language Processing — Hashemite University

**Team:** Sara Alkhraishah · Hala AlJafary · Malak Abu Haweleh · Aseel Alshishani

## Overview

With the rise of large language models, telling machine-generated text from human writing has become an important problem, and Arabic adds extra difficulty: rich morphology, diacritics, and inconsistent encodings. This project builds a full pipeline for Arabic AI-text detection, from merging noisy multi-source CSV files to comparing many models on the same stratified test split.

The dataset has about 1,545 cleaned samples. Labels are binary: `1` = AI-generated, `0` = human-written.

## Pipeline

| Part | Stage | Details |
|------|-------|---------|
| 1 | Data merging | Merge CSVs from Google Drive, encoding fallback (`utf-8-sig → cp1256 → utf-8 → iso-8859-1`), filter corrupted / non-Arabic rows, drop duplicates |
| 2 | Preprocessing | Keyword-based label encoding, diacritics removal, Alef/Yaa/Taa normalisation, non-Arabic and digit removal, stopword removal (NLTK) |
| 3 | Representations | Bag-of-Words (3,000), TF-IDF (3,000), averaged Word2Vec / FastText |
| 4 | Classical ML | Naive Bayes, SVM (RBF), Decision Tree, Random Forest, AdaBoost × 3 representations |
| 5 | FNN | Dense 128 → Dense 64 → Softmax(2), on TF-IDF and Word2Vec |
| 6 | BiLSTM | Embedding → BiLSTM(64) → Dense 32 → Softmax(2) |
| 7 | SVM tuning | `GridSearchCV` over `C` and `gamma` on TF-IDF |
| 8 | AraBERT | Fine-tuning `aubmindlab/bert-base-arabertv02` (3 epochs) |
| 9 | Comparison | Final accuracy table across all models |

All models use a stratified 80/20 split with `random_state=42`.

## Results (test set)

| Model | Features | Accuracy |
|-------|----------|----------|
| Tuned SVM (C=10, gamma=scale) | TF-IDF | **0.8835** |
| AraBERT (fine-tuned) | AraBERT tokenizer | 0.8835 |
| SVM (RBF) baseline | TF-IDF | 0.8803 |
| BiLSTM | Tokenized sequences | 0.8641 |
| FNN | TF-IDF | 0.8608 |
| Random Forest | BoW | 0.8544 |
| Naive Bayes | TF-IDF | 0.8544 |

Key takeaways: TF-IDF beats BoW and Word2Vec; a tuned SVM matches AraBERT with far less compute; neural models overfit quickly on such a small dataset; Word2Vec trained from scratch underperforms. See the full report for per-class metrics and the discussion of Arabic-specific challenges.

## Dataset

The dataset (`AI-vs-HUMAN.csv`) was built by merging several Arabic CSV files from different sources into one cleaned corpus.

| Property | Value |
|----------|-------|
| Language | Arabic |
| Task | Binary text classification |
| Size | ~1,545 samples after cleaning (test split: 309) |
| Columns | `text` (Arabic text sample), `label` (class) |
| Labels | `1` = AI-generated, `0` = human-written |
| Encoding | UTF-8 with BOM (`utf-8-sig`) |

**Cleaning applied while building it:**
- Files read with an encoding fallback to fix garbled Arabic text.
- Rows removed if empty, made only of `?` characters, or containing no Arabic letters.
- Free-form labels (e.g. `AI`, `generated`, `human`, `written`) mapped to `0` / `1`; unrecognised labels dropped.
- Exact duplicate texts removed.

The preprocessed version (`Preprocessed_NLP_Dataset.csv`) adds two columns: `tokens` (token list) and `cleaned_text` (normalised text without stopwords, used as model input).

## Repository contents

```
├── Arabic_AI_vs_Human_Text_Detection.ipynb   # Full pipeline (Google Colab notebook)
├── AI-vs-HUMAN.csv          # Cleaned merged dataset
├── NLP_Full_Report_Final.docx                # Detailed technical report
└── README.md
```

## Getting started

The notebook is written for **Google Colab** with Google Drive.

1. Upload the CSV files (each with `text` and `label` columns) to `MyDrive/Colab Notebooks/AI vs. Human Text/`.
2. Open the notebook in Colab (a GPU runtime is recommended for Parts 5, 6 and 8).
3. Run the cells in order. Intermediate files are saved to Drive:
   - `AI-vs-HUMAN.csv`
   - `Preprocessed_NLP_Dataset.csv`

### Main dependencies

`pandas` · `numpy` · `nltk` · `scikit-learn` · `gensim` · `tensorflow` · `torch` · `transformers` · `datasets` · `accelerate` · `evaluate`

(The notebook installs `gensim`, `transformers`, `datasets`, `accelerate` and `evaluate` with `pip` where needed.)

## Notes

- Paths are hard-coded for Google Drive; edit them if you run locally.
- Part 3 first tries the pre-trained `fasttext-wiki-news-subwords-300` embeddings (English) and falls back to a Word2Vec model trained on the dataset.
