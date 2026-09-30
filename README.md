# IMDB Movie Review Sentiment Analysis

## Dataset
Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Expected CSV columns are `review` and `sentiment` (`positive`/`negative`).

## Files
- `imdb_sentiment_training.ipynb`: complete NLP training, EDA, cleaning, TF-IDF, model comparison, validation/test, saving and download.
- `cli.py`: command-line prediction prototype.
- `one_page_summary.txt`: one-page project summary with spaces to fill in measured results.
- `README.md`: this guide.

The `.pkl` model is intentionally **not included**. Generate and download it by running the notebook.

## Setup
Recommended: Python 3.10+. Install dependencies:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

## Dataset setup
1. Download `IMDB Dataset.csv` from the Kaggle page above (Kaggle sign-in/terms may apply).
2. Place it beside the notebook, or upload it when prompted.
3. Open the notebook in Jupyter or Google Colab and run cells from top to bottom.

## Training workflow
The notebook checks structure, missing values, blank reviews, duplicates, label consistency, class distribution and review lengths. It normalizes HTML and whitespace, removes unusable rows and duplicate review/label pairs, then uses a stratified 70% train / 15% validation / 15% test split.

It compares Multinomial Naive Bayes, Logistic Regression and Linear SVM with TF-IDF unigrams/bigrams. Validation macro-F1 selects the model; the held-out test set is used for final metrics. It saves `imdb_sentiment_model.pkl`, which includes both the fitted vectorizer and classifier. In Google Colab, the notebook triggers a download; in local Jupyter, use the file browser to download the saved file.

## Run CLI
Copy `imdb_sentiment_model.pkl` into the same directory as `cli.py`.

Single review:

```bash
python cli.py "This movie was wonderfully acted and moving."
```

Interactive mode:

```bash
python cli.py
```

Type a review and press Enter. Type `quit` or `exit` to stop.

Use a model at another path:

```bash
python cli.py --model /path/to/imdb_sentiment_model.pkl "I disliked this film."
```

## Notes and limitations
- Use the metrics printed by your own notebook run; scores depend on data and software versions.
- The CLI and notebook share the same text normalization.
- Linear SVM decision scores are not probabilities.
- Only load `.pkl`/joblib files from sources you trust because deserialization can execute code.
- The model is designed for movie reviews and may not generalize to other domains. Do not use for high-stakes decisions.
