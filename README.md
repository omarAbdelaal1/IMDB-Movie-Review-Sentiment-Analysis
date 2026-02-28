<div align="center">

# 📝 IMDB Movie Review Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-green.svg)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Natural Language Processing (NLP) machine learning pipeline designed to automatically classify IMDB movie reviews as positive or negative using TF-IDF and Logistic Regression.**

</div>

---

## 📑 Table of Contents
- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Methodology](#-methodology)
- [Results & Performance](#-results--performance)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Contact](#-contact)

---

## 📖 About the Project

This project focuses on **Sentiment Analysis**, a fundamental subfield of Natural Language Processing (NLP). Utilizing the famous IMDB Dataset of 50k Movie Reviews, the pipeline cleans and transforms unstructured textual data into actionable numerical vectors for binary classification.

The primary objective is to demonstrate robust text preprocessing techniques paired with a reliable, interpretable algorithm (Logistic Regression) to automate the categorization of textual feedback as either `positive` or `negative`.

All code and analytical steps are seamlessly compiled within a single Jupyter Notebook (`NLP.ipynb`).

---

## ✨ Key Features

- **Robust Text Preprocessing**: Eliminates noise via lowercase conversion and regex filtering (removing non-alphabetic characters).
- **Linguistic Normalization**: Applies `nltk` tools for word tokenization, Stop Word removal, and Lemmatization to reduce words to their base dictionary forms.
- **Advanced Text Vectorization**: Transforms the refined text into numerical features using `TfidfVectorizer` mapped to the top 5,000 most semantically meaningful words.
- **Interpretable Modeling**: Implements a highly efficient `LogisticRegression` classifier, guaranteeing fast training times and transparent probability scoring.
- **Modular Prediction Function**: Includes an easy-to-use custom function (`predict_sentiment(text)`) allowing users to perform live, on-the-fly sentiment classification on any string.

---

## 🛠️ Tech Stack

- **Data Manipulation**: `pandas`, `numpy`
- **Natural Language Processing**: `nltk` (WordNetLemmatizer, stopwords, word_tokenize)
- **Feature Extraction & Modeling**: `scikit-learn` (TfidfVectorizer, LogisticRegression, `train_test_split`)
- **Regular Expressions**: `re`

---

## 🔬 Methodology

1. **Data Ingestion & Cleaning**: Loads the `IMDB Dataset.csv` (contains 25,000 Positive / 25,000 Negative label splits) and securely drops null rows.
2. **Text Normalization Pipeline**: 
   - Non-alphabetic character removal via built in RegEx.
   - Stop words exclusion filter (drops unhelpful words like "the", "is", "a").
   - Lemmatization mapping word variations (e.g. "running" → "run").
3. **Train-Test Splitting**: Enforces a stratified **80/20 data split**, ensuring balanced positive/negative distributions in both test and training models.
4. **TF-IDF Vectors**: Computes Term Frequency-Inverse Document Frequency metrics to penalize excessively common terms while boosting highly descriptive unique words.
5. **Model Optimization**: Integrates the feature rich vectorized text to a binary `LogisticRegression` algorithm capped gracefully at `max_iter=200`.

---

## 📈 Results & Performance

The carefully tuned logistic pipeline consistently produces outstanding classification metrics on a holdout test sample of 10,000 varying reviews:

- **Overall Accuracy**: ~89.0%
- **Precision/Recall**: Extremely balanced across negative (0.90/0.88) and positive (0.88/0.90) polarities, underscoring systemic reliability and absence of algorithmic bias.
- **Macro F1 Score**: ~0.89

> **Confusion Matrix Details:** The model proves its accuracy via generation of minimal false positives (601) against minimal false negatives (500).

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed and the fundamental NLP libraries:

```bash
git clone https://github.com/omarAbdelaal1/IMDB-Movie-Review-Sentiment-Analysis
cd IMDB-Movie-Review-Sentiment-Analysis
pip install pandas numpy scikit-learn nltk matplotlib
```

*(Note: The internal architecture will securely initiate an automated NLTK payload downloading required internal corpora (`punkt`, `wordnet`, `stopwords`) on initial run.)*

### Running the Pipeline

1. **Obtain Data**: Verify that your `IMDB Dataset.csv` sits in the project's active core directory.
2. **Launch Jupyter System**:
   ```bash
   jupyter notebook
   ```
3. **Execute Application**: Launch the `NLP.ipynb` instance and opt to `Run All` sequential operational blocks.
4. **Live Execution Querying**: Navigate directly toward the bottom sequence variable block: `predict_sentiment("This movie was fantastic!")` to inject tailored textual testing sequences against the loaded algorithm pipeline!

---

## 📂 Project Structure

```text
├── NLP.ipynb             # Core notebook housing analytical ingestion, vectorization processes & logistic models
├── README.md             # General project infrastructure documents
└── IMDB Dataset.csv      # IMDB movie dataset structure file (To be acquired/sourced independently locally)
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📬 Contact

**Omar Abdelaal** - [oabdelall2004@gmail.com](mailto:oabdelall2004@gmail.com)

Project Link: [https://github.com/omarAbdelaal1/IMDB-Movie-Review-Sentiment-Analysis](https://github.com/omarAbdelaal1/IMDB-Movie-Review-Sentiment-Analysis)
