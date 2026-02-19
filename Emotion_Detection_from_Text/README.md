
# 🧠 Multi-Class Sentiment Analysis (NLP Pipeline)

Production-style NLP project that classifies social media text into **anger, fear, joy, and sadness** using TF-IDF feature engineering and Logistic Regression.

This project demonstrates a full machine learning workflow: data cleaning → feature engineering → model training → evaluation → export for BI tools.

---

## 🚀 Project Highlights

* Built an end-to-end NLP pipeline for multi-class classification
* Designed robust text preprocessing for noisy social media data
* Engineered TF-IDF features (5,000 max features)
* Achieved **87% accuracy** on unseen test data
* Exported predictions for dashboard integration (Tableau / Power BI ready)

---

## 📂 Dataset Overview

**Input Columns:**

* `ID`
* `content` (raw social media text)
* `sentiment` (target label)

**Classes:**

* Anger
* Fear
* Joy
* Sadness

The dataset is moderately imbalanced, with **fear** being the most frequent class.

---

## 🔎 NLP Preprocessing Pipeline

To handle noisy social media text, the following steps were implemented:

* Lowercasing
* URL & domain removal
* Punctuation removal
* Repeated character normalization (e.g., `coooool → cool`)
* Non-alphabetic filtering
* Whitespace cleanup
* Tokenization (NLTK)
* Stopword removal
* Lemmatization (WordNet)

This improves signal quality and reduces feature sparsity before vectorization.

---

## 📊 Exploratory Data Analysis

Performed EDA to understand linguistic patterns across sentiment classes:

* Sentiment distribution visualization
* Word count per sentiment (boxplot)
* Letter count per sentiment (boxplot)

### Key Observations

* Fear-related posts tend to use slightly more words on average.
* Joy posts are generally shorter and more concise.
* Anger and sadness show higher variability in length.
* Word count and letter count trends are strongly correlated.

---

## 🔢 Feature Engineering

Used:

```python
TfidfVectorizer(max_features=5000)
```

### Why TF-IDF?

* Captures term importance relative to corpus
* Works efficiently with linear models
* Interpretable feature space
* Computationally lightweight

Limitation: does not capture contextual meaning (same word → same vector regardless of context).

---

## 🤖 Model Development

**Model:** Logistic Regression
**Train/Test Split:** 80/20
**Max Iterations:** 1000 (to ensure convergence)

---

## 📈 Model Performance

**Accuracy: 87%**

| Class   | Precision | Recall | F1-score |
| ------- | --------- | ------ | -------- |
| Anger   | 0.92      | 0.83   | 0.88     |
| Fear    | 0.79      | 0.94   | 0.86     |
| Joy     | 0.95      | 0.89   | 0.92     |
| Sadness | 0.87      | 0.79   | 0.83     |

### Performance Analysis

* Strong macro F1-score (~0.87)
* High recall for **fear** (94%) → model captures most fear instances
* Slight over-prediction of fear (lower precision)
* Best balanced performance observed in **joy**

The results show that traditional ML with strong preprocessing can still deliver competitive performance without deep learning.

---

## 🧪 Inference Pipeline

Custom prediction function:

```python
def predict_sentiment(text):
    processed_content = preprocess_text(text)
    vector = tfidf.transform([processed_content]).toarray()
    prediction = model.predict(vector)[0]
    return prediction
```

Example:

```python
predict_sentiment("I am so scared about this project!")
```

Output:

```
fear
```

---

## 📤 Business Integration

Predictions are appended to the dataset and exported as:

```
sentiment_predictions.csv
```

This enables:

* Dashboard visualization (Tableau / Power BI)
* Downstream analytics
* Reporting workflows

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib

---

## 📌 Skills Demonstrated

* NLP preprocessing for unstructured text
* Feature engineering with TF-IDF
* Multi-class classification
* Model evaluation & interpretation
* Handling class imbalance insights
* Pipeline-style inference design
* Exporting ML outputs for analytics teams

---

## 🔮 Future Improvements

* Transformer-based models (BERT / DistilBERT) for contextual embeddings
* FastText for subword modeling on informal text
* Hyperparameter tuning with cross-validation
* Class weighting for imbalance mitigation
* Confusion matrix & error analysis deep dive
* REST API deployment (FastAPI)

---

## 🧠 Takeaway

This project demonstrates how well-engineered preprocessing combined with classical machine learning can achieve strong real-world performance while remaining computationally efficient and interpretable.

It highlights the trade-off between:

* Lightweight, explainable models (TF-IDF + Logistic Regression)
  vs
* Heavy contextual models (Transformers)

---

