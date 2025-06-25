
# 💬 Twitter Sentiment Analysis & Visualization

This project performs exploratory sentiment analysis on a combined Twitter dataset (`training` + `validation`). It uses Python to analyze public sentiment across various topics, visualizes distribution patterns, and generates word clouds per sentiment class.

---

## 📁 Dataset

- `twitter_training.csv` – labeled training data (ID, Topic, Sentiment, Tweet)
- `twitter_validation.csv` – labeled validation data
- Source: [Sentiment Analysis on Twitter Dataset (via Kaggle or provided)]

---

## 📊 Visualizations

Generated charts and word clouds are saved under the `images/` directory:

| File | Description |
|------|-------------|
| `overall_sentiment_distribution.png` | Distribution of positive, negative, and neutral tweets |
| `sentiment_by_top_topics.png` | Sentiment breakdown for top 10 most discussed topics |
| `wordcloud_positive.png` | Most frequent positive sentiment words |
| `wordcloud_negative.png` | Most frequent negative sentiment words |
| `wordcloud_neutral.png`  | Most frequent neutral sentiment words |

---

## ⚙️ Output Columns

After cleaning:
- Standardized sentiments
- Removed noise (URLs, mentions, punctuation)
- Added `Cleaned_Tweet` column

---

## 🚀 How to Run

### Clone the Repository

```bash
git clone https://github.com/SouvikTikader/twitter_sentiment_analysis.git
cd twitter_sentiment_analysis
```

### Install Required Packages

```bash
pip install pandas matplotlib seaborn wordcloud nltk
```

### Run the Script

```bash
python twitter_sentiment_analysis.py
```

The script will:

* Clean and merge the training + validation datasets
* Visualize sentiment trends
* Generate sentiment-specific word clouds
* Save plots to `images/` directory

---

## 📦 Directory Structure

```
.
├── dataset/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── images/
│   ├── *.png (output plots)
├── twitter_sentiment_analysis.py
└── README.md
```

---

## 📧 Author

**Souvik Tikader**
GitHub: [@SouvikTikader](https://github.com/SouvikTikader)
