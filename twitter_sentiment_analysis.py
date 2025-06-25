import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
import os
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))


# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Load CSV files
train = pd.read_csv("dataset/twitter_training.csv", header=None)
val = pd.read_csv("dataset/twitter_validation.csv", header=None)

# Assign column names
cols = ['ID', 'Topic', 'Sentiment', 'Tweet']
train.columns = cols
val.columns = cols

# Combine datasets
df = pd.concat([train, val], ignore_index=True)

# Drop rows with missing tweets
df.dropna(subset=['Tweet'], inplace=True)

# Standardize sentiment labels
df['Sentiment'] = df['Sentiment'].str.strip().str.lower().str.capitalize()

# Filter only expected sentiment classes
expected_sentiments = ['Positive', 'Negative', 'Neutral']
df = df[df['Sentiment'].isin(expected_sentiments)]

# Preview dataset
print(df['Sentiment'].value_counts())
print("Unique topics:", df['Topic'].nunique())

palette = {
    'Negative': '#FF0000',  # Red
    'Neutral': '#0000FF',   # Blue
    'Positive': '#00AA00'   # Green
}


# Overall Sentiment Distribution with percentage labels
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Sentiment',hue='Sentiment', order=df['Sentiment'].value_counts().index, palette=palette)
total = len(df)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{100 * height / total:.1f}%', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=9)
plt.title('Overall Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Tweet Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/overall_sentiment_distribution.png",dpi=300,bbox_inches='tight')
plt.close()

# Sentiment Distribution by Top 10 Topics
top_topics = df['Topic'].value_counts().nlargest(10).index
top_df = df[df['Topic'].isin(top_topics)]
topic_order = df['Topic'].value_counts().loc[top_topics].index

plt.figure(figsize=(12, 6))
sns.countplot(data=top_df, x='Topic', hue='Sentiment', order=topic_order, palette=palette)
plt.title('Sentiment Distribution by Top 10 Topics')
plt.xlabel('Topic / Brand')
plt.ylabel('Tweet Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig("images/sentiment_by_top_topics.png",dpi=300,bbox_inches='tight')
plt.close()

# Clean text function for WordClouds
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+", "", str(text)) 
    text = re.sub(r"[^\w\s]", "", text) 
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() 
    return " ".join([word for word in text.split() if word not in stop_words])

# Clean all tweets
df['Cleaned_Tweet'] = df['Tweet'].apply(clean_text)

# Generate WordCloud for each sentiment
for sentiment in expected_sentiments:
    print(f"Generating word cloud for: {sentiment}")
    text = " ".join(df[df['Sentiment'] == sentiment]['Cleaned_Tweet'])
    wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='Set2').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {sentiment} Tweets")
    plt.savefig(f"images/wordcloud_{sentiment.lower()}.png",dpi=300,bbox_inches='tight')
    plt.close()
