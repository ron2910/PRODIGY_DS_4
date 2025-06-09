import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

nltk.download('vader_lexicon')

# Load your dataset
df = pd.read_csv("Entity_Sentiment.csv")  # Update path if needed

# Rename columns to something useful
df.columns = ['id', 'entity', 'sentiment', 'tweet']

# Clean the tweet text
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)  # Remove URLs, @mentions, hashtags
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)     # Remove punctuation/symbols
    return text.lower()

df["cleaned"] = df["tweet"].astype(str).apply(clean_text)

# Sentiment scoring
sia = SentimentIntensityAnalyzer()
df["compound"] = df["cleaned"].apply(lambda x: sia.polarity_scores(x)["compound"])

# Label sentiment based on compound score
df["sentiment_label"] = df["compound"].apply(
    lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
)

# Plot sentiment distribution
sns.countplot(x="sentiment_label", data=df, palette="coolwarm")
plt.title("Sentiment Distribution of Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.show()
print(df.columns)
