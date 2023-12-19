import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download("vader_lexicon")
import pandas as pd
import spacy
data = pd.read_csv("MovieReview_reddit-scraper.csv")
review = data['body'].dropna()

#Removing URLs
nlp = spacy.load("en_core_web_sm")

# Create a new column to store cleaned review
data['cleaned_review'] = ""

# Iterate through the review to remove URLs and store cleaned review
for index, row in data.iterrows():
    review_text = row['body']
    
    # Remove URLs from the review text
    doc = nlp(review_text)
    cleaned_review = ' '.join([token.text for token in doc if not token.like_url])
    
    # Store the cleaned review in the new column
    data.at[index, 'cleaned_review'] = cleaned_review


#Compound negative, positve and neutral scores.
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["cleaned_review"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["cleaned_review"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["cleaned_review"]]
data['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in data["cleaned_review"]]

#Add labels
score = data["Compound"].values
sentiment = []
for i in score:
    if i >= 0.05 :
        sentiment.append('Positive')
    elif i <= -0.05 :
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
data["Sentiment"] = sentiment

new_data = data[['cleaned_review', 'Sentiment']].copy()
new_data.columns = ['Review', 'Sentiment']

print(new_data["Sentiment"].value_counts())
new_data.to_csv("new_data.csv")