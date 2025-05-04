# train_and_save_model.py

import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK stopwords (only the first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
data = pd.read_csv("D:/AI-Based-Customer-Sentiment-Analysis-main/AI-Based-Customer-Sentiment-Analysis-main/reviews.csv")
data['Review_Header'] = data['Review_Header'].fillna('')
data['Review_text'] = data['Review_text'].fillna('')
data['Full_Review'] = data['Review_Header'] + " " + data['Review_text']

# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data['Cleaned_Review'] = data['Full_Review'].apply(clean_text)

# Encode sentiment labels
label_encoder = LabelEncoder()
data['Sentiment_Label'] = label_encoder.fit_transform(data['Own_Rating'])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['Cleaned_Review'])
y = data['Sentiment_Label']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and preprocessing objects
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("✅ Model, vectorizer, and label encoder saved successfully.")
