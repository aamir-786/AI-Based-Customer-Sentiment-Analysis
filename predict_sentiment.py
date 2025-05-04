import pandas as pd
import re
import joblib
from nltk.corpus import stopwords

# Load model and preprocessing tools
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

stop_words = set(stopwords.words('english'))

# Clean function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load new review dataset
new_data = pd.read_csv("your_new_dataset.csv")  # 🔁 Replace with your dataset
new_data['Review_Header'] = new_data['Review_Header'].fillna('')
new_data['Review_text'] = new_data['Review_text'].fillna('')
new_data['Full_Review'] = new_data['Review_Header'] + " " + new_data['Review_text']
new_data['Cleaned_Review'] = new_data['Full_Review'].apply(clean_text)

# Predict sentiments
X_new = vectorizer.transform(new_data['Cleaned_Review'])
predictions = model.predict(X_new)
predicted_labels = label_encoder.inverse_transform(predictions)

# Print predictions
print("\n✅ Sentiment Predictions:\n")
for i, sentiment in enumerate(predicted_labels):
    print(f"Review {i+1}: {sentiment}")
