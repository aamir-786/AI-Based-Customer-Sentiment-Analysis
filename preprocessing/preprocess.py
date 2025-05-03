import pandas as pd
import nltk
print(nltk.__version__)  # Should display the installed version
import nltk
nltk.download('popular')  # Downloads commonly used packages
nltk.download('punkt_tab')        # For tokenizing text
nltk.download('stopwords')    # For removing common words like "is", "the", etc.
nltk.download('wordnet')      # For lemmatizing words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string   

# Load data
df = pd.read_csv('data/Amazon Review Data Web Scrapping - Amazon Review Data Web Scrapping.csv')

# Combine Review_Header and Review_Text into one column
df['Combined_Review'] = df['Review_Header'].fillna('') + ' ' + df['Review_text'].fillna('')

# Set up stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a function to clean the text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

print(df['Combined_Review'].head())

# Apply the cleaning function to each review
df['Cleaned_Review'] = df['Combined_Review'].apply(clean_text)

# Check unique values in the 'Own_Rating' column
print(df['Rating'].unique())

# Map ratings to sentiment categories
def encode_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

# Apply the encoding function to the 'Own_Rating' column
df['Sentiment'] = df['Rating'].apply(encode_sentiment)

# Check the result
print(df[['Rating', 'Sentiment']].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the cleaned reviews
X_tfidf = tfidf_vectorizer.fit_transform(df['Cleaned_Review'])

# Convert the result into a DataFrame for easier inspection
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Check the shape of the resulting DataFrame
print(X_tfidf_df.shape)

from sklearn.model_selection import train_test_split

# Define the features (X) and target (y)
X = X_tfidf
y = df['Sentiment']

# Split the dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting sets
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Convert the TF-IDF data into a DataFrame for easy sharing
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Save the features (X_tfidf) and target (y) to CSV
X_tfidf_df.to_csv('preprocessed_tfidf_features.csv', index=False)
y.to_csv('preprocessed_sentiment_labels.csv', index=False)

# Confirm the files are saved
print("Data saved successfully!")
