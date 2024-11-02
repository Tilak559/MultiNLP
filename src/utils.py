import re

# Function to clean and standardize text data
def preprocess_text(text):
    # Convert text to lowercase for consistency
    text = text.lower()
    
    # Remove punctuation and special characters, leaving only words and spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

# Function to create sentiment labels based on star ratings
def create_sentiment_label(stars):
    # Define positive sentiment for ratings of 4 or higher
    if stars >= 4:
        return "positive"
    # Define neutral sentiment for a rating of 3
    elif stars == 3:
        return "neutral"
    # Define negative sentiment for ratings below 3
    else:
        return "negative"
