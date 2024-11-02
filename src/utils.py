import re

def preprocess_text(text):
    """
    Lowercase text and remove punctuation for basic text preprocessing.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def create_sentiment_label(stars):

    if stars >= 4:
        return "positive"
    elif stars == 3:
        return "neutral"
    else:
        return "negative"
