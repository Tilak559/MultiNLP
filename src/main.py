from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from model import MultiTaskModel
from utils import preprocess_text, create_sentiment_label
from torch import tensor
from sklearn.utils.class_weight import compute_class_weight
import logging

# Configure logging to track progress
logging.basicConfig(filename='training_log.txt', level=logging.INFO)

# Define a mapping for broad categories
category_mapping = {
    'Restaurants': ['Pizza', 'Mexican', 'Chinese', 'Italian', 'Restaurants'],
    'Beauty & Spas': ['Beauty & Spas', 'Nail Salons', 'Hair Salons'],
    'Food & Beverage': ['Coffee & Tea', 'Grocery', 'Ice Cream & Frozen Yogurt', 'Food'],
    'Automotive': ['Automotive', 'Auto Repair'],
    'Pets': ['Veterinarians', 'Pets']
}

def map_to_broad_category(categories):
    """
    Maps specific categories to a broad category based on defined category_mapping.
    """
    if not isinstance(categories, str):
        return 'Other'  # Assign 'Other' if categories is None or not a string
    for broad_category, keywords in category_mapping.items():
        if any(keyword in categories for keyword in keywords):
            return broad_category
    return 'Other'  # For categories that don't fit any predefined broad category



# Load and preprocess the data with a sample size of 100,000, with progress bar
def load_and_prepare_data(review_path, business_path, sample_size=50000, chunk_size=10000):
    print("Loading data in chunks with progress...")

    # Load review data in chunks and sample with progress tracking
    review_chunks = []
    for chunk in tqdm(pd.read_json(review_path, lines=True, chunksize=chunk_size), desc="Loading Review Data"):
        review_chunks.append(chunk)
    review_data = pd.concat(review_chunks).sample(n=sample_size, random_state=42)

    # Load business data and merge with sampled review data
    print("Loading Business Data...")
    business_data = pd.read_json(business_path, lines=True)
    
    print("Merging Review and Business Data...")
    data = review_data.merge(business_data[['business_id', 'categories']], on='business_id', how='left')
    
    print("Mapping categories to broader categories...")
    data['categories'] = data['categories'].apply(map_to_broad_category)
    
    # Preprocess text and sentiment labels with progress bars
    print("Preprocessing text and sentiment labels...")
    data['cleaned_text'] = [preprocess_text(text) for text in tqdm(data['text'], desc="Processing Text")]
    data['sentiment'] = [create_sentiment_label(stars) for stars in tqdm(data['stars'], desc="Creating Sentiments")]
    
    print("Category distribution after mapping:")
    print(data['categories'].value_counts())
    
    print(f"Data loaded and sampled to {sample_size} records.")
    return data

# Custom Dataset for DataLoader
class YelpDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        # Mapping categories and sentiments to numerical values
        self.category_mapping = {category: i for i, category in enumerate(data['categories'].unique())}
        self.sentiment_mapping = {"positive": 0, "neutral": 1, "negative": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['cleaned_text']
        category = self.data.iloc[idx]['categories']
        sentiment = self.data.iloc[idx]['sentiment']
        
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        
        # Convert category and sentiment to numerical values
        category = tensor(self.category_mapping[category], dtype=torch.long)
        sentiment = tensor(self.sentiment_mapping[sentiment], dtype=torch.long)
        
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), category, sentiment

# Train function with more iterations for better observation
def train(model, dataloader, optimizer, criterion_category, criterion_sentiment, device, max_iters=60):
    model.train()
    iters = 0
    correct_category_predictions = 0
    correct_sentiment_predictions = 0
    total_samples = 0
    
    for input_ids, attention_mask, labels, sentiments in dataloader:
        if iters >= max_iters:
            break
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        labels, sentiments = labels.to(device), sentiments.to(device)
        optimizer.zero_grad()
        
        # Get model predictions
        classification_output, sentiment_output = model(input_ids, attention_mask)
        
        # Calculate loss
        category_loss = criterion_category(classification_output, labels)
        sentiment_loss = criterion_sentiment(sentiment_output, sentiments)
        loss = 0.7 * category_loss + 0.3 * sentiment_loss
        logging.info(f"Iteration {iters + 1}, Loss: {loss.item()}, Category Loss: {category_loss.item()}, Sentiment Loss: {sentiment_loss.item()}")
        
        # Get predicted classes
        predicted_category = classification_output.argmax(dim=1)
        predicted_sentiment = sentiment_output.argmax(dim=1)
        
        # Calculate accuracy
        correct_category_predictions += (predicted_category == labels).sum().item()
        correct_sentiment_predictions += (predicted_sentiment == sentiments).sum().item()
        total_samples += labels.size(0)
        
        # Calculate and log accuracy percentage
        category_accuracy = (correct_category_predictions / total_samples) * 100
        sentiment_accuracy = (correct_sentiment_predictions / total_samples) * 100
        logging.info(f"Category Accuracy: {category_accuracy:.2f}%, Sentiment Accuracy: {sentiment_accuracy:.2f}%")
        
        # Print accuracy for each iteration
        print(f"Iteration {iters + 1}, Loss: {loss.item():.4f}, Category Accuracy: {category_accuracy:.2f}%, Sentiment Accuracy: {sentiment_accuracy:.2f}%")
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        iters += 1
    
    print("Training complete for the limited number of iterations.")

# Main function
def main():
    # Load and prepare data with a sample size of 100,000
    data = load_and_prepare_data('yelp_dataset/yelp_review.json', 
                                 'yelp_dataset/yelp_business.json', 
                                 sample_size=50000)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Initialize Dataset and DataLoader with larger batch size
    train_dataset = YelpDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Dynamically set num_classes based on unique categories
    num_classes = len(train_dataset.category_mapping)
    transformer_model = AutoModel.from_pretrained("distilbert-base-uncased")
    model = MultiTaskModel(transformer_model, num_classes=num_classes)  # Adjusted num_classes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Compute class weights for categories
    class_weights = compute_class_weight('balanced', classes=np.array(list(train_dataset.category_mapping.values())), 
                                         y=train_data['categories'].map(train_dataset.category_mapping))
    category_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion_category = torch.nn.CrossEntropyLoss(weight=category_weights)
    criterion_sentiment = torch.nn.CrossEntropyLoss()

    # Separate learning rates for different parts of the model
    optimizer = torch.optim.Adam([
        {'params': model.transformer.parameters(), 'lr': 1e-5},
        {'params': model.classification_head.parameters(), 'lr': 1e-5},
        {'params': model.sentiment_head.parameters(), 'lr': 1e-5}
    ])
    
    # Train the model with more iterations for better loss observation
    train(model, train_dataloader, optimizer, criterion_category, criterion_sentiment, device, max_iters=100)


if __name__ == "__main__":
    main()
