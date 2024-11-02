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

# Setup logging to record progress and key events during training
logging.basicConfig(filename='training_log.txt', level=logging.INFO)

# Define broad category mappings to group specific categories under a general label
category_mapping = {
    'Restaurants': ['Pizza', 'Mexican', 'Chinese', 'Italian', 'Restaurants'],
    'Beauty & Spas': ['Beauty & Spas', 'Nail Salons', 'Hair Salons'],
    'Food & Beverage': ['Coffee & Tea', 'Grocery', 'Ice Cream & Frozen Yogurt', 'Food'],
    'Automotive': ['Automotive', 'Auto Repair'],
    'Pets': ['Veterinarians', 'Pets']
}

# Function to map specific categories to broader categories
def map_to_broad_category(categories):
    if not isinstance(categories, str):
        return 'Other'
    for broad_category, keywords in category_mapping.items():
        if any(keyword in categories for keyword in keywords):
            return broad_category
    return 'Other'

# Load data from review and business files, preprocess, and sample it for training
def load_and_prepare_data(review_path, business_path, sample_size=50000, chunk_size=10000):
    print("Loading data in chunks with progress...")

    review_chunks = []
    for chunk in tqdm(pd.read_json(review_path, lines=True, chunksize=chunk_size), desc="Loading Review Data"):
        review_chunks.append(chunk)
    review_data = pd.concat(review_chunks).sample(n=sample_size, random_state=42)

    print("Loading Business Data...")
    business_data = pd.read_json(business_path, lines=True)
    
    print("Merging Review and Business Data...")
    data = review_data.merge(business_data[['business_id', 'categories']], on='business_id', how='left')
    
    print("Mapping categories to broader categories...")
    data['categories'] = data['categories'].apply(map_to_broad_category)
    
    print("Preprocessing text and sentiment labels...")
    data['cleaned_text'] = [preprocess_text(text) for text in tqdm(data['text'], desc="Processing Text")]
    data['sentiment'] = [create_sentiment_label(stars) for stars in tqdm(data['stars'], desc="Creating Sentiments")]
    
    print("Category distribution after mapping:")
    print(data['categories'].value_counts())
    
    print(f"Data loaded and sampled to {sample_size} records.")
    return data

# Define a custom dataset class for Yelp data, specifying how to process each sample
class YelpDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.category_mapping = {category: i for i, category in enumerate(data['categories'].unique())}
        self.sentiment_mapping = {"positive": 0, "neutral": 1, "negative": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['cleaned_text']
        category = self.data.iloc[idx]['categories']
        sentiment = self.data.iloc[idx]['sentiment']
        
        # Tokenize text input for model compatibility
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        
        category = tensor(self.category_mapping[category], dtype=torch.long)
        sentiment = tensor(self.sentiment_mapping[sentiment], dtype=torch.long)
        
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), category, sentiment

# Set up layer-wise learning rates for different model parts to optimize performance
def get_optimizer(model):
    # Specify custom learning rates for base transformer and task-specific heads
    layerwise_learning_rates = {
        "transformer": 1e-5,
        "classification_head": 1e-4,
        "sentiment_head": 1e-4
    }
    
    # Organize model parameters by learning rate for optimizer
    optimizer_parameters = [
        {"params": model.transformer.transformer.layer[:3].parameters(), "lr": layerwise_learning_rates["transformer"] / 2},
        {"params": model.transformer.transformer.layer[3:].parameters(), "lr": layerwise_learning_rates["transformer"]},
        {"params": model.classification_head.parameters(), "lr": layerwise_learning_rates["classification_head"]},
        {"params": model.sentiment_head.parameters(), "lr": layerwise_learning_rates["sentiment_head"]}
    ]
    
    # Create an optimizer for model training
    optimizer = torch.optim.Adam(optimizer_parameters)
    return optimizer

# Training function with limited iterations for quick experimentation
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
        
        classification_output, sentiment_output = model(input_ids, attention_mask)
        
        category_loss = criterion_category(classification_output, labels)
        sentiment_loss = criterion_sentiment(sentiment_output, sentiments)
        loss = 0.7 * category_loss + 0.3 * sentiment_loss
        logging.info(f"Iteration {iters + 1}, Loss: {loss.item()}, Category Loss: {category_loss.item()}, Sentiment Loss: {sentiment_loss.item()}")
        
        # Calculate accuracy for current batch
        predicted_category = classification_output.argmax(dim=1)
        predicted_sentiment = sentiment_output.argmax(dim=1)
        
        correct_category_predictions += (predicted_category == labels).sum().item()
        correct_sentiment_predictions += (predicted_sentiment == sentiments).sum().item()
        total_samples += labels.size(0)
        
        category_accuracy = (correct_category_predictions / total_samples) * 100
        sentiment_accuracy = (correct_sentiment_predictions / total_samples) * 100
        logging.info(f"Category Accuracy: {category_accuracy:.2f}%, Sentiment Accuracy: {sentiment_accuracy:.2f}%")
        
        print(f"Iteration {iters + 1}, Loss: {loss.item():.4f}, Category Accuracy: {category_accuracy:.2f}%, Sentiment Accuracy: {sentiment_accuracy:.2f}%")
        
        loss.backward()
        optimizer.step()
        iters += 1
    
    print("Training complete for the limited number of iterations.")

# Function to test sentence embeddings by generating embeddings for sample sentences
def test_sentence_embeddings(model, tokenizer, device):
    sample_sentences = [
        "The food was amazing and the service was excellent!",
        "I had a terrible experience with the car repair service.",
        "The salon offers great beauty treatments and friendly staff.",
    ]
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for sentence in sample_sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
            embedding = model.transformer(**inputs).last_hidden_state[:, 0, :]  # Extract the [CLS] embedding
            print(f"Sentence: {sentence}")
            print("Embedding:", embedding.squeeze().cpu().numpy())
            print("=" * 80)

# Main function to load data, initialize model and tokenizer, and begin training
def main():
    data = load_and_prepare_data('yelp_dataset/yelp_review.json', 
                                 'yelp_dataset/yelp_business.json', 
                                 sample_size=50000)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    train_dataset = YelpDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    num_classes = len(train_dataset.category_mapping)
    transformer_model = AutoModel.from_pretrained("distilbert-base-uncased")
    model = MultiTaskModel(transformer_model, num_classes=num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Compute class weights for balanced loss calculation
    class_weights = compute_class_weight('balanced', classes=np.array(list(train_dataset.category_mapping.values())), 
                                         y=train_data['categories'].map(train_dataset.category_mapping))
    category_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion_category = torch.nn.CrossEntropyLoss(weight=category_weights)
    criterion_sentiment = torch.nn.CrossEntropyLoss()

    # Get layer-wise optimizer
    optimizer = get_optimizer(model)
    
    train(model, train_dataloader, optimizer, criterion_category, criterion_sentiment, device, max_iters=60)
    
    # Test the model on sample sentences for embeddings
    print("\nTesting Sentence Embeddings:")
    test_sentence_embeddings(model, tokenizer, device)

if __name__ == "__main__":
    main()
