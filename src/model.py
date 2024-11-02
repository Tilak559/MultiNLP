import torch
import torch.nn as nn

# Define a custom neural network model for multi-task learning
class MultiTaskModel(nn.Module):
    def __init__(self, transformer_model, num_classes, num_sentiments=3):
        super(MultiTaskModel, self).__init__()
        
        # Use a pre-trained transformer model as the base of this multi-task model
        self.transformer = transformer_model
        
        # Define a classification head for predicting categories (broad categories like 'Restaurants')
        self.classification_head = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
        # Define a sentiment head for predicting sentiments (e.g., positive, neutral, negative)
        self.sentiment_head = nn.Linear(self.transformer.config.hidden_size, num_sentiments)

    def forward(self, input_ids, attention_mask):
        # Forward pass through the transformer model; get output embeddings for input tokens
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the [CLS] token embedding, which is commonly used for classification tasks
        cls_embedding = transformer_output.last_hidden_state[:, 0, :]
        
        # Pass the [CLS] embedding to the classification head for category prediction
        classification_output = self.classification_head(cls_embedding)
        
        # Pass the [CLS] embedding to the sentiment head for sentiment prediction
        sentiment_output = self.sentiment_head(cls_embedding)
        
        # Return outputs for both classification and sentiment tasks
        return classification_output, sentiment_output
