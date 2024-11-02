import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, transformer_model, num_classes, num_sentiments=3):
        super(MultiTaskModel, self).__init__()
        self.transformer = transformer_model
        self.classification_head = nn.Linear(self.transformer.config.hidden_size, num_classes)
        self.sentiment_head = nn.Linear(self.transformer.config.hidden_size, num_sentiments)

    def forward(self, input_ids, attention_mask):
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = transformer_output.last_hidden_state[:, 0, :]
        classification_output = self.classification_head(cls_embedding)
        sentiment_output = self.sentiment_head(cls_embedding)
        return classification_output, sentiment_output
