import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Initialize BERT tokenizer and model for textual encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Initialize ResNet model for image encoding (using ResNet-50 here)
resnet_model = models.resnet50(pretrained=True)
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])  # Remove final classification layer

# Cross-Attention Layer
class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, text_features, image_features):
        # Cross-attention: text to image and image to text
        text_features = text_features.unsqueeze(0)  # Add batch dimension
        image_features = image_features.unsqueeze(0)  # Add batch dimension

        # Perform attention between text and image features
        attn_output_text, _ = self.attention(text_features, image_features, image_features)
        attn_output_image, _ = self.attention(image_features, text_features, text_features)
        
        # Combine the outputs (you can experiment with different strategies like sum, concat, etc.)
        combined_output = attn_output_text + attn_output_image
        combined_output = self.fc(combined_output)  # Feedforward layer after attention
        return combined_output

# Entity Disambiguation Head
class EntityDisambiguationHead(nn.Module):
    def __init__(self, hidden_size, num_candidates):
        super(EntityDisambiguationHead, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, num_candidates)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = self.fc2(x)
        return self.softmax(x)

# Model combining Text and Image features
class MultimodalEntityLinkingModel(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_candidates):
        super(MultimodalEntityLinkingModel, self).__init__()
        self.text_encoder = bert_model
        self.image_encoder = resnet_model
        self.cross_attention_layer = CrossAttentionLayer(hidden_size, num_attention_heads)
        self.disambiguation_head = EntityDisambiguationHead(hidden_size, num_candidates)

    def forward(self, text_input, image_input):
        # Textual feature extraction using BERT
        encoded_input = tokenizer(text_input, return_tensors='pt', padding=True, truncation=True)
        text_output = self.text_encoder(**encoded_input).last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        
        # Image feature extraction using ResNet
        image_input = image_input.unsqueeze(0)  # Add batch dimension
        image_features = self.image_encoder(image_input)  # shape: (batch_size, channels, 1, 1)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten to (batch_size, feature_size)

        # Apply cross-attention to fuse text and image features
        combined_features = self.cross_attention_layer(text_output, image_features)

        # Disambiguate and predict the entity
        entity_scores = self.disambiguation_head(combined_features.squeeze(0))  # Removing batch dimension
        return entity_scores

# Helper function for calculating loss and accuracy
def calculate_loss_and_accuracy(model, text_input, image_input, correct_entity_index):
    # Forward pass through the model
    entity_scores = model(text_input, image_input)

    # Compute the cross-entropy loss for entity disambiguation
    labels = torch.tensor([correct_entity_index])  # The index of the correct entity in the candidate list
    criterion = nn.CrossEntropyLoss()
    loss = criterion(entity_scores, labels)
    
    # Get the predicted entity
    predicted_entity = torch.argmax(entity_scores, dim=1)
    accuracy = (predicted_entity == labels).float().mean()
    
    return loss, accuracy

# Initialize model parameters
hidden_size = 768  # BERT hidden size
num_attention_heads = 8
num_candidates = 10  # Number of candidates for entity linking

# Initialize the model
model = MultimodalEntityLinkingModel(hidden_size, num_attention_heads, num_candidates)

# Example input (text and image)
text_input = "The Lions versus the Packers (2007)."
image_path = "path_to_image.jpg"  # Replace with your image path
image_input = Image.open(image_path)

# Preprocess image to fit ResNet input format
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_input = preprocess(image_input)

# Assume we have the index of the correct entity (for example purposes)
correct_entity_index = 0

# Calculate loss and accuracy
loss, accuracy = calculate_loss_and_accuracy(model, text_input, image_input, correct_entity_index)
print(f"Loss: {loss.item()}, Accuracy: {accuracy.item()}")
