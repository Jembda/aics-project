# -------------------------------------
# Siamese Network for Multimodal Similarity Learning on WikiDiverse Dataset
# ---------------------------------------------------------------------------

# This implementation is a modification and integration of concepts from multisources:
##Primary Sources: 

# 1. GeeksforGeeks (2024). “Siamese Neural Network in Deep Learning”. In: Geeks-
# forGeeks. Accessed: 2024-11-17. url: https://www.geeksforgeeks.org/
# nlp/siamese-neural-network-in-deep-learning/. 
# 2. Dutt, Aditya (2021). “Siamese networks introduction and implementation”. In:
# Medium, Towards Data Science 11.
# 3. Singh, Prabhnoor (2019). Siamese network keras for image and text similarity.
# 4. Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov (2015). “Siamese
# Neural Networks for One-shot Image Recognition”. In: Proceedings of the
# ICML Deep Learning Workshop

#--------------------------------------
# Key Modifications and Integrations:
# -------------------------------------
# Extend to multimodal (image + text) inputes instead of single modality
# Added cross-attention mechanism
# Implemenated robust data loader for Wikidivesre dataset
# Added comprhensive logging and error handling 
# Integrated BERT for text encoding alongside ResNet for images
# Added advanced training features (early stopping, LR scheduling)
# Implemented multi-feature fusion

# -------------------------------------
# An overview
# -------------------------------------
# Siamese networks consist of two identical sub-networks that share weights and learn to compute the similarity between two input samples. The goal is to learn embeddings such that similar inputs are close in the embedding space, while dissimilar inputs are far apart. For the WikiDiverse dataset, where we have image-caption pairs, we can build a Siamese network that processes text and image data (or just one modality like text or image) and learns to compute similarity between two entities from the knowledge base.
# * Siamese Network Structure: Two identical sub-networks that compute embeddings for input pairs and learn their similarity
# * Application: For WikiDiverse, compute similarity between image-caption pairs to link knowledge-base entities.

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import hashlib
import re
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

#-----------------------------------
# 1. Dataset Path and Hyperparameters
#-----------------------------------
BASE_DIR = "/home/gusjembda@GU.GU.SE/aics-project"
DATASET_PATH = os.path.join(BASE_DIR, "wikidiverse_w_cands")
IMAGE_DIR = os.path.join(BASE_DIR, "wikinewsImgs")
TRAIN_JSON = os.path.join(DATASET_PATH, "train_w_10cands.json")
VALID_JSON = os.path.join(DATASET_PATH, "valid_w_10cands.json")

# Create image directory if it doesn't exist
os.makedirs(IMAGE_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 2e-5
EMBED_DIM = 512
MARGIN = 1.0
PATIENCE = 5
MAX_PAIRS_PER_ENTITY = 100

#-----------------------------------
# 2. Dataset Loading and Pair Generation
#-----------------------------------
def load_and_process_data(json_path):
    """
    Load and process WikiDiverse JSON data:
    Based on data handling concept from:
    - Sing(2019) for multimodal data preparation
    - Koch et al (2015) for paired training
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from {json_path}")
        return [{
            'sentence': item[0],
            'img_url': item[1],
            'mention': item[2],
            'entity_url': item[6]
        } for item in data]
    except Exception as e:
        logger.error(f"Error loading {json_path}: {e}")
        return []

def get_local_image_path(img_url):
    """Convert image URL to local file path with robust handling"""
    filename = img_url.split('/')[-1]
    prefix = hashlib.md5(filename.encode()).hexdigest()
    
    # Extract extension using regex
    ext_match = re.search(r'\.(jpe?g|png|gif|svg|bmp|tiff?|webp)$', filename, re.IGNORECASE)
    ext = ext_match.group(0).lower() if ext_match else '.jpg'
    
    # Handle SVG conversion
    if ext.lower() in ['.svg', '.svgz']:
        ext = '.png'
    
    return os.path.join(IMAGE_DIR, f"{prefix}{ext}")

def generate_pairs(data, max_pairs_per_entity=MAX_PAIRS_PER_ENTITY):
    """
    Generate balanced positive and negative pairs with limits.
    Pair generation strategy adapted from:
    -Dutt (2021) for balanced positive/negetative sampling
    -Kotch et al (2015) one-shot learning pair construction
    """
    if not data:
        logger.warning("No data to generate pairs")
        return []
    
    # Group by entity
    entity_dict = {}
    for item in data:
        entity_url = item['entity_url']
        if entity_url not in entity_dict:
            entity_dict[entity_url] = []
        entity_dict[entity_url].append(item)
    
    pairs = []
    entities = list(entity_dict.keys())
    valid_entities = [e for e in entities if len(entity_dict[e]) > 1]
    
    # Generate positive pairs (same entity) with limits
    for entity in valid_entities:
        items = entity_dict[entity]
        n = len(items)
        max_pairs = min(max_pairs_per_entity, n * (n - 1) // 2)
        
        # Generate unique pairs
        indices = list(range(n))
        random.shuffle(indices)
        generated_pairs = set()
        
        while len(generated_pairs) < max_pairs:
            i, j = random.sample(indices, 2)
            if i != j:
                pair_key = tuple(sorted((i, j)))
                if pair_key not in generated_pairs:
                    generated_pairs.add(pair_key)
                    pairs.append({
                        'img1': items[i]['img_url'],
                        'text1': items[i]['sentence'],
                        'img2': items[j]['img_url'],
                        'text2': items[j]['sentence'],
                        'label': 1  # Positive pair
                    })
    
    num_positive = len(pairs)
    logger.info(f"Generated {num_positive} positive pairs")
    
    # Generate negative pairs (different entities)
    for _ in range(num_positive):
        entity1, entity2 = random.sample(valid_entities, 2)
        item1 = random.choice(entity_dict[entity1])
        item2 = random.choice(entity_dict[entity2])
        pairs.append({
            'img1': item1['img_url'],
            'text1': item1['sentence'],
            'img2': item2['img_url'],
            'text2': item2['sentence'],
            'label': 0  # Negative pair
        })
    
    logger.info(f"Generated {len(pairs) - num_positive} negative pairs")
    return pairs

#-----------------------------------
# 3. Dataset Class with Robust Handling
#-----------------------------------
class WikiDiverseDataset(Dataset):
    """
    Custom dataset for multimodal data loading
    - Sigh (2019) for multimodal data loading
    - GeeksforGeeks (2024) for Siamese network data structure
    """
    def __init__(self, pairs, transform=None, tokenizer=None, max_length=128):
        self.pairs = self._filter_valid_pairs(pairs)
        self.transform = transform or self.get_default_transform()
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        logger.info(f"Initialized dataset with {len(self.pairs)} valid pairs")

    def get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def _filter_valid_pairs(self, pairs):
        """Filter out pairs with missing images"""
        valid_pairs = []
        for pair in pairs:
            img1_path = get_local_image_path(pair['img1'])
            img2_path = get_local_image_path(pair['img2'])
            
            # Skip pairs with missing images
            if not os.path.exists(img1_path):
                logger.debug(f"Missing image: {img1_path}")
                continue
            if not os.path.exists(img2_path):
                logger.debug(f"Missing image: {img2_path}")
                continue
            
            valid_pairs.append(pair)
        return valid_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Helper function to load images safely
        def load_image(img_url):
            img_path = get_local_image_path(img_url)
            try:
                img = Image.open(img_path).convert('RGB')
                return self.transform(img)
            except Exception as e:
                logger.warning(f"Error loading image {img_path}: {e}")
                # Create blank image for missing files
                blank = Image.new('RGB', (224, 224), (0, 0, 0))
                return self.transform(blank)
        
        # Load and transform images
        img1 = load_image(pair['img1'])
        img2 = load_image(pair['img2'])
        
        # Tokenize text
        text1 = self.tokenizer(
            pair['text1'], 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text2 = self.tokenizer(
            pair['text2'], 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'img1': img1,
            'img2': img2,
            'input_ids1': text1['input_ids'].squeeze(0),
            'attention_mask1': text1['attention_mask'].squeeze(0),
            'input_ids2': text2['input_ids'].squeeze(0),
            'attention_mask2': text2['attention_mask'].squeeze(0),
            'label': torch.tensor(pair['label'], dtype=torch.float32)
        }

#-----------------------------------
# 4. Model Components
#-----------------------------------
class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for multimodal feature fusion
    Extension beyond origianl sources:
    -Added cross-model attention between image and text features
    -Inspired by moden transformation architectures
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        return self.norm(query + attn_output)

class ImageEncoder(nn.Module):
    """
    Image encoder  with cross-attention capablities 
    Based on concepts from:
    -Koch et al. (2015) for image features extraction in Siamese network
    -Sigh (2019) for multimodal integration    
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.cross_attention = CrossAttention(embed_dim)
        
    def forward(self, x, text_features=None):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if text_features is not None:
            # Reshape for attention: (seq_len, batch, features)
            x = self.cross_attention(
                query=x.unsqueeze(0),
                key=text_features.unsqueeze(0),
                value=text_features.unsqueeze(0)
            ).squeeze(0)
        return x

class TextEncoder(nn.Module):
    """
    Text encoder with cross-attention capablities.
    Based on concepts from:
    -Singh (2019) for text in multimodal Siamese networks
    -Modern BERT integration for text features 
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.proj = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.cross_attention = CrossAttention(embed_dim)
        
    def forward(self, input_ids, attention_mask, image_features=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        text_features = self.proj(outputs.last_hidden_state[:, 0, :])
        if image_features is not None:
            text_features = self.cross_attention(
                query=text_features.unsqueeze(0),
                key=image_features.unsqueeze(0),
                value=image_features.unsqueeze(0)
            ).squeeze(0)
        return text_features

class SiameseNetwork(nn.Module):
    """
    Multimodal Siamese Network for Similarity learning 
    Core architecture based on:
    - Koch et al. (2015) - Original Siamese network concept
    - Dutt (2021) - practical implementation details
    - GeeksforGeeks (2024) - Architecture overview
    - Singh (2019) - Multimodal extensions

    Key modification:
    -Added cross-modal attention between image and text
    - Multi-feature fusion
    - Enhanced classifier head for similarity prediction
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.img_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, img1, input_ids1, attention_mask1, img2, input_ids2, attention_mask2):
        # First pair
        text_feat1 = self.text_encoder(input_ids1, attention_mask1)
        img_feat1 = self.img_encoder(img1, text_feat1)
        text_feat1 = self.text_encoder(input_ids1, attention_mask1, img_feat1)
        
        # Second pair
        text_feat2 = self.text_encoder(input_ids2, attention_mask2)
        img_feat2 = self.img_encoder(img2, text_feat2)
        text_feat2 = self.text_encoder(input_ids2, attention_mask2, img_feat2)
        
        # Combine features
        combined1 = torch.cat([img_feat1, text_feat1], dim=1)
        combined2 = torch.cat([img_feat2, text_feat2], dim=1)
        
        # Similarity features
        diff = torch.abs(combined1 - combined2)
        product = combined1 * combined2
        similarity_features = torch.cat([combined1, combined2, diff, product], dim=1)
        
        return self.classifier(similarity_features).squeeze(1)

#-----------------------------------
# 5. Training and Evaluation Functions
#-----------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    """
    Training loop for one epoch
    Training strategy  incorporation:
    - Modern practices for stable training
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Prepare data
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(img1, input_ids1, attention_mask1, 
                        img2, input_ids2, attention_mask2)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * img1.size(0)
        progress_bar.set_postfix(loss=loss.item())
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    """
    Evaluation function with comprehensive metrics

    Evaluation approach based on:
    - Standard practices from all cited sources
    - comprehensive metrics calculation for binary classification
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            # Prepare data
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(img1, input_ids1, attention_mask1, 
                            img2, input_ids2, attention_mask2)
            loss = criterion(outputs, labels)
            
            # Collect predictions
            total_loss += loss.item() * img1.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, all_preds_binary)
    precision = precision_score(all_labels, all_preds_binary)
    recall = recall_score(all_labels, all_preds_binary)
    f1 = f1_score(all_labels, all_preds_binary)
    auc = roc_auc_score(all_labels, all_preds)
    
    return total_loss / len(loader.dataset), accuracy, precision, recall, f1, auc

#-----------------------------------
# 6. Main Execution
#-----------------------------------
def main():
    """
    Main training pipeline integration concepts from all sources
    """
    # Load and prepare data
    logger.info("Loading training data...")
    train_data = load_and_process_data(TRAIN_JSON)
    logger.info(f"Loaded {len(train_data)} training items")
    
    logger.info("Loading validation data...")
    valid_data = load_and_process_data(VALID_JSON)
    logger.info(f"Loaded {len(valid_data)} validation items")
    
    logger.info("Generating training pairs...")
    train_pairs = generate_pairs(train_data)
    logger.info("Generating validation pairs...")
    valid_pairs = generate_pairs(valid_data)
    
    logger.info(f"Generated {len(train_pairs)} training pairs")
    logger.info(f"Generated {len(valid_pairs)} validation pairs")
    
    # Create datasets and dataloaders
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    logger.info("Creating datasets...")
    train_dataset = WikiDiverseDataset(train_pairs, transform, tokenizer)
    valid_dataset = WikiDiverseDataset(valid_pairs, transform, tokenizer)
    
    # Check if datasets have samples
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        logger.error("No valid samples in datasets. Exiting.")
        return
    
    # Use persistent workers for efficiency
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True,
                             persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, 
                             shuffle=False, num_workers=4, pin_memory=True,
                             persistent_workers=True)
    
    # Initialize model
    logger.info("Initializing model...")
    model = SiameseNetwork(EMBED_DIM).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Training components
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training variables
    best_val_loss = float('inf')
    best_val_auc = 0.0
    no_improve = 0
    train_losses = []
    val_losses = []
    val_metrics = []
    
    # Training loop
    for epoch in range(EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train phase
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validation phase
        val_loss, acc, prec, rec, f1, auc = evaluate(model, valid_loader, criterion, device)
        val_losses.append(val_loss)
        val_metrics.append((acc, prec, rec, f1, auc))
        
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        logger.info(f"F1 Score: {f1:.4f} | AUC: {auc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_model_loss.pth")
            logger.info("Saved best model (lowest loss)!")
        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), "best_model_auc.pth")
            logger.info("Saved best model (highest AUC)!")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info(f"No improvement for {PATIENCE} epochs. Early stopping!")
                break
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    metrics = np.array(val_metrics)
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    for i in range(5):
        plt.plot(metrics[:, i], label=labels[i])
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()