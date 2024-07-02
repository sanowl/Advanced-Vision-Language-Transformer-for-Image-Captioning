import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm
import argparse
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import secrets

class ImageCaptioningDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, max_length=50):
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)
        
        self.image_files = list(self.captions.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        caption = secrets.choice(self.captions[image_file])
        tokens = self.tokenizer.encode_plus(
            caption, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        return image, tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class VisionLanguageTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(VisionLanguageTransformer, self).__init__()
        
        # Image encoder (ResNet-152)
        resnet = models.resnet152(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Image feature projection
        self.image_projection = nn.Linear(2048, d_model)
        
        # Text embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output layer
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Encode image
        src = self.image_encoder(src)
        src = src.flatten(2).permute(2, 0, 1)  # (seq_len, batch_size, channels)
        src = self.image_projection(src) * np.sqrt(self.d_model)
        
        # Embed and position-encode text
        tgt = self.embedding(tgt) * np.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # Create masks if not provided
        if src_mask is None:
            src_mask = torch.zeros((src.shape[0], src.shape[0])).bool().to(src.device)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)
        
        # Project to vocabulary
        output = self.fc(output)
        
        return output

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif mask is None:
            mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def train_epoch(model, dataloader, criterion, sup_con_loss, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        images, captions, attention_mask = batch
        images, captions, attention_mask = images.to(device), captions.to(device), attention_mask.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images, captions[:, :-1])
            loss_ce = criterion(outputs.view(-1, outputs.size(-1)), captions[:, 1:].contiguous().view(-1))
            
            # Contrastive loss
            image_features = model.image_projection(model.image_encoder(images).mean([-2, -1]))
            text_features = outputs.mean(0)
            combined_features = torch.cat([image_features.unsqueeze(1), text_features.unsqueeze(1)], dim=1)
            loss_con = sup_con_loss(combined_features)
            
            loss = loss_ce + 0.1 * loss_con

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images, captions, attention_mask = batch
            images, captions, attention_mask = images.to(device), captions.to(device), attention_mask.to(device)
            
            outputs = model(images, captions[:, :-1])
            loss = criterion(outputs.view(-1, outputs.size(-1)), captions[:, 1:].contiguous().view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageCaptioningDataset(args.image_dir, args.train_captions, transform)
    val_dataset = ImageCaptioningDataset(args.image_dir, args.val_captions, transform)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Initialize model
    model = VisionLanguageTransformer(vocab_size=train_dataset.tokenizer.vocab_size).to(device)
    
    # Define loss functions
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_token_id)
    sup_con_loss = SupConLoss()
    
    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_dataloader, criterion, sup_con_loss, optimizer, scheduler, scaler, device)
        val_loss = validate(model, val_dataloader, criterion, device)
        
        logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_path)
            logging.info("Model saved")
    
    logging.info("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision-Language Transformer for Image Captioning")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--train_captions", type=str, required=True, help="JSON file containing training captions")
    parser.add_argument("--val_captions", type=str, required=True, help="JSON file containing validation captions")
    parser.add_argument("--model_path", type=str, default="vision_language_transformer.pth", help="Path to save the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    
    args = parser.parse_args()
    main(args)
