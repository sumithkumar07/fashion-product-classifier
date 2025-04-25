import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import get_data_loaders
from model import create_model, get_loss_fn
import time
from tqdm import tqdm
import json
import numpy as np
from torch.cuda.amp import autocast, GradScaler

def get_weighted_loss_fn(label_mappings):
    """Get weighted loss functions for each attribute"""
    loss_fns = {}
    for attr, mapping in label_mappings.items():
        # Convert class weights from label-based to index-based
        num_classes = len(mapping['labels'])
        weights = torch.ones(num_classes)
        
        # Map the weights to their corresponding indices
        for label, idx in mapping['label_to_idx'].items():
            if str(label) in mapping['class_weights']:
                weights[idx] = mapping['class_weights'][str(label)]
        
        # Normalize weights
        weights = weights / weights.mean()
        loss_fns[attr] = nn.CrossEntropyLoss(weight=weights.cuda())
    
    return loss_fns

def train_epoch(model, train_loader, optimizer, loss_fns, device, scaler):
    model.train()
    total_loss = 0
    correct_predictions = {attr: 0 for attr in model.heads.keys()}
    total_samples = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(images)
            
            # Calculate loss for each attribute
            losses = []
            for attribute in model.heads.keys():
                loss = loss_fns[attribute](outputs[attribute], labels[attribute])
                losses.append(loss)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs[attribute].data, 1)
                correct_predictions[attribute] += (predicted == labels[attribute]).sum().item()
            
            # Total loss is the weighted sum of individual losses
            weights = {
                'articleType': 1.0,
                'baseColour': 0.8,
                'season': 0.6,
                'gender': 0.4
            }
            total_batch_loss = sum(loss * weights[attr] for loss, attr in zip(losses, model.heads.keys()))
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(total_batch_loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += total_batch_loss.item()
        total_samples += labels[list(labels.keys())[0]].size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    accuracies = {
        attr: correct / total_samples * 100
        for attr, correct in correct_predictions.items()
    }
    
    return avg_loss, accuracies

def validate(model, val_loader, loss_fns, device):
    model.eval()
    total_loss = 0
    correct_predictions = {attr: 0 for attr in model.heads.keys()}
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss for each attribute
            losses = []
            for attribute in model.heads.keys():
                loss = loss_fns[attribute](outputs[attribute], labels[attribute])
                losses.append(loss)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs[attribute].data, 1)
                correct_predictions[attribute] += (predicted == labels[attribute]).sum().item()
            
            total_loss += sum(losses).item()
            total_samples += labels[list(labels.keys())[0]].size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader)
    accuracies = {
        attr: correct / total_samples * 100
        for attr, correct in correct_predictions.items()
    }
    
    return avg_loss, accuracies

def train_model(model, train_loader, val_loader, label_mappings, num_epochs=20, device='cuda'):
    # Move model to device
    model = model.to(device)
    
    # Loss functions with class weights
    loss_fns = get_weighted_loss_fn(label_mappings)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    best_accuracies = {attr: 0.0 for attr in model.heads.keys()}
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fns, device, scaler)
        print(f"Train Loss: {train_loss:.4f}")
        print("Train Accuracies:")
        for attr, acc in train_acc.items():
            print(f"  {attr}: {acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, loss_fns, device)
        print(f"Val Loss: {val_loss:.4f}")
        print("Val Accuracies:")
        for attr, acc in val_acc.items():
            print(f"  {attr}: {acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_accuracies = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, 'best_model.pth')
            print("Saved best model!")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    
    print("\nBest Validation Accuracies:")
    for attr, acc in best_accuracies.items():
        print(f"{attr}: {acc:.2f}%")

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, label_mappings = get_data_loaders(
        csv_file='data/processed_styles.csv',
        img_dir='data/images',
        batch_size=32
    )
    
    # Create model
    model = create_model(label_mappings)
    
    # Train model
    train_model(model, train_loader, val_loader, label_mappings, num_epochs=20, device=device) 