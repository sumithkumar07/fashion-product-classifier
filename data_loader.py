import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import json
import numpy as np

class FashionProductDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_training=True, is_test=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            img_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            is_training (bool): Whether this is training set or not
            is_test (bool): Whether this is test set or not
        """
        # Read CSV with error handling
        try:
            self.df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            raise
            
        self.img_dir = img_dir
        self.transform = transform
        self.is_training = is_training
        self.is_test = is_test
        
        # Define the columns we want to keep
        self.target_columns = ['articleType', 'baseColour', 'season', 'gender']
        
        # Load label mappings
        try:
            with open('label_mappings.json', 'r') as f:
                self.label_mappings = json.load(f)
        except Exception as e:
            print(f"Error loading label mappings: {e}")
            raise
        
        # Filter out rows where image doesn't exist
        self.df['image_path'] = self.df['id'].apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))
        self.df = self.df[self.df['image_path'].apply(os.path.exists)].reset_index(drop=True)
        
        # Convert labels to lowercase
        for column in self.target_columns:
            self.df[column] = self.df[column].str.lower()
        
        print(f"Dataset loaded with {len(self.df)} valid samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.df.iloc[idx]['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), 'black')
        
        # Get labels for all target columns
        labels = {}
        for column in self.target_columns:
            label = str(self.df.iloc[idx][column]).lower()
            try:
                label_idx = self.label_mappings[column]['label_to_idx'][label]
                labels[column] = label_idx
            except KeyError:
                print(f"Warning: Label '{label}' not found in mapping for {column}")
                # Use the most common label as fallback
                most_common_label = max(self.label_mappings[column]['class_weights'].items(), 
                                      key=lambda x: x[1])[0]
                labels[column] = int(most_common_label)
        
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to {img_path}: {e}")
                # Return a black tensor if transform fails
                image = torch.zeros((3, 224, 224))
            
        return image, labels
    
    def get_sample_weights(self):
        """Calculate sample weights for balanced sampling"""
        weights = torch.ones(len(self))
        for column in self.target_columns:
            # Convert labels to indices
            labels = self.df[column].str.lower().apply(
                lambda x: self.label_mappings[column]['label_to_idx'].get(str(x), 0)
            )
            
            # Get class weights
            class_weights = self.label_mappings[column]['class_weights']
            
            # Convert class weights to tensor format
            weight_tensor = torch.tensor([
                class_weights.get(str(i), 1.0) 
                for i in range(len(self.label_mappings[column]['labels']))
            ])
            
            # Get weight for each sample
            sample_weights = weight_tensor[labels]
            
            # Multiply current weights by new weights
            weights *= sample_weights
        
        return weights

def get_data_loaders(csv_file, img_dir, batch_size=32, num_workers=0, train_transform=None, val_transform=None, is_test=False):
    """
    Create train, validation, and test data loaders with balanced sampling
    """
    # Use default transforms if none provided
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Create datasets
    if is_test:
        test_dataset = FashionProductDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=val_transform,  # Use validation transforms for test
            is_training=False,
            is_test=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        return None, None, test_loader, test_dataset.label_mappings
    
    else:
        # Split dataset into train and validation
        dataset = FashionProductDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=train_transform,
            is_training=True
        )
        
        # Calculate split sizes
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        # Split dataset
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Apply validation transforms to validation set
        val_dataset.dataset.transform = val_transform
        
        # Calculate sample weights for balanced sampling
        train_weights = dataset.get_sample_weights()
        train_weights = train_weights[train_dataset.indices]
        
        # Create samplers
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, None, dataset.label_mappings

if __name__ == "__main__":
    # Test the data loader
    train_loader, val_loader, test_loader, label_mappings = get_data_loaders(
        csv_file='data/processed_styles.csv',
        img_dir='data/images',
        batch_size=32,
        num_workers=0  # Set to 0 for debugging
    )
    
    print("\nNumber of classes for each attribute:")
    for column, mapping in label_mappings.items():
        print(f"{column}: {len(mapping['labels'])}")
    
    if train_loader:
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
    if test_loader:
        print(f"\nTest batches: {len(test_loader)}")
    
    # Get a sample batch
    if train_loader:
        images, labels = next(iter(train_loader))
        print(f"Train batch shape: {images.shape}")
        print("Train labels shape for each attribute:")
        for column, label_tensor in labels.items():
            print(f"{column}: {label_tensor.shape}")
    if test_loader:
        images, labels = next(iter(test_loader))
        print(f"Test batch shape: {images.shape}")
        print("Test labels shape for each attribute:")
        for column, label_tensor in labels.items():
            print(f"{column}: {label_tensor.shape}") 