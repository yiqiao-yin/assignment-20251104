import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from typing import Dict, Any
import os

class LogisticRegressionModel(nn.Module):
    """Simple Logistic Regression model for MNIST classification"""
    
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        # Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        return self.linear(x)

def load_mnist_data(batch_size: int = 64):
    """Load and preprocess MNIST dataset"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization values
    ])
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader, train_dataset, test_dataset

def train_model(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def test_model(model, test_loader, criterion, device):
    """Test the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def train_mnist_model(epochs: int = 10, learning_rate: float = 0.01, batch_size: int = 64) -> Dict[str, Any]:
    """
    Train a logistic regression model on MNIST dataset
    
    Args:
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for data loading
    
    Returns:
        Dictionary containing training results and model information
    """
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Starting MNIST Logistic Regression Training...")
        print(f"Device: {device}")
        print(f"Parameters: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}")
        
        start_time = time.time()
        
        # Load data
        print("Loading MNIST dataset...")
        train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(batch_size)
        
        # Initialize model
        model = LogisticRegressionModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        # Training history
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        
        # Training loop
        print(f"\nStarting training for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Test
            test_loss, test_acc = test_model(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s):")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        total_time = time.time() - start_time
        
        # Save model
        model_path = f"mnist_logistic_model_e{epochs}_lr{learning_rate}_bs{batch_size}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'final_train_accuracy': train_accuracies[-1],
            'final_test_accuracy': test_accuracies[-1],
        }, model_path)
        
        # Prepare results
        results = {
            "status": "success",
            "message": f"Successfully trained logistic regression model on MNIST",
            "training_parameters": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "device": str(device)
            },
            "dataset_info": {
                "training_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                "input_features": 784,
                "output_classes": 10
            },
            "training_results": {
                "total_training_time_seconds": round(total_time, 2),
                "final_train_loss": round(train_losses[-1], 4),
                "final_train_accuracy": round(train_accuracies[-1], 2),
                "final_test_loss": round(test_losses[-1], 4),
                "final_test_accuracy": round(test_accuracies[-1], 2),
                "best_test_accuracy": round(max(test_accuracies), 2),
                "model_saved_as": model_path
            },
            "training_history": {
                "train_losses": [round(x, 4) for x in train_losses],
                "train_accuracies": [round(x, 2) for x in train_accuracies],
                "test_losses": [round(x, 4) for x in test_losses],
                "test_accuracies": [round(x, 2) for x in test_accuracies]
            }
        }
        
        print(f"\n✅ Training completed successfully!")
        print(f"Final test accuracy: {test_accuracies[-1]:.2f}%")
        print(f"Model saved as: {model_path}")
        
        return results
        
    except Exception as e:
        error_results = {
            "status": "error",
            "message": f"Training failed: {str(e)}",
            "error_type": type(e).__name__
        }
        print(f"❌ Training failed: {str(e)}")
        return error_results

def get_model_info() -> Dict[str, Any]:
    """Get information about the MNIST logistic regression model and dataset"""
    
    info = {
        "model_architecture": {
            "type": "Logistic Regression",
            "framework": "PyTorch",
            "input_size": 784,
            "output_size": 10,
            "parameters": 7850,  # 784 * 10 + 10 (weights + biases)
            "layers": [
                {
                    "name": "Linear Layer",
                    "input_features": 784,
                    "output_features": 10,
                    "parameters": 7850
                }
            ]
        },
        "dataset_info": {
            "name": "MNIST",
            "description": "Handwritten digits dataset",
            "training_samples": 60000,
            "test_samples": 10000,
            "image_size": "28x28 pixels",
            "channels": 1,
            "classes": 10,
            "class_names": list(range(10))
        },
        "preprocessing": {
            "normalization": "Mean=0.1307, Std=0.3081",
            "transforms": [
                "ToTensor()",
                "Normalize((0.1307,), (0.3081,))"
            ]
        },
        "typical_performance": {
            "expected_accuracy_range": "85-92%",
            "training_time": "1-5 minutes (CPU)",
            "convergence_epochs": "5-15 epochs"
        }
    }
    
    return info

if __name__ == "__main__":
    # Example usage
    print("Testing MNIST Logistic Regression...")
    
    # Get model info
    print("\n" + "="*50)
    print("MODEL INFO")
    print("="*50)
    info = get_model_info()
    print(f"Model: {info['model_architecture']['type']}")
    print(f"Parameters: {info['model_architecture']['parameters']}")
    print(f"Dataset: {info['dataset_info']['name']}")
    print(f"Training samples: {info['dataset_info']['training_samples']}")
    
    # Train model with default parameters
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    results = train_mnist_model(epochs=3, learning_rate=0.01, batch_size=64)
    
    if results["status"] == "success":
        print(f"\n✅ Training successful!")
        print(f"Final accuracy: {results['training_results']['final_test_accuracy']}%")
    else:
        print(f"\n❌ Training failed: {results['message']}")
