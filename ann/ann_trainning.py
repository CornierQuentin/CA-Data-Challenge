import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def train_model(model, criterion, optimizer, train_loader, num_epochs, checkpoint_path):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Save the model checkpoint if it has the best loss so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("Model weights saved!")

def load_data(train_input_path, train_output_path, test_input_path, test_output_path, batch_size):
    # Load the CSV files
    train_input = pd.read_csv(train_input_path).values
    train_output = pd.read_csv(train_output_path).values
    test_input = pd.read_csv(test_input_path).values
    test_output = pd.read_csv(test_output_path).values

    # Convert to PyTorch tensors
    train_input_tensor = torch.tensor(train_input, dtype=torch.float32)
    train_output_tensor = torch.tensor(train_output, dtype=torch.float32)
    test_input_tensor = torch.tensor(test_input, dtype=torch.float32)
    test_output_tensor = torch.tensor(test_output, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_input_tensor, train_output_tensor)
    test_dataset = TensorDataset(test_input_tensor, test_output_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage
input_size = 373
hidden_size = 512
output_size = 5
num_epochs = 50
learning_rate = 0.001
batch_size = 32
checkpoint_path = 'ann/model_weights/best_model_weights.pth'

train_input_path = 'dataset/dataset_normalized/train_input_normalized.csv'
train_output_path = 'dataset/dataset_normalized/train_output_normalized.csv'
test_input_path = 'dataset/dataset_normalized/test_input_normalized.csv'
test_output_path = 'dataset/dataset_normalized/test_output_normalized.csv'

# Load data
train_loader, test_loader = load_data(train_input_path, train_output_path, test_input_path, test_output_path, batch_size)

# Initialize model, criterion, and optimizer
model = NeuralNetwork(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, criterion, optimizer, train_loader, num_epochs, checkpoint_path)