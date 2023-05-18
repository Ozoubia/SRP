import numpy as np
from torch.utils.data import DataLoader
from torch_loader import TimeSeriesDataset
from preprocessing import Preprocessing
from model import BERTSeg
import torch
import torch.nn as nn
import torch.optim as optim

data = Preprocessing(data_name="mHealth", boundary_ratio=0.1)
tr, tt = data.generate_long_time_series()
X_long_tr, y_long_tr, y_seg_long_tr, file_boundaries_tr = tr
X_long_tt, y_long_tt, y_seg_long_tt, file_boundaries_tt = tt
input_length = 512
seed = 0

mask_tr = np.ones(len(y_long_tr))
mask_tt = np.ones(len(y_long_tt))

ts_tr  = TimeSeriesDataset(input_length, X_long_tr, y_long_tr, y_seg_long_tr, mask_tr, file_boundaries_tr, seed)
ts_tt = TimeSeriesDataset(input_length, X_long_tt, y_long_tt, y_seg_long_tt, mask_tt, file_boundaries_tt, seed)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate your model
model = BERTSeg(num_classes=12, num_features=23).to(device)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define your training and validation data loaders
train_loader = DataLoader(ts_tr, 32, shuffle=False)
val_loader = DataLoader(ts_tt, 32, shuffle=False)

# Define the number of training epochs
num_epochs = 1000
best_val_loss = float('inf')
best_model_state = None
patience = 3
counter = 0

# Training loop
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, _, _ in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        outputs.logits = outputs.logits.view(-1, 12)
        labels = labels.view(-1)
        loss = criterion(outputs.logits, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        predicted = torch.argmax(outputs.logits,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss += loss.item()

    train_accuracy = correct / total
    training_loss = train_loss / len(ts_tr)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            # Forward pass
            outputs = model(inputs)
            outputs.logits = outputs.logits.view(-1, 12)
            labels = labels.view(-1)
            loss = criterion(outputs.logits, labels)

            # Calculate validation accuracy
            predicted = torch.argmax(outputs.logits,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_loss += loss.item()

    val_accuracy = correct / total
    validation_loss = val_loss / len(ts_tt)

    # Print training and validation results for each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Check for early stopping
    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        best_model_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break