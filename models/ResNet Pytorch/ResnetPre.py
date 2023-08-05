import torch.nn as nn
import torch
from torchmetrics import Accuracy
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from ESC50Dataset import ESC50Dataset
from ESC50DataModule import ESCDataModule

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

num_classes = 50
model.fc = nn.Sequential(
    nn.BatchNorm1d(2048),
    nn.Dropout(p=0.5),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, num_classes)
)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
accuracy = Accuracy(task="multiclass", num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

# Load your training data and set up data loaders (omitted for brevity)
dataset = ESC50Dataset(None)
data = dataset.read_csv_as_dict()
data_module = ESCDataModule(data, num_folds=5)
data_module.setup()
data_module.current_fold = 1
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            running_acc = accuracy(predicted, labels)
            running_acc += running_acc
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    acc_avg = running_acc / len(val_loader)
    validation_accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {validation_accuracy:.4f} - Val AVG: {acc_avg}")

# Save the fine-tuned model (optional)
torch.save(model.state_dict(), "resnet50_pre2.pth")
