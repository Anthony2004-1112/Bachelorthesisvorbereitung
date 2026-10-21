import random
import numpy as np
import torch
import torch.nn as nn

# Utility to set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, loader, criterion, optimizer, scheduler,
                num_epochs, device, max_grad_norm=1.0):
    model.to(device)
    best_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Automatically handle both 2D one-hot vectors and 1D integer targets
            if targets.dim() > 1 and targets.size(1) > 1:
                y_idx = targets.argmax(dim=1)
            else:
                y_idx = targets.view(-1).long()

            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, y_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # --- checkpointing logic ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # load best weights back into model before returning
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return model

# Define model class
def get_model(input_length: int = 2800, num_classes: int = 10, input_channels: int = 1):
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv1d(input_channels, 16, kernel_size=31, padding=15),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=31, padding=15),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            )
            self.conv3 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=31, padding=15),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            )
            conv_output_length = input_length // 8
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * conv_output_length, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.fc(x)
            
    return CNN()

# Evaluation function
def eval_model(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Automatically handle both 2D one-hot vectors and 1D integer targets
            if targets.dim() > 1 and targets.size(1) > 1:
                y_idx = targets.argmax(dim=1)
            else:
                y_idx = targets.view(-1).long()

            out = model(inputs)
            preds = out.argmax(dim=1)
            correct += (preds == y_idx).sum().item()
            total += targets.size(0)
            
    return 100. * correct / total
