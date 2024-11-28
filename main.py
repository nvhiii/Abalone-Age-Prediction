import torch
import torch.nn as nn
import torch.optim as optim
from model import AbaloneNet
from data_loader import load_data

def train(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            predictions = model(X).squeeze()
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # eval
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                predictions = model(X).squeeze()
                val_loss += criterion(predictions, y).item()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# eval using MSE (mean square error and/or mean square error)
def evaluate(model, test_loader, criterion):
    """
    Evaluate the model on the test set.

    :param model: The trained PyTorch model.
    :param test_loader: DataLoader for the test set.
    :param criterion: Loss function.
    """
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            predictions = model(X).squeeze()
            loss = criterion(predictions, y)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"RMSE: {torch.sqrt(torch.tensor(test_loss / len(test_loader))):.4f}")

def main():
    train_loader, val_loader, test_loader = load_data()
    
    # Initialize the model, loss function, and optimizer
    input_size = next(iter(train_loader))[0].shape[1]
    model = AbaloneNet(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    train(model, train_loader, val_loader, criterion, optimizer, epochs=20)

    # line for eval
    evaluate(model, test_loader, criterion)

if __name__ == "__main__":
    main() # test