import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Definicja modelu CNN
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Dostosowanie wymiarów
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Aktualizacja wymiarów
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)  # Ustawienie wymiarów na 8x8
        x = x.view(x.size(0), -1)  # Spłaszczenie
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Funkcja treningowa
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Zerowanie gradientów
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass i aktualizacja wag
            loss.backward()
            optimizer.step()
            
            # Statystyki
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")
        
        # Walidacja modelu
        val_acc = validate_model(model, val_loader, criterion, device)
        best_val_acc = max(best_val_acc, val_acc)
    
    return best_val_acc

# Funkcja walidacyjna
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_acc = 100 * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")
    return val_acc

# Funkcja testowania modelu
def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

# Funkcja do testowania różnych optymalizatorów
def test_optimizers(optimizers, model_class, train_loader, val_loader, test_loader, num_classes, criterion, num_epochs, device):
    results = {}
    for optimizer_name, optimizer_fn in optimizers.items():
        print(f"\nTesting optimizer: {optimizer_name}")
        
        # Inicjalizacja modelu
        model = model_class(num_classes).to(device)
        
        # Inicjalizacja optymalizatora
        optimizer = optimizer_fn(model.parameters())
        
        # Trening i walidacja
        best_val_acc = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
        
        # Testowanie modelu
        y_pred, y_true = test_model(model, test_loader, device)
        test_acc = 100 * (y_pred == y_true).sum() / len(y_true)
        results[optimizer_name] = {
            "best_val_acc": best_val_acc,
            "test_acc": test_acc
        }
        print(f"Test Accuracy with {optimizer_name}: {test_acc:.2f}%")
    return results

# Funkcja wizualizacji wyników
def plot_results(results):
    optimizer_names = list(results.keys())
    val_accuracies = [metrics["best_val_acc"] for metrics in results.values()]
    test_accuracies = [metrics["test_acc"] for metrics in results.values()]

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(optimizer_names))

    plt.bar(index, val_accuracies, bar_width, label="Validation Accuracy")
    plt.bar([i + bar_width for i in index], test_accuracies, bar_width, label="Test Accuracy", alpha=0.7)

    plt.xlabel("Optimizers")
    plt.ylabel("Accuracy (%)")
    plt.title("Optimizer Performance Comparison")
    plt.xticks([i + bar_width / 2 for i in index], optimizer_names)
    plt.legend()
    plt.show()

# Wczytanie danych
pkl_file_path = "processed_dataset\\data_new.pkl"
with open(pkl_file_path, "rb") as f:
    X_train, y_train, X_val, y_val, X_test, y_test, classes = pickle.load(f)

# Przygotowanie danych
train_dataset = TensorDataset(torch.tensor(X_train).permute(0, 3, 1, 2).float(), torch.tensor(y_train).long())
val_dataset = TensorDataset(torch.tensor(X_val).permute(0, 3, 1, 2).float(), torch.tensor(y_val).long())
test_dataset = TensorDataset(torch.tensor(X_test).permute(0, 3, 1, 2).float(), torch.tensor(y_test).long())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Konfiguracja eksperymentu
optimizers = {
    "Adam": lambda params: optim.Adam(params, lr=0.001),
    "SGD": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
    "Adamax": lambda params: optim.Adamax(params, lr=0.002),
    "RMSprop": lambda params: optim.RMSprop(params, lr=0.001),
}

# Parametry treningu
num_epochs = 10
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testowanie optymalizatorów
results = test_optimizers(optimizers, CNNModel, train_loader, val_loader, test_loader, len(classes), criterion, num_epochs, device)

# Wizualizacja wyników
plot_results(results)

# Wyniki końcowe
print("\nFinal Results:")
for optimizer_name, metrics in results.items():
    print(f"{optimizer_name} -> Best Validation Accuracy: {metrics['best_val_acc']:.2f}%, Test Accuracy: {metrics['test_acc']:.2f}%")
