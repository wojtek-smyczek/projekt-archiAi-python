import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

def train_professional_model(data_dir='my_dataset', save_path='model_wojtka.pth'):
    # 1. Konfiguracja sprzętowa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używam urządzenia: {device}")

    # 2. Augmentacja - klucz do eliminacji błędów 1 vs 2
    # Dodajemy losowe rotacje, pochylenia i zmiany jasności
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(15),           # Rotacja +/- 15 stopni
        transforms.RandomAffine(0, shear=10),    # Pochylenie (shear)
        transforms.ColorJitter(brightness=0.2),  # Odporność na oświetlenie iPada
        transforms.ToTensor(),
    ])

    # 3. Ładowanie i podział danych (80% trening, 20% test)
    if not os.path.exists(data_dir):
        print(f"BŁĄD: Folder {data_dir} nie istnieje! Najpierw przygotuj zdjęcia.")
        return

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 4. Definicja modelu z Dropoutem
    # ResNet18 zmodyfikowany pod Twoje potrzeby
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),      # Warstwa Dropout - zapobiega uczeniu się na pamięć
        nn.Linear(num_ftrs, 10)
    )
    model.to(device)

    # 5. Optymalizator i funkcja straty
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    # 6. Pętla treningowa
    num_epochs = 40 # Zwiększamy liczbę epok dla lepszej stabilności przy Dropout
    best_acc = 0.0

    print(f"Rozpoczynam trening na {len(full_dataset)} obrazkach...")

    for epoch in range(num_epochs):
        model.train() # Włącza Dropout
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Ewaluacja na zbiorze testowym
        model.eval() # Wyłącza Dropout dla testów
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        
        # Zapisujemy model tylko jeśli jest lepszy niż poprzedni rekord
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), save_path)
            status = "[REKORD - ZAPISANO]"
        else:
            status = ""

        print(f"Epoka {epoch+1}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {accuracy:.2f}% {status}")

    print(f"\nTrening zakończony! Najlepsza celność: {best_acc:.2f}%")
    print(f"Model zapisany jako: {save_path}")

if __name__ == "__main__":
    train_professional_model()