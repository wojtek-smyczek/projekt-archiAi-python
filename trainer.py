import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler, Subset
import os
import random

def train_professional_model(data_dir='my_dataset', save_path='model_wojtka.pth'):
    # 1. Konfiguracja sprzętowa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używam urządzenia: {device}")

    # 2. Augmentacja — wzmocniona wersja
    user_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(25),
        transforms.RandomAffine(0, shear=15, scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
    ])

    # Transform dla MNIST (28x28 → 64x64, białe-na-czarnym jak user data)
    mnist_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(25),
        transforms.RandomAffine(0, shear=15, scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
    ])

    # Transform do ewaluacji (bez augmentacji)
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    eval_mnist_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # 3. Ładowanie danych użytkownika
    if not os.path.exists(data_dir):
        print(f"BŁĄD: Folder {data_dir} nie istnieje! Najpierw przygotuj zdjęcia.")
        return

    user_dataset_train = datasets.ImageFolder(data_dir, transform=user_transform)
    user_dataset_eval = datasets.ImageFolder(data_dir, transform=eval_transform)

    # Podział danych użytkownika (80% trening, 20% test)
    user_train_size = int(0.8 * len(user_dataset_train))
    user_test_size = len(user_dataset_train) - user_train_size
    generator = torch.Generator().manual_seed(42)
    user_train_indices, user_test_indices = random_split(
        range(len(user_dataset_train)), [user_train_size, user_test_size], generator=generator
    )
    user_train = Subset(user_dataset_train, user_train_indices.indices)
    user_test = Subset(user_dataset_eval, user_test_indices.indices)

    print(f"Dane użytkownika: {len(user_train)} trening, {len(user_test)} test")

    # 4. Ładowanie MNIST (podzbiór 5000 próbek)
    print("Pobieranie MNIST...")
    mnist_full = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=mnist_transform)
    mnist_eval_full = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=eval_mnist_transform)

    random.seed(42)
    mnist_indices = random.sample(range(len(mnist_full)), 5000)
    mnist_train = Subset(mnist_full, mnist_indices)
    mnist_eval = Subset(mnist_eval_full, mnist_indices[:1000])

    print(f"MNIST: {len(mnist_train)} próbek (podzbiór z 60000)")

    # 5. Definicja modelu
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 10)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 6. Test loader (tylko dane użytkownika — to jest nasz cel)
    test_loader = DataLoader(user_test, batch_size=16, shuffle=False)

    # =============================================
    # FAZA 1: Pretrain na MNIST + dane użytkownika
    # =============================================
    print("\n=== FAZA 1: Pretrain (MNIST + dane użytkownika) ===")

    combined_dataset = ConcatDataset([user_train, mnist_train])

    # WeightedRandomSampler: dane użytkownika 5x ważniejsze
    weights = []
    user_weight = 5.0
    mnist_weight = 1.0
    for i in range(len(combined_dataset)):
        if i < len(user_train):
            weights.append(user_weight)
        else:
            weights.append(mnist_weight)

    sampler = WeightedRandomSampler(weights, num_samples=len(combined_dataset), replacement=True)
    combined_loader = DataLoader(combined_dataset, batch_size=16, sampler=sampler)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    best_acc = 0.0
    phase1_epochs = 15

    for epoch in range(phase1_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in combined_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Ewaluacja na danych użytkownika
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), save_path)
            status = "[REKORD - ZAPISANO]"
        else:
            status = ""

        print(f"[F1] Epoka {epoch+1}/{phase1_epochs} | Loss: {running_loss/len(combined_loader):.4f} | Accuracy: {accuracy:.2f}% {status}")

    print(f"Faza 1 zakończona. Najlepsza celność: {best_acc:.2f}%")

    # =============================================
    # FAZA 2: Finetune TYLKO na danych użytkownika
    # =============================================
    print("\n=== FAZA 2: Finetune (tylko dane użytkownika) ===")

    # Załaduj najlepszy model z fazy 1
    model.load_state_dict(torch.load(save_path, map_location=device))

    user_train_loader = DataLoader(user_train, batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    phase2_epochs = 25

    for epoch in range(phase2_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in user_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), save_path)
            status = "[REKORD - ZAPISANO]"
        else:
            status = ""

        print(f"[F2] Epoka {epoch+1}/{phase2_epochs} | Loss: {running_loss/len(user_train_loader):.4f} | Accuracy: {accuracy:.2f}% {status}")

    print(f"\nTrening zakończony! Najlepsza celność: {best_acc:.2f}%")
    print(f"Model zapisany jako: {save_path}")

if __name__ == "__main__":
    train_professional_model()
