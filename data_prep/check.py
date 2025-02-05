import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split 
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import pickle

# Ustawienia
data_dir = "A:\Polibuda\SEM_VII\Systemy cyber-fizyczne z uczeniem\data_prep\dataset"  # Ścieżka do oryginalnego datasetu
output_dir = "processed_dataset"  # Ścieżka do zapisu przetworzonych danych
IMG_SIZE = 128  # Rozmiar zdjęć (128x128)
AUGMENTATION_MULTIPLIER = 2  # Ilość augmentowanych danych na każdy oryginalny obraz

# Tworzenie folderu wyjściowego
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Eksploracja danych
def get_class_distribution(data_dir):
    classes = [cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))]
    class_distribution = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}
    return classes, class_distribution

# 2. Ładowanie i przetwarzanie obrazów
def load_and_preprocess_images(data_dir, img_size):
    images, labels = [], []
    for label, cls in enumerate(classes):
        folder_path = os.path.join(data_dir, cls)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))  # Zmiana rozmiaru
                img = img / 255.0  # Normalizacja
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# 3. Augmentacja danych
def augment_data(X, y, classes, multiplier, img_size):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    augmented_images, augmented_labels = [], []
    for label, cls in enumerate(classes):
        class_images = X[y == label]
        for img in class_images:
            augmented = datagen.flow(np.expand_dims(img, axis=0), batch_size=1)
            for _ in range(multiplier):
                aug_img = next(augmented)[0]  # Poprawione na next(augmented)
                augmented_images.append(aug_img)
                augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)

# 4. Zapisywanie przetworzonych danych do folderu
def save_processed_data(output_dir, X_train, y_train, X_val, y_val, X_test, y_test, classes):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "data_new.pkl"), "wb") as f:
        pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test, classes), f)

# 5. Wizualizacje
def plot_distribution(distribution, title, labels_rotation=45):
    plt.bar(distribution.keys(), distribution.values())
    plt.title(title)
    plt.xticks(rotation=labels_rotation)
    plt.show()

# --- Pipeline ---
# A. Eksploracja danych
classes, class_distribution = get_class_distribution(data_dir)
print(f"Znaleziono {len(classes)} klasy: {classes}")
plot_distribution(class_distribution, "Rozkład obrazów w klasach (przed augmentacją)")

# B. Ładowanie i normalizacja
X, y = load_and_preprocess_images(data_dir, IMG_SIZE)

# C. Augmentacja danych
aug_X, aug_y = augment_data(X, y, classes, AUGMENTATION_MULTIPLIER, IMG_SIZE)

# Łączenie oryginalnych i augmentowanych danych
X = np.concatenate((X, aug_X), axis=0)
y = np.concatenate((y, aug_y), axis=0)

# Nowy rozkład danych po augmentacji
augmented_distribution = {cls: sum(y == label) for label, cls in enumerate(classes)}
plot_distribution(augmented_distribution, "Rozkład obrazów w klasach (po augmentacji)")

# D. Podział danych
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Rozkład w zbiorach
sets_distribution = {
    "Treningowy": len(X_train),
    "Walidacyjny": len(X_val),
    "Testowy": len(X_test),
}
plot_distribution(sets_distribution, "Rozkład obrazów w zbiorach danych")

# E. Zapisywanie przetworzonych danych
save_processed_data(output_dir, X_train, y_train, X_val, y_val, X_test, y_test, classes)
print(f"Dane przetworzone zapisano w folderze: {output_dir}")
