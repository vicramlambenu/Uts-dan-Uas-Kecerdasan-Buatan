import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# Step 1: Preprocessing
def create_dataset():
    """
    Membuat dataset sederhana untuk klasifikasi buah berdasarkan ciri-cirinya.
    Data berisi ciri-ciri buah seperti panjang, warna, dan tekstur.
    """
    features = [
        ['panjang', 'kuning', 'halus'],    # Pisang
        ['bulat', 'merah', 'halus'],      # Apel
        ['lonjong', 'hijau', 'kasar'],    # Mangga
        ['bulat', 'hijau', 'kasar'],      # Semangka
        ['lonjong', 'kuning', 'kasar'],   # Nangka
        ['bulat', 'hijau', 'halus'],      # Melon
        ['panjang', 'kuning', 'kasar'],   # Pisang Mentah
        ['lonjong', 'kuning', 'halus'],   # Mangga Matang
        ['bulat', 'merah', 'kasar'],      # Tomat
        ['bulat', 'kuning', 'halus'],     # Jeruk
        ['panjang', 'hijau', 'kasar'],    # Pisang Mentah
        ['lonjong', 'hijau', 'kasar'],    # Mangga Mentah
        ['panjang', 'kuning', 'halus'],   # Pisang
        ['bulat', 'merah', 'halus'],      # Apel
        ['lonjong', 'hijau', 'kasar'],    # Mangga
        ['bulat', 'kuning', 'halus'],     # Jeruk
        ['bulat', 'merah', 'halus'],      # Apel
        ['panjang', 'kuning', 'halus'],   # Pisang
        ['bulat', 'kuning', 'halus'],     # Jeruk
        ['lonjong', 'kuning', 'halus'],   # Mangga
    ]

    labels = [
        'Pisang', 'Apel', 'Mangga', 'Semangka', 'Nangka', 'Melon', 'Pisang', 'Mangga', 'Tomat', 'Jeruk',
        'Pisang', 'Mangga', 'Pisang', 'Apel', 'Mangga', 'Jeruk', 'Apel', 'Pisang', 'Jeruk', 'Mangga'
    ]

    # Periksa distribusi kelas
    print("Distribusi awal kelas:", Counter(labels))

    # Tambahkan data untuk kelas dengan anggota kurang dari 2
    while any(count < 2 for count in Counter(labels).values()):
        for label, count in Counter(labels).items():
            if count < 2:
                idx = labels.index(label)
                features.append(features[idx])  # Duplikasi fitur
                labels.append(label)  # Duplikasi label

    print("Distribusi setelah penambahan data:", Counter(labels))
    return features, labels

# Step 2: Preprocessing for Model
def preprocess_data(features, labels):
    """
    Mengubah data fitur menjadi numerik menggunakan encoding manual.
    """
    feature_map = {
        'panjang': 0, 'bulat': 1, 'lonjong': 2,
        'kuning': 0, 'merah': 1, 'hijau': 2,
        'halus': 0, 'kasar': 1
    }

    encoded_features = np.array([[feature_map[feat] for feat in sample] for sample in features])
    return encoded_features, np.array(labels)

# Step 3: Train KNN Model
def train_knn(X_train, y_train, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

# Step 4: Predict Fruits from User Input
def predict_fruit_from_input(model, feature_map):
    print("Masukkan ciri-ciri buah yang ingin Anda prediksi")
    bentuk = input("Bentuk (panjang/bulat/lonjong): ")
    warna = input("Warna (kuning/merah/hijau): ")
    tekstur = input("Tekstur (halus/kasar): ")

    # Encode fitur berdasarkan peta fitur
    encoded_input = np.array([feature_map[bentuk], feature_map[warna], feature_map[tekstur]]).reshape(1, -1)

    # Prediksi menggunakan model
    prediction = model.predict(encoded_input)
    print(f"Prediksi buah berdasarkan ciri-ciri yang dimasukkan adalah: {prediction[0]}")

# Main Program
if __name__ == "__main__":
    # Load dataset
    features, labels = create_dataset()
    encoded_features, encoded_labels = preprocess_data(features, labels)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_features, encoded_labels, test_size=0.3, random_state=42, stratify=encoded_labels
    )

    # Train model
    knn_model = train_knn(X_train, y_train, k=3)

    # Evaluate model
    y_pred = knn_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Test with user input
    feature_map = {
        'panjang': 0, 'bulat': 1, 'lonjong': 2,
        'kuning': 0, 'merah': 1, 'hijau': 2,
        'halus': 0, 'kasar': 1
    }

    # Prediksi berdasarkan input pengguna
    predict_fruit_from_input(knn_model, feature_map)
