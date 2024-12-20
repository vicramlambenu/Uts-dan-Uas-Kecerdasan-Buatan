import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# Fungsi untuk membaca gambar dan labelnya
def load_images_and_labels(base_folder):
    images = []
    labels = []
    label_names = []

    for label, subfolder in enumerate(os.listdir(base_folder)):
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path):
            label_names.append(subfolder)
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                if file_path.endswith(('.png', '.jpg', '.jpeg')):
                    # Baca gambar
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (128, 128))  # Resize gambar
                    # Ekstraksi fitur menggunakan HOG
                    features, _ = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
                    images.append(features)
                    labels.append(label)

    return np.array(images), np.array(labels), label_names

# Load dataset
base_folder = r'C:\Users\asus\Documents\Uts dan Uas Kecerdasan Buatan\UTS\Train1'
X, y, label_names = load_images_and_labels(base_folder)

# Split dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Evaluasi akurasi
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy * 100:.2f}%")

# Fungsi untuk prediksi tanda tangan baru
def predict_signature(image_path, model, label_names):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    features, _ = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    features = features.reshape(1, -1)  # Ubah menjadi array 2D untuk prediksi
    
    prediction = model.predict(features)
    predicted_label = label_names[prediction[0]]
    
    # Estimasi probabilitas
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities) * 100
    return predicted_label, confidence

# Contoh penggunaan
input_image = r"C:\Users\asus\Documents\Uts dan Uas Kecerdasan Buatan\UTS\Test1/qhiran1.png"
img = cv2.imread(input_image)


predicted_label, confidence = predict_signature(input_image, svm_model, label_names)
print(f"Tanda tangan terdeteksi sebagai: {predicted_label} dengan kepercayaan {confidence:.2f}%")
cv2.imshow(predicted_label, img)
# if confidence < 30.0:
#         print("Tanda tangan tidak cocok dengan data yang ada")
#         cv2.imshow("Anonim", img)
# else:    
#     print(f"Tanda tangan terdeteksi sebagai: {predicted_label} dengan kepercayaan {confidence:.2f}%")
#     cv2.imshow(predicted_label, img)


cv2.waitKey(0)
cv2.destroyAllWindows()
