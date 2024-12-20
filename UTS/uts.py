import os
import numpy as np
import cv2 as cv
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt

TrainPath = r'C:\Users\asus\Documents\Uts dan Uas Kecerdasan Buatan\UTS\Train1'
categories = [
    'TTD AvavSam',
    'TTD FAHRIL ANTONIO HANDE',
    'TTD Lyra Attallah Aurellia',
    'TTD MARVELLOUS DEMETRIUS MAIT',
    'TTD MUHAMMAD AL-FARABY MOIDADY',
    'TTD Qhiran',
    'TTD SOPI',
    'TTD Vicram',
    'TTD_aqilah',
    'TTD_NOVITA',
    'TTD BRIANT_CJ',
    'TTD Fitri Handayani',
    'TTD MAHRECZY ADITYA',
    'TTD MOHAMMAD RAIYAN',
    'TTD NAtsya labaso',
    'TTD RIZKA FILARDI TOLIZ',
    'TTD SUPARMAN',
    'TTD Yuyun',
    'TTD_Hasby Ashidiq',
    'TTD Fadhil Akmal Zakaria',
    'TTD Fransisca',
    'TTD MARSYA CIKITA',
    'TTD MUH HALIIM',
    'TTD PUTRI CASIOLA',
    'TTD Siti Nurvatika',
    'TTD VIAAAAAAAAA',
    'TTD Zaky Putra',
    'TTD_Muh. Mashaq Ramadhan. M'
]

data = []

for category in categories:
    label = categories.index(category)
    path = os.path.join(TrainPath, category)
    for img in os.listdir(path):
        imgPath = os.path.join(path, img)
        try:
            imgArray = cv.imread(imgPath)
            imgResized = cv.resize(imgArray, (50, 50))
            image = np.array(imgResized).flatten()
            data.append([image, label])
        except Exception as e:
            pass

with open('dataImage.pickle', 'wb') as pick:
    pickle.dump(data, pick)

with open('dataImage.pickle', 'rb') as pick:
    data = pickle.load(pick)

features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

features = np.array(features)
labels = np.array(labels)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.1)

model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(xtrain, ytrain)

pick = open('model.sav', 'wb')
pickle.dump(model, pick)
pick.close()

pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()

image_path = r'C:\Users\asus\Documents\Uts dan Uas Kecerdasan Buatan\UTS\Test1\fatir_test.jpg'

imgArray = cv.imread(image_path)

imgResized = cv.resize(imgArray, (50, 50))

image_flatten = np.array(imgResized).flatten()

prediction = model.predict([image_flatten])

categories = [
    'TTD AvavSam',
    'TTD FAHRIL ANTONIO HANDE',
    'TTD Lyra Attallah Aurellia',
    'TTD MARVELLOUS DEMETRIUS MAIT',
    'TTD MUHAMMAD AL-FARABY MOIDADY',
    'TTD Qhiran',
    'TTD SOPI',
    'TTD Vicram',
    'TTD_aqilah',
    'TTD_NOVITA',
    'TTD BRIANT_CJ',
    'TTD Fitri Handayani',
    'TTD MAHRECZY ADITYA',
    'TTD MOHAMMAD RAIYAN',
    'TTD NAtsya labaso',
    'TTD RIZKA FILARDI TOLIZ',
    'TTD SUPARMAN',
    'TTD Yuyun',
    'TTD_Hasby Ashidiq',
    'TTD Fadhil Akmal Zakaria',
    'TTD Fransisca',
    'TTD MARSYA CIKITA',
    'TTD MUH HALIIM',
    'TTD PUTRI CASIOLA',
    'TTD Siti Nurvatika',
    'TTD VIAAAAAAAAA',
    'TTD Zaky Putra',
    'TTD_Muh. Mashaq Ramadhan. M'
]

print("Prediksi gambar: ", categories[prediction[0]])

accuracy = model.score(xtest, ytest)
print("Model accuracy: ", accuracy)

imgResized = imgResized.reshape(50, 50, 3)
plt.imshow(imgResized, cmap='gray')
plt.show()
