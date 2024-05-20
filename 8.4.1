import numpy as np
from tensorflow import keras
from tensorflow.keras import layers         #iako je podcrtano radi
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#1
print("Za učenje:", len(x_train))
print("Za testiranje:", len(x_test))
print("Primjer ulazne veličine:", x_train[0].shape)     #matrica 28x28
print("Primjer izlazne veličine:", y_train[0])          #broj prikazan na slici

#2
plt.imshow(x_train[0])
#plt.show()
print("Oznaka slike:", y_train[0])

#3
model = keras.Sequential()
model.add(layers.Input(shape=(784, )))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.summary()

#4
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",])

#5
x_train = np.reshape(x_train, (len(x_train), x_train.shape[1]*x_train.shape[2]))        #dobiva se br slika puta broj piksela u jednoj slici
x_test = np.reshape(x_test, (len(x_test), x_test.shape[1]*x_test.shape[2]))

ohe = OneHotEncoder()
y_train_ohe = ohe.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
y_test_ohe = ohe.fit_transform(np.reshape(y_test, (-1, 1))).toarray()

history = model.fit(x_train, y_train_ohe, batch_size=32, epochs=20, validation_split=0.1)

#6
score = model.evaluate(x_test, y_test_ohe, verbose=0)

#7
y_test_pred = model.predict(x_test)             #ovo zapravo ne vraca znamenke koje trebaju za confusion matrix
y_test_pred = np.argmax(y_test_pred, axis=1)    #znamenke su zapravo indeksi jedinog mjesta gdje pise 1 a ne 0

cm = confusion_matrix(y_test, y_test_pred)
print("Matrica zabune:", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred))
disp.plot()
plt.show()

#8
KERAS_MODEL_NAME = "Model/keras.hdf5"
keras.models.save_model(model, KERAS_MODEL_NAME)
del model
