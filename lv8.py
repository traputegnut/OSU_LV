#8.4.1 
import numpy as np
from tensorflow import keras
from keras import layers 
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

plt.figure()
plt.imshow(x_train[0])
plt.show()
print(y_train[0])


# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

x_train_2 = x_train_s.reshape(60000, 784)
x_test_2 = x_test_s.reshape(10000, 784)


model = keras.Sequential()
model.add(layers.Dense(100, activation="relu", input_shape=(784, )))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",])



history = model.fit(x_train_2, y_train_s, batch_size=32, epochs=20, validation_split=0.1)


score = model.evaluate(x_test_2, y_test_s, verbose=0)
print(score)
y_p = model.predict(x_test_2)
cm = confusion_matrix(y_test_s.argmax(axis=1), y_p.argmax(axis=1))
print(cm)



model.save("FCN/")


#8.4.2
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

model = keras.models.load_model("FCN/")
model.summary()

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# skaliranje slike na raspon [0,1]
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_test_s = np.expand_dims(x_test_s, -1)

y_test_s = keras.utils.to_categorical(y_test, num_classes)

x_test_2 = x_test_s.reshape(10000, 784)

y_p = model.predict(x_test_2)

y_test_s_2 = y_test_s.argmax(axis=1)
y_p_2 = y_p.argmax(axis=1)