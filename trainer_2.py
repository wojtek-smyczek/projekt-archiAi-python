import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. zaladowanie i przygotowanie danych
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# 2. budowa i trenowanie modelu (jesli nie istnieje)
if not os.path.exists('handwritten.model'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)
    model.save('handwritten.model')
else:
    # 3. ladowanie instniejacego modelu
    model = tf.keras.models.load_model('handwritten.model')

# 4. ewaluacja modelu na zbiorze testowym
loss, accuracy = model.evaluate(x_test, y_test)
print(f"strata: {loss}")
print(f"dokladnosc: {accuracy}")

# 5. przewidywanie na wlasnych obrazach z folderu 'digits'
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        # wczytanie obrazu w skali szarosci
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        # zmiana rozmiaru do 28x28 (wymagane przez model)
        img = cv2.resize(img, (28, 28))
        # inwersja kolorow (model mnist uzywa bialych cyfr na czarnym tle)
        img = np.invert(np.array([img]))
        
        prediction = model.predict(img)
        print(f"ta cyfra to prawdopodobnie: {np.argmax(prediction)}")
        
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"blad: {e}")
    finally:
        image_number += 1