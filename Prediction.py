import cv2
import tensorflow as tf
import numpy as np


# method to predict class

def image_classification(new_image):
    model = tf.keras.models.load_model('classification_model_5.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    img = cv2.imread(new_image)
    img = cv2.resize(img, (150, 150))
    img = np.reshape(img, [1, 150, 150, 3])

    classes = model.predict_classes(img)

    print(classes)
    return classes
