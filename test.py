import tensorflow as tf
import matplotlib.pyplot as plt


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print(tf.__version__)
x = tf.constant(4, shape=(1,1), dtype=tf.float32)
print(x)


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255
test_images = test_images / 255
print('train_images.shape:', train_images.shape)
print('train_labels.shape:', train_labels.shape)

fig, ax = plt.subplots(nrows=1, ncols=10)
for i in range(10):
    ax[i].imshow(train_images[i], cmap='gray')
    ax[i].title.set_text(train_labels[i])
plt.show()


#   Build model
m = tf.keras.models.Sequential()
m.add(tf.keras.layers.Flatten(input_shape=(28,28)))
m.add(tf.keras.layers.Dense(128, activation='relu'))
m.add(tf.keras.layers.Dense(10, activation='softmax'))
m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
m.fit(train_images, train_labels, epochs=2)
loss, acc = m.evaluate(test_images, test_labels)
print('loss: ', loss)
print('accuracy:', acc)