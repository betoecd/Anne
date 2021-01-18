import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pathlib
import datetime
from keras_preprocessing import image
from tensorflow.keras import layers # redimensionar de 0 para 1
from tensorflow.keras.models import Sequential

batch_size = 32  # >=9
img_height = 100
img_width = 100
data_dir_train = "../../sensix_daninhas/dataset_100x100/train"
data_dir_test = "../../sensix_daninhas/dataset_100x100/validation"


# Contrucao do dataset a partir do preprocessing do keras, nessa contrucao é escolhido pelo
# proprio keras um valor de "validation_split=0.2" para a parte de validacao
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  directory = data_dir_train,
  #validation_split=0.2,
  #subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Contrucao do dataset a partir do preprocessing do keras 
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  directory = data_dir_test,
  #validation_split=0.99,
  #subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Recebendo o nome das classes
class_names = val_ds.class_names
print(class_names)

# Plot de algumas imagens
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
plt.show()

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 2

model = tf.keras.models.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.summary()

epochs=400

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[tensorboard_callback]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


path = '../../sensix_daninhas/dataset_redimensionated/validation/daninha_21.jpg'
img = image.load_img(path, target_size=(100, 100))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if (classes[0][0]>classes[0][1]):
    print(path + " daninha")
else:
    print(path + " sem_daninha")


path = '../../sensix_daninhas/dataset_redimensionated/validation/sem_daninha_192.jpg'
img = image.load_img(path, target_size=(100, 100))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if (classes[0][0]>classes[0][1]):
    print(path + " daninha")
else:
    print(path + " sem_daninha")

path = '../../sensix_daninhas/daninhas_duvidosas/daninha_1279-7.jpg'
img = image.load_img(path, target_size=(100, 100))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if (classes[0][0]>classes[0][1]):
    print(path + " daninha")
else:
    print(path + " sem_daninha")
 

model.save("daninhas_preprocessing.h5")