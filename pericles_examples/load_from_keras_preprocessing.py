import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pathlib
from tensorflow.keras import layers # redimensionar de 0 para 1

batch_size = 32  # >=9
img_height = 180
img_width = 180
data_dir_flowers = "../../sensix_daninhas/flowers/flower_photos"

# Contrucao do dataset a partir do preprocessing do keras, nessa contrucao é escolhido pelo
# proprio keras um valor de "validation_split=0.2" para a parte de validacao
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  directory = data_dir_flowers,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Contrucao do dataset a partir do preprocessing do keras 
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  directory = data_dir_flowers,
  validation_split=0.2,
  subset="validation",
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
