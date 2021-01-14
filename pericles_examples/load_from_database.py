import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pathlib

# base de dados das flores
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#dataset_url = "../flowers"

# get_file é um metodo que baixa imagens via link, oringin = link, fname e nome do diretorio principal, untar descompacta
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=False)
# acessa o diretorio
data_dir = pathlib.Path(data_dir)

# conta o numero de imagens no diretorio principal
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# faz uma lista das imagens no diretorio /roses
roses = list(data_dir.glob('roses/*'))

# relaciona cada inteiro a uma das imagens
show_img = PIL.Image.open(str(roses[0]))
show_img.show()
