import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pathlib
from matplotlib import image
from numpy import asarray

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

print(show_img)           # mostra informacoes gerais sobre a imagem
print(show_img.mode)      # mostra se a imagem e do tipo rgb, grayscale
print(show_img.format)    # mostra as dimensoes da imagem
show_img.show()           # mostra a imagem


# converte a imagem em um array
image_load = image.imread("../../sensix_daninhas/flowers/flower_photos/dandelion/8475758_4c861ab268_m.jpg")
#image_load = image_load / 255.0
print(image_load)
print("shape : ", image_load.shape)
print("type : ", image_load.dtype)
print("Max size : ", image_load.max())
print("Min Size : ", image_load.min())

# apresenta a imagem a partir do array
plt.imshow(image_load)
#plt.plot([1,2,3])
#plt.draw()
#plt.show(block=False)
plt.show()

# converte o array em um objeto image pyllow (objeto do tipo Image)
array2img = PIL.Image.fromarray(image_load)
#print(array2img)
print("array2img", type(array2img))

# convertendo um objeto do tipo imagem para um array numpy
# carregando a imagem
img2np = PIL.Image.open("../../sensix_daninhas/flowers/flower_photos/dandelion/8475758_4c861ab268_m.jpg")
# metodo asrray que realiza a conversao
array2np = asarray(img2np)
print("type : ", array2np.dtype)
print("shape : ", array2np.shape)
#image.save("../../sensix_daninhas/flowers/teste.png", format = "PNG")

# converte em escala de cinza
#image_cinza = image.convert(mode  = "L")
#print("shape : ", image_cinza.shape)
#plt.show()

# converte a imagem nas dimensoes citadas, porem mantem a proporcao
img2np.thumbnail((200,200))
print("image thumbnail", img2np.size)
# converte a imagem nas dimensoes citadas sem se preocupar com a proporcao
image_resize = img2np.resize((200,200))
print("image resize", image_resize.size)

# inversao horizontal
horizontal_image = img2np.transpose(PIL.Image.FLIP_LEFT_RIGHT)
plt.imshow(horizontal_image)
plt.show()

# rotacionando imagem
image_rot = PIL.Image.open("../../sensix_daninhas/flowers/flower_photos/dandelion/8475758_4c861ab268_m.jpg")
plt.imshow(image_rot.rotate(45))
plt.show()

# cortando imagens em partes especificas
plt.imshow(img2np.crop((50, 50, 100, 100)))
plt.show()

# efetuacao da normalizacao
image_load = image.imread("../../sensix_daninhas/flowers/flower_photos/dandelion/8475758_4c861ab268_m.jpg")
image_pillow = PIL.Image.open("../../sensix_daninhas/flowers/flower_photos/dandelion/8475758_4c861ab268_m.jpg")
# imread ja carrega e converte em array
print(image_load)
# PIL.Image.open carrega a imagem como um objeto do tipo imagem
print(image_pillow)

# convertendo em array
pixel_np = asarray(image_pillow)
print("Type : ", pixel_np.dtype)
print("Max : ", pixel_np.max(), ",", "Min : ", pixel_np.min())

pixel_np = pixel_np.astype('float32')
pixel_np /= pixel_np.max()
print("Max : ", pixel_np.max(), ",", "Min : ", pixel_np.min())

