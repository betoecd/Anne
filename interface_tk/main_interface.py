import tkinter as tk
from tkinter import messagebox as mbox
import tkinter

from tkinter.constants import S
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import PIL
import cv2
import numpy as np
from osgeo import gdal,ogr,osr
import shutil
import segmentation_models as sm
import imgaug as ia
from keras_segmentation.predict import predict
from keras.preprocessing import image
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
from skimage.transform import radon, rescale, rotate
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star, binary_closing, closing, opening, skeletonize, erosion, dilation)
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.measure import label, regionprops, regionprops_table
from skimage.io import imread, imsave
import math
import pandas as pd
import numpy as np
import glob
import os
import pathlib

from sklearn.metrics import jaccard_score
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star, binary_closing, closing, opening, skeletonize, erosion, dilation)
from skimage.transform import rescale, resize, downscale_local_mean
import time

from tensorflow.python.keras.preprocessing.image import img_to_array

BACKBONE = 'vgg16'
model = sm.Linknet(backbone_name=BACKBONE, encoder_weights='imagenet', encoder_freeze=True, classes=1, activation='sigmoid', weights = '../pericles_examples/jocival/vgg16_Linknet_Test28.hdf5')

class Interface(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root)
        menubar = tk.Menu(self)
        fileMenu = tk.Menu(self)
        root.maxsize(400, 400) 
        root.resizable(False,False)

        self.path_shp = ''
        self.old_choose = '' 
        self.OptionList = ["Test Neural Network", 
                           "Generate Shape from RGB Tif", 
                           "Generate Shape from Binary Tif", 
                           "Generate a Binary Tif from RGB Tif"] 

        menubar.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_cascade(label="Draw", command=self.select_image)
        fileMenu.add_cascade(label="Choose Neural Network", command=self.select_image)

        #root.configure(menu=menubar)

        #Botao de Sair
        Exit1 = tk.Button(root, text="Sair", command=root.destroy)
        Exit1.pack(side='bottom')

        #root.geometry("800x800+200+200")

    def create_buttons(self):
        #Botao para Selecionar uma Imagem
        root.maxsize(800, 800) 

        self.btn_selet_image = tk.Button(root, text="Select an image", command=self.select_image)
        self.btn_selet_image.place(x=50, y=10)

        #Botao para Segmentacao
        self.btn_segmentation = tk.Button(root, text="Apply Segmentation", command=self.predict_image)
        self.btn_segmentation.place(x=400, y=10)

        #Botao para escolher a marcara de referencia
        self.btn_mask_true = tk.Button(root, text="Select Mask", command=self.select_true_binary)
        self.btn_mask_true.place(x=50, y=326)

        #Botao para analisar as diferencas entre a predicao e a mascara
        self.btn_diff_imgs = tk.Button(root, text="Differences", command=self.diff_imgs)
        self.btn_diff_imgs.place(x=400, y=326)

        #brightup = tk.Button(root, text="RotLeft", command="buttonpressed")
        #brightdown = tk.Button(root, text="RotLeft", command="buttonpressed")

        #Paineis para Exibicao

        #Painei Superior Esquerdo
        self.painel_up_left = tk.Label(root)
        self.painel_up_left.place(x=20, y=50)

        #Painei Superior Direito
        self.painel_up_right = tk.Label(root)
        self.painel_up_right.place(x=320, y=50)

        #Painei Inferior Esquerdo 
        self.painel_down_left = tk.Label(root)
        self.painel_down_left.place(x=20, y=366)

        #Painei Inferior Direito
        self.painel_down_right = tk.Label(root)
        self.painel_down_right.place(x=320, y=366)

    def remove_buttons(self):

        self.btn_selet_image.destroy()
        self.btn_segmentation.destroy()
        self.btn_mask_true.destroy()
        self.btn_diff_imgs.destroy()

        self.painel_up_left.destroy()
        self.painel_up_right.destroy()
        self.painel_down_left.destroy()
        self.painel_down_right.destroy()

    def predict_image(self):
        
        img = image.load_img(self.path_rgb, target_size=(256,256))
        img = image.img_to_array(img)
        img = img / 255

        pr = model.predict(np.array([img]))[0]
        pr = pr[:,:, 0]
        pr[pr >= 0.1] = 1
        pr[pr < 0.5] = 0
        pr = pr.astype('uint8')

        pr[pr == 1] = 255
        self.img_pred = pr
        pr = PIL.Image.fromarray(pr)
        image_tk = ImageTk.PhotoImage(pr)

        self.painel_up_right.configure(image=image_tk)
        self.painel_up_right.image = image_tk

    def diff_imgs(self):

        dif_img =  cv2.subtract(self.img_true, self.img_pred)
        dif_img = PIL.Image.fromarray(dif_img)
        image_tk = ImageTk.PhotoImage(dif_img)

        pred = self.iou(self.img_true, self.img_pred)
        pred = round(pred, 3)

        w = tk.Label(root, text='Jaccard Index : ' +  str(pred))
        w.place(x=340, y=640)

        self.painel_down_right.configure(image=image_tk)
        self.painel_down_right.image = image_tk,

    def iou(self, prediction, target):

        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def select_true_binary(self):
        
        self.path_true = filedialog.askopenfilename()
        if self.path_true:
        
            self.img_true = cv2.imread(self.path_true, cv2.COLOR_BGR2RGB)
            self.img_true = cv2.resize(self.img_true, (256,256))

            image = Image.fromarray(self.img_true)
            image_tk = ImageTk.PhotoImage(image)

            self.painel_down_left.configure(image=image_tk)
            self.painel_down_left.image = image_tk

    def select_image(self):

        self.path_rgb = filedialog.askopenfilename()
        if self.path_rgb:
        
            self.img_rgb = cv2.imread(self.path_rgb, -1)
            self.img_rgb = cv2.resize(self.img_rgb, (256,256))

            image = Image.fromarray(self.img_rgb)
            image_tk = ImageTk.PhotoImage(image)

            self.painel_up_left.configure(image=image_tk)
            self.painel_up_left.image = image_tk

    def choose_opt(self, app):

        self.variable = tk.StringVar(app)
        self.variable.set('Choose a Option')

        self.opt = tk.OptionMenu(app, self.variable, *self.OptionList)
        #opt.config(width=90, font=('Helvetica', 12))
        self.opt.place(x=20, y=50)

        self.labelTest = tk.Label(text="", font=('Helvetica', 12), fg='red')
        self.labelTest.pack(side="top")

        self.variable.trace("w", self.callback_opt)

        #Painei Inferior Esquerdo 
        #self.painel_down_left = tk.Label(root)
        #self.painel_down_left.place(x=20, y=366)
    
    def callback_opt(self, *args):

        #self.labelTest.configure(text="The selected item is {}".format(self.variable.get()))
        if (self.old_choose == 'Test Neural Network' and self.variable.get() != 'Test Neural Network'):
            self.remove_buttons()

        if(self.variable.get() == 'Test Neural Network'):
            self.labelTest.destroy()
            self.opt.destroy()
            print('removendo')
            self.create_buttons()

        elif(self.variable.get() == 'Generate Shape from RGB Tif'):
            root.maxsize(400, 400) 
            if self.load_tif():
                self.generate_binary_tif()
                self.generate_shape()

            else:
                mbox.showerror('Error', 'Nenhum Ortomosaico.tif foi Selecionado :')
                tif_loaded = False
        
        elif(self.variable.get() == 'Generate Shape from Binary Tif'):
            root.maxsize(400, 400) 
            if self.load_tif():
                self.generate_shape()
            else:
                mbox.showerror('Error', 'Nenhum Ortomosaico.tif foi Selecionado :')
                tif_loaded = False
                
        self.old_choose = self.variable.get()
             
    def load_tif(self):

        path_shp = filedialog.askopenfilename()
        if path_shp.endswith('tif'):

            self.mosaico = gdal.Open(path_shp)
            print(path_shp)
            self.red = self.mosaico.GetRasterBand(1)
            self.green = self.mosaico.GetRasterBand(2)
            self.blue = self.mosaico.GetRasterBand(3)
            self.alpha = self.mosaico.GetRasterBand(4)

            self.nx = self.mosaico.RasterXSize   
            self.ny = self.mosaico.RasterYSize

            file_path = pathlib.Path(path_shp)
            self.out_file = pathlib.Path('/')
            self.out_file = file_path.parent/("out" + ".shp")
            path_temp = file_path.parent/'temp_files'

            if path_temp.exists():
                shutil.rmtree(str(path_temp))
                os.mkdir(str(path_temp))
            else:
                os.mkdir(str(path_temp))
                
            self.dst_img = gdal.GetDriverByName('GTiff').Create(str(path_temp/'outfile.tif'), self.nx, self.ny, 1, gdal.GDT_Byte)
            self.dst_img.SetGeoTransform(self.mosaico.GetGeoTransform())
            self.srs = osr.SpatialReference()
            self.srs.ImportFromWkt(self.mosaico.GetProjection())
            self.dst_img.SetProjection(self.srs.ExportToWkt())
            tif_loaded = True
        
        else:
            mbox.showerror('Error', 'Selecione um Arquivo .tif')
            tif_loaded = False
        
        return tif_loaded
        
    def generate_binary_tif(self): 
        mbox.showinfo("Information", "Geradondo Resultados: Isso pode demorar um pouco: ")
        iterator_x = 256
        iterator_y = 256

        for x in range(0, self.mosaico.RasterXSize, iterator_x):
            for y in range(0, self.mosaico.RasterYSize, iterator_y):
                    
                if ((x+iterator_x)>self.mosaico.RasterXSize) or ((y+iterator_y)>self.mosaico.RasterYSize):
                    continue
                    
                blueparcela = self.blue.ReadAsArray(x,y,iterator_x,iterator_y)
                greenparcela = self.green.ReadAsArray(x,y,iterator_x,iterator_y)
                redparcela = self.red.ReadAsArray(x,y,iterator_x,iterator_y)
                imgparcela = cv2.merge((blueparcela,greenparcela,redparcela))
                img = imgparcela / 255

                pr = model.predict(np.array([img]))[0]
                pr = pr[:,:, 0]
                pr[pr >= 0.1] = 255
                pr[pr < 0.5] = 0
                pr = pr.astype('uint8')
                
                if (imgparcela.max()>0) and (imgparcela.min()<255):
                    write_image = pr

                else:
                    pr[pr>=255]= 0
                    write_image = pr
                    
                self.dst_img.GetRasterBand(1).WriteArray(write_image, xoff=x, yoff=y)
                self.dst_img.FlushCache()

    def generate_shape(self):
        src_band = self.dst_img.GetRasterBand(1)
        dst_layername = 'daninhas'
        drv = ogr.GetDriverByName("ESRI Shapefile")
        dst_ds = drv.CreateDataSource(str(self.out_file))
        dst_layer = dst_ds.CreateLayer(dst_layername, srs = self.srs)

        gdal.Polygonize(src_band, src_band, dst_layer, -1, [], callback=None )
        dst_ds.Destroy()
        mbox.showinfo("Information", "Shape Gerado com Sucesso!: ")

if __name__ == "__main__":

    root = tk.Tk()
    Interface(root).pack(fill="both", expand=True)
    root.title('Semantic Segmetation Tools')
    root.resizable(False,False)
    Interface(root).choose_opt(root)
    root.geometry("800x800+200+200")
    root.mainloop()