import tkinter as tk
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import PIL
import cv2
import numpy as np

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
from sklearn.metrics import jaccard_score
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star, binary_closing, closing, opening, skeletonize, erosion, dilation)
from skimage.transform import rescale, resize, downscale_local_mean
import time

from tensorflow.python.keras.preprocessing.image import img_to_array

BACKBONE = 'vgg16'
model = sm.Linknet(backbone_name=BACKBONE, encoder_weights='imagenet', encoder_freeze=True, classes=1, activation='sigmoid', weights = '../pericles_examples/jocival/vgg16_Linknet_Test21.hdf5')

class Interface(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root)
        menubar = tk.Menu(self)
        fileMenu = tk.Menu(self)
        recentMenu = tk.Menu(self)
        
        Exit1 = tk.Button(root, text="Sair", command=root.destroy)
        Exit1.pack(side='bottom')

        self.btn_selet_image = tk.Button(root, text="Select an image", command=self.select_image)
        self.btn_selet_image.place(x=50, y=10)

        self.btn_segmentation = tk.Button(root, text="Apply Segmentation", command=self.predict_image)
        self.btn_segmentation.place(x=400, y=10)

        self.btn_mask_true = tk.Button(root, text="Select Mask", command=self.select_true_binary)
        self.btn_mask_true.place(x=50, y=326)

        self.btn_diff_imgs = tk.Button(root, text="Differences", command=self.diff_imgs)
        self.btn_diff_imgs.place(x=400, y=326)

        #brightup = tk.Button(root, text="RotLeft", command="buttonpressed")
        #brightdown = tk.Button(root, text="RotLeft", command="buttonpressed")

        self.painelSupEqs = tk.Label(root)
        self.painelSupEqs.place(x=20, y=50)
        #painelSupEqs.pack(side="left", padx=10, pady=10)

        self.painelSupDir = tk.Label(root)
        self.painelSupDir.place(x=320, y=50)
        #painelSupDir.pack(side="right", padx=10, pady=100)

        self.painelInfEqs = tk.Label(root)
        self.painelInfEqs.place(x=20, y=366)

        self.painelInfDir = tk.Label(root)
        self.painelInfDir.place(x=320, y=366)

        root.maxsize(600, 700) 
        root.resizable(False,False)
        menubar.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_cascade(label="Draw", command=self.select_image)
        fileMenu.add_cascade(label="Choose Neural Network", command=self.select_image)

        #for name in ("file1.txt", "file2.txt", "file3.txt"):
        #    recentMenu.add_command(label=name)

        root.configure(menu=menubar)
        root.geometry("800x800+200+200")

    def predict_image(self):
        global path_rgb, img_pred
        
        img = image.load_img(path_rgb, target_size=(256,256))
        img = image.img_to_array(img)
        img = img / 255
        #ia.imshow(img2)
        pr = model.predict(np.array([img]))[0]
        pr = pr[:,:, 0]
        pr[pr >= 0.1] = 1
        pr[pr < 0.5] = 0
        pr = pr.astype('uint8')
        #ia.imshow(pr)
        pr[pr == 1] = 255
        img_pred = pr
        pr = PIL.Image.fromarray(pr)
        image_tk = ImageTk.PhotoImage(pr)

        self.painelSupDir.configure(image=image_tk)
        self.painelSupDir.image = image_tk

    def diff_imgs(self):
        global path_true, path_rgb

        """ 
        img_true = cv2.imread(path_true, cv2.IMREAD_COLOR)
        img_rgb = cv2.imread(path_rgb, cv2.IMREAD_COLOR)

        img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)    
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        img_true = cv2.resize(img_true, (256,256))
        img_rgb = cv2.resize(img_rgb, (256,256))

        (thresh, img_true) = cv2.threshold(img_true, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        (thresh, img_rgb) = cv2.threshold(img_rgb, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        """

        dif_img =  cv2.subtract(img_true, img_pred)
        dif_img = PIL.Image.fromarray(dif_img)
        image_tk = ImageTk.PhotoImage(dif_img)

        pred = self.iou(img_true, img_pred)
        pred = round(pred, 3)

        w = tk.Label(root, text='Jaccard Index : ' +  str(pred))
        w.place(x=340, y=640)

        self.painelInfDir.configure(image=image_tk)
        self.painelInfDir.image = image_tk,

    def iou(self, prediction, target):
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def select_true_binary(self):
        global img_true, path_true
        
        path_true = filedialog.askopenfilename()
        if path_true:
        
            img_true = cv2.imread(path_true, cv2.COLOR_BGR2RGB)
            img_true = cv2.resize(img_true, (256,256))
            #img_keras = image.load_img(path_true, target_size=(256,256))

            grayScale = ''
            
            image = Image.fromarray(img_true)
            image_tk = ImageTk.PhotoImage(image)

            """
            if vargray.get() == 1:
                grayScale = cv2.imread(path_true, 0)  
                grayScale_img = Image.fromarray(grayScale)
                grayScale_img_tk = ImageTk.PhotoImage(grayScale_img)

                painelSupDir.configure(image=grayScale_img_tk)
                painelSupDir.image = grayScale_img_tk 
            """
            self.painelInfEqs.configure(image=image_tk)
            self.painelInfEqs.image = image_tk


    def select_image(self):
        global img_rgb, path_rgb

        path_rgb = filedialog.askopenfilename()


        if path_rgb:
        
            img_rgb = cv2.imread(path_rgb, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (256,256))
            #img_keras = image.load_img(path_rgb, target_size=(256,256))

            grayScale = ''
            
            image = Image.fromarray(img_rgb)
            image_tk = ImageTk.PhotoImage(image)

            """
            if vargray.get() == 1:
                grayScale = cv2.imread(path_rgb, 0)  
                grayScale_img = Image.fromarray(grayScale)
                grayScale_img_tk = ImageTk.PhotoImage(grayScale_img)

                painelSupDir.configure(image=grayScale_img_tk)
                painelSupDir.image = grayScale_img_tk 
            """
            self.painelSupEqs.configure(image=image_tk)
            self.painelSupEqs.image = image_tk

 
if __name__ == "__main__":
    root = tk.Tk()
    Interface(root).pack(fill="both", expand=True)
    root.title('Semantic Segmetation Tools')  # window title is toolbox
    root.geometry("800x800+200+200")
    root.resizable(False,False)
    #Interface(root).pack(fill="both", expand=False)
    root.mainloop()