import tkinter as tk
import shutil
from typing import Text
import numpy as np
import os
import pathlib
import PIL
from numpy.core.defchararray import title
from numpy.lib.npyio import load

import imgaug as ia
import cv2

from functools import partial
from tkinter import messagebox as mbox
from tkinter.constants import S
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
from osgeo import gdal,ogr,osr
from keras_segmentation.predict import predict
from keras.preprocessing import image
from tensorflow.python.keras.backend import print_tensor
from tensorflow.python.keras.preprocessing.image import img_to_array

from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.metrics import jaccard_score

os.environ["SM_FRAMEWORK"] = "tf.keras" 
import segmentation_models as sm

BACKBONE = 'vgg16'
model = sm.Linknet(backbone_name=BACKBONE, encoder_weights='imagenet', encoder_freeze=True, classes=1, activation='sigmoid', weights = '../pericles_examples/jocival/vgg16_Linknet_Test24.hdf5')

class Interface(tk.Frame):

    def __init__(self, root):
        tk.Frame.__init__(self, root)
                
        self.width_size = 600
        self.hight_size = 600

        menubar = tk.Menu(self)
        fileMenu = tk.Menu(self)
        root.maxsize(self.width_size, self.hight_size) 
        root.resizable(False,False)

        self.x_crop = 0
        self.y_crop = 0
        self.iterator_x = 400
        self.iterator_y = 300
        self.cnt_validator = []
        self.background_percent = 0.8

        self.f = {"Back":"0", "Next":"1"}
        self.first_click = False
        self.change_button = {}

        self.var = tk.IntVar()
        self.old_choose = '' 
        self.OptionList = ["Test Neural Network", 
                           "Generate Shape from RGB Tif", 
                           "Generate Shape from Binary Tif", 
                           "Generate a Binary Tif from RGB Tif",
                           "Compare Results"] 

        self.event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))
        menubar.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_cascade(label="Draw", command=self.select_image)
        fileMenu.add_cascade(label="Choose Neural Network", command=self.select_image)

    def create_buttons(self):
        #Botao para Selecionar uma Imagem
        root.maxsize(700, 700) 

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

        #Painel Superior Esquerdo
        self.painel_up_left = tk.Label(root)
        self.painel_up_left.place(x=20, y=50)

        #Painel Superior Direito
        self.painel_up_right = tk.Label(root)
        self.painel_up_right.place(x=320, y=50)

        #Painel Inferior Esquerdo 
        self.painel_down_left = tk.Label(root)
        self.painel_down_left.place(x=20, y=366)

        #Painel Inferior Direito
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
        self.painel_down_right.image = image_tk

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
        self.opt.place(x=self.width_size*0.35, y=self.hight_size*0.05)

        self.labelTest = tk.Label(text="", font=('Helvetica', 12), fg='red')
        self.labelTest.pack(side="top")

        self.variable.trace("w", self.callback_opt)

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

            root.maxsize(self.width_size, self.hight_size) 
            if self.load_rgb_tif()[0]:
                self.generate_binary_tif()
                self.generate_shape()

            else:
                mbox.showerror('Error', 'Nenhum Ortomosaico.tif foi Selecionado :')
                tif_loaded = False
        
        elif(self.variable.get() == 'Generate Shape from Binary Tif'):
            root.maxsize(self.width_size, self.hight_size) 
            if self.load_rgb_tif()[0]:
                self.generate_shape()
            else:
                mbox.showerror('Error', 'Nenhum Ortomosaico.tif foi Selecionado :')
                tif_loaded = False

        elif(self.variable.get() == 'Compare Results'):

            root.maxsize(self.width_size, self.hight_size)
            [unused, self.name_reference_binary, self.name_reference_neural] = self.load_shp(2)
            #self.reference_binary = self.shp_to_bin(name_reference_binary)
            self.name_tif = self.load_rgb_tif()

            #self.reference_binary =
            print('name_tif :', self.name_tif[1])
            print('ref_binary :', self.name_reference_binary)
            print('ref_neural', self.name_reference_neural)

            self.reference_binary = gdal.Open(self.shp_to_bin(self.name_reference_binary, self.name_tif[1]))
            self.reference_neural = gdal.Open(self.shp_to_bin(self.name_reference_neural, self.name_tif[1]))

            print(self.reference_binary)
            print(self.reference_neural)

            if self.name_tif[0]:

                button_left = tk.Button(root, text="Back")
                button_left.place(x=self.width_size*0.04, y=self.hight_size*0.5)
                button_left.bind("<Button-1>", partial(self.button_click, key="0"))

                button_right = tk.Button(root, text="Next")
                button_right.place(x=self.width_size*0.83, y=self.hight_size*0.5)
                button_right.bind("<Button-1>", partial(self.button_click, key="1"))
                
            else:
                mbox.showerror('Error', 'Nenhum Ortomosaico.tif foi Selecionado :')
                tif_loaded = False
                
        self.old_choose = self.variable.get()
             
    def button_click(self, event=None, key=None):
        
        self.cnt_validator = []
        self.painel_center = tk.Label(root)
        self.painel_center.place(x=self.width_size*0.15, y=self.hight_size*0.2)
         #setting up a tkinter canvas with scrollbars
        
        frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        xscroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        xscroll.grid(row=1, column=0, sticky=tk.E+tk.W)
        yscroll = tk.Scrollbar(frame)
        self.canvas = tk.Canvas(root, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        #self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

        if (key == "1"):
    
            if (self.x_crop+self.iterator_x < self.mosaico.RasterXSize):
                self.x_crop += self.iterator_x

            if (self.x_crop+self.iterator_x > self.mosaico.RasterYSize):
                self.x_crop =0
                self.y_crop += self.iterator_y

        elif (key == "0"):

            if (self.x_crop+self.iterator_x < self.mosaico.RasterXSize):
                self.x_crop -= self.iterator_x

            if (self.x_crop+self.iterator_x > self.mosaico.RasterYSize):
                self.x_crop =0
                self.y_crop -= self.iterator_y

        blueparcela = self.blue.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        greenparcela = self.green.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        redparcela = self.red.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        self.imgparcela = cv2.merge((blueparcela, greenparcela, redparcela))

        img_neural = self.reference_neural.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        img_binary   = self.reference_binary.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        union, dif = self.diff_contourns(img_neural, img_binary)
    
        self.contours = self.find_contourns(dif)
        self.draw = cv2.drawContours(self.imgparcela, self.contours, -1, (255, 0, 0), 3)
        
        img = PIL.Image.fromarray(self.draw)
        image_tk = ImageTk.PhotoImage(img)
        
        self.painel_center.configure(image=image_tk)
        self.painel_center.image=image_tk  

        self.first_click = True      
        self.painel_center.bind("<ButtonPress-1>",self.printcoords)
        
        return key

    def find_contourns(self, img):
        
        dots            = cv2.GaussianBlur(img, (21, 21), 0)
        dots_cpy        = cv2.erode(dots, (3, 3))
        #dots_cpy        = cv2.dilate(dots, None, iterations=1)
        filter          = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
        contours, hier  = cv2.findContours(filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(len(contours))    
        for idx, c in enumerate(contours):  # numbers the contours
            self.x_ctn = int(sum(c[:,0,0]) / len(c))
            self.y_ctn = int(sum(c[:,0,1]) / len(c))
            #print(' x :', x,' y :', y,' c :', c)

        return contours

    def diff_contourns(self, img_reference, img_neural):

        img_neural = cv2.threshold(img_neural, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        img_reference = cv2.threshold(img_reference, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        union = np.logical_or(img_neural, img_reference)
        union = union.astype(np.uint8)*255
        union[union < 128] = 0
        union[union > 100] = 255

        dif = cv2.subtract(union, img_neural)
        dif[dif < 128] = 0
        dif[dif > 100] = 255

        return union, dif

    def printcoords(self, event):

        cx, cy = self.event2canvas(event, self.canvas)
        self.ctn = []
        if self.first_click == True: 

            for i in range(0, len(self.contours)):
                self.cnt_validator.append(False)
            
            print("False")
            self.first_click = False

        for i in range(0, len(self.cnt_validator)):   
            r = cv2.pointPolygonTest(self.contours[i], (cx, cy), False)
            #print(r)
            if r > 0:
                self.cnt_validator[i] = (not self.cnt_validator[i])    
                print("Selected contour ", i)   
                self.ctn = self.contours[i]

                if self.cnt_validator[i] == True:
                    self.draw = cv2.drawContours(self.imgparcela, self.ctn, -1, (0, 255, 0), 3)

                else:
                    self.draw = cv2.drawContours(self.imgparcela, self.ctn, -1, (255, 0, 0), 3)

        print('validator :', self.cnt_validator)

        img = PIL.Image.fromarray(self.draw)
        image_tk = ImageTk.PhotoImage(img)

        self.canvas.create_image(0,0,image=image_tk,anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
        self.painel_center.configure(image=image_tk)
        self.painel_center.image=image_tk

    def load_rgb_tif(self):

        path_rgb_shp = filedialog.askopenfilename(title='Selecione O Mosaico')
        if path_rgb_shp.endswith('tif'):

            self.mosaico = gdal.Open(path_rgb_shp)
            print(path_rgb_shp)
            self.red = self.mosaico.GetRasterBand(1)
            self.green = self.mosaico.GetRasterBand(2)
            self.blue = self.mosaico.GetRasterBand(3)
            self.alpha = self.mosaico.GetRasterBand(4)

            self.nx = self.mosaico.RasterXSize   
            self.ny = self.mosaico.RasterYSize

            file_path = pathlib.Path(path_rgb_shp)
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
        
        return tif_loaded, path_rgb_shp

    def load_shp(self, type_shape=0, option='Compare Results'):
        """
        Carrega o shp que sera utilido nas comparacoes
        type : 0 - Representa o shape de referência, 
               1 - Representa o shape da rede neural
               2 - selecao do 2 shapes simultaneos
        option : Refere-se ao tipo de operacao a ser executada
        """
        if option == 'Compare Results':
            if type_shape == 0:
                path_reference_shp = filedialog.askopenfilename(title='Selecione o Shape de Referência :')
                path_reference_tif = self.load_rgb_tif()[1]

                return path_reference_tif, path_reference_shp, None 

            elif type_shape == 1:
                path_neural_shp = filedialog.askopenfilename(title='Selecione o Shape da Rede Neural :')

                return None, None, path_neural_shp 

            elif type_shape == 2:
                path_reference_shp = filedialog.askopenfilename(title='Selecione o Shape de Referência :')
                #path_reference_tif = self.load_rgb_tif()[1]
                path_neural_shp = filedialog.askopenfilename(title='Selecione o Shape da Rede Neural :')

                return None, path_reference_shp, path_neural_shp

        #if path_reference_shp.endswith('shp') and path_neural_shp.endswith('shp'):

    def shp_to_bin(self, name_shp, name_tif, burn=255):

        base_img = gdal.Open(name_tif, gdal.GA_ReadOnly)
        base_shp = ogr.Open(name_shp)
        base_shp_layer = base_shp.GetLayer()

        #output_name = name_shp + '_out.tif
        output = gdal.GetDriverByName('GTiff').Create(name_shp + '_out.tif', base_img.RasterXSize, base_img.RasterYSize, 1, gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
        output.SetProjection(base_img.GetProjectionRef())
        output.SetGeoTransform(base_img.GetGeoTransform()) 

        Band = output.GetRasterBand(1)
        raster = gdal.RasterizeLayer(output, [1], base_shp_layer, burn_values=[burn])

        Band = None
        output = None
        base_img = None
        base_shp = None

        return name_shp + '_out.tif'

    def generate_binary_tif(self): 
        mbox.showinfo("Information", "Gerando Resultados: Isso pode demorar um pouco: ")
        iterator_x = 256
        iterator_y = 256

        for x in range(0, self.mosaico.RasterXSize, iterator_x):

            for y in range(0, self.mosaico.RasterYSize, iterator_y):
                    
                if ((x+iterator_x)>self.mosaico.RasterXSize) or ((y+iterator_y)>self.mosaico.RasterYSize):
                    continue
                    
                blueparcela = self.blue.ReadAsArray(x,y,iterator_x,iterator_y)
                greenparcela = self.green.ReadAsArray(x,y,iterator_x,iterator_y)
                redparcela = self.red.ReadAsArray(x,y,iterator_x,iterator_y)
                self.imgparcela = cv2.merge((blueparcela,greenparcela,redparcela))
                img = self.imgparcela / 255

                pr = model.predict(np.array([img]))[0]
                pr = pr[:,:, 0]
                pr[pr >= 0.1] = 255
                pr[pr < 0.5] = 0
                pr = pr.astype('uint8')
                
                if (self.imgparcela.max()>0) and (self.imgparcela.min()<255):
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
    root.geometry("800x800+400+400")
    Exit1 = tk.Button(root, text="Sair", command=root.destroy)
    Exit1.pack(side='bottom')
    root.mainloop()