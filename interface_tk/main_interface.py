
import sys
from numpy.core.numeric import count_nonzero
import tkinter as tk
import shutil
from typing import Text
import numpy as np
import os
import pathlib
import PIL
from numpy.core.defchararray import asarray, join, title
from numpy.lib.npyio import load

import imgaug as ia
import cv2

from functools import partial
from tkinter import BooleanVar, Event, PhotoImage, Widget, messagebox as mbox
from tkinter.constants import ROUND, S
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
from osgeo import gdal,ogr,osr
from neural import NeuralFunctions as nf

class Interface(tk.Frame):

    def __init__(self, root):

        tk.Frame.__init__(self, root)
        menubar = tk.Menu(self)
        fileMenu = tk.Menu(self)
        
        self.width_size = 800
        self.hight_size = 600
        self.x_crop = 0
        self.y_crop = 0
        self.iterator_x = 256
        self.iterator_y = 256
        self.iterator_recoil = 0.8
        self.cnt_validator = []
        self.background_percent = 0.8
        self.array_clicks = []
        self.save_draw_array = None
        self.name_tif = ''
        self.name_reference_binary = ''
        self.name_reference_neural = ''
        self.slider_pencil = 10
        self.path_save_img_rgb = 'dataset/rgb'
        self.path_save_img_bin = 'dataset/binario'
        root.maxsize(self.width_size, self.hight_size) 
        root.resizable(False,False)

        self.f = {"Back":"0", "Next":"1"}
        self.first_click = False
        self.first_click_bool = False
        self.ready_start = False

        self.change_button = {}
        self.bool_value = tk.StringVar() # Necessario ser como string para funcionar
        self.spn_box_1 = tk.StringVar()
        self.spn_box_2 = tk.StringVar()
        self.spn_box_3 = tk.StringVar()
        self.current_value = tk.DoubleVar()
        self.btn_int = tk.IntVar()

        self.var = tk.IntVar()
        self.old_choose = '' 
        self.OptionList = ["Compare Results",
                           "Draw Weeds"] 

        self.event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y ))
        menubar.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_cascade(label="Draw", command=self.select_image)
        fileMenu.add_cascade(label="Choose Neural Network", command=self.select_image)

    def start(self):
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        self.style = tk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        root.geometry("527x505+400+200")
        root.minsize(1, 1)
        root.maxsize(1351, 738)
        root.resizable(1,  1)
        root.title("Comparador de Contornos")
        root.configure(highlightcolor="black")

        self.TSeparator1 = tk.Separator(root)
        self.TSeparator1.place(relx=0.021, rely=0.535,  relwidth=0.962)

        self.menubar = tk.Menu(root,font="TkMenuFont",bg=_bgcolor,fg=_fgcolor)
        root.configure(menu = self.menubar)

        self.Scale1 = tk.Scale(root, from_=0.0, to=100.0)
        self.Scale1.place(relx=0.417, rely=0.044, relheight=0.5
                , relwidth=0.173)
        self.Scale1.configure(orient="horizontal")
        self.Scale1.configure(troughcolor="#000000")

        self.btn_load_mosaico = tk.Button(root)
        self.btn_load_mosaico.place(relx=0.74, rely=0.572, height=28, width=123)
        self.btn_load_mosaico.configure(takefocus="")
        self.btn_load_mosaico.configure(text='Mosaico')
        self.btn_load_mosaico.bind('<Button-1>', partial(self.get_btn, key="0"))

        self.btn_shape_reference = tk.Button(root)
        self.btn_shape_reference.place(relx=0.74, rely=0.659, height=28, width=123)
        self.btn_shape_reference.configure(takefocus="")
        self.btn_shape_reference.configure(text='Shape de Refer')
        self.btn_shape_reference.bind('<Button-1>', partial(self.get_btn, key="1"))

        self.btn_shape_neural = tk.Button(root)
        self.btn_shape_neural.place(relx=0.74, rely=0.747, height=28, width=123)
        self.btn_shape_neural.configure(takefocus="")
        self.btn_shape_neural.configure(text='Shape da Rede')
        self.btn_shape_neural.bind('<Button-1>', partial(self.get_btn, key="2"))

        self.btn_start = tk.Button(root)
        self.btn_start.place(relx=0.742, rely=0.871, height=48, width=123)
        self.btn_start.configure(takefocus="")
        self.btn_start.configure(text='Iniciar')
        self.btn_start.bind('<Button-1>', partial(self.get_btn, key="5"))

        self.Spinbox1 = tk.Spinbox(root, from_=10.0, to=800.0, increment=10, textvariable=self.spn_box_1)
        self.Spinbox1.place(relx=0.74, rely=0.178, relheight=0.046, relwidth=0.243)
        self.Spinbox1.configure(activebackground="#f9f9f9")
        self.Spinbox1.configure(background="white")
        self.Spinbox1.configure(font="TkDefaultFont")
        self.Spinbox1.configure(highlightbackground="black")
        self.Spinbox1.configure(selectbackground="blue")
        self.Spinbox1.configure(selectforeground="white")
        self.Spinbox1.configure(command=self.get_values_spinbox)

        self.Spinbox2 = tk.Spinbox(root, from_=100.0, to=800.0, increment=100, textvariable=self.spn_box_2)
        self.Spinbox2.place(relx=0.74, rely=0.271, relheight=0.046
                , relwidth=0.243)
        self.Spinbox2.configure(activebackground="#f9f9f9")
        self.Spinbox2.configure(background="white")
        self.Spinbox2.configure(font="TkDefaultFont")
        self.Spinbox2.configure(highlightbackground="black")
        self.Spinbox2.configure(selectbackground="blue")
        self.Spinbox2.configure(selectforeground="white")
        self.Spinbox2.configure(command=self.get_values_spinbox)

        self.Spinbox3 = tk.Spinbox(root, from_=100.0, to=800.0, increment=100, textvariable=self.spn_box_3)
        self.Spinbox3.place(relx=0.74, rely=0.364, relheight=0.046, relwidth=0.243)
        self.Spinbox3.configure(activebackground="#f9f9f9")
        self.Spinbox3.configure(background="white")
        self.Spinbox3.configure(font="TkDefaultFont")
        self.Spinbox3.configure(highlightbackground="black")
        self.Spinbox3.configure(selectbackground="blue")
        self.Spinbox3.configure(selectforeground="white")
        self.Spinbox3.configure(command=self.get_values_spinbox)

        self.Label1 = tk.Label(root)
        self.Label1.place(relx=0.015, rely=0.178, height=21, width=245)
        self.Label1.configure(activebackground="#f9f9f9")
        self.Label1.configure(text='Escolha a porcentagem de iteração :')

        self.Radiobutton1 = tk.Radiobutton(root)
        self.Radiobutton1.place(relx=0.721, rely=0.455, relheight=0.046
                , relwidth=0.132)
        self.Radiobutton1.configure(justify='left')
        self.Radiobutton1.configure(text='Padrão', value=True, variable=self.bool_value, command=self.get_values_radio,)

        self.Label2 = tk.Label(root)
        self.Label2.place(relx=0.021, rely=0.269, height=21, width=235)
        self.Label2.configure(text='Valor do comprimento da imagem :')

        self.Label3 = tk.Label(root)
        self.Label3.place(relx=0.021, rely=0.36, height=21, width=186)
        self.Label3.configure(text='Valor de altura da imagem :')

        self.Label4 = tk.Label(root)
        self.Label4.place(relx=0.014, rely=0.446, height=21, width=186)
        self.Label4.configure(text='Usar configuração padrão :')

        self.Label5 = tk.Label(root)
        self.Label5.place(relx=0.023, rely=0.58, height=21, width=138)
        self.Label5.configure(text='Selecionar Mosaico :')

        self.Label6 = tk.Label(root)
        self.Label6.place(relx=0.019, rely=0.66, height=21, width=213)
        self.Label6.configure(text='Selecionar shape de referencia :')

        self.Label7 = tk.Label(root)
        self.Label7.place(relx=0.019, rely=0.74, height=38, width=221)
        self.Label7.configure(text='Selecionar shape da rede neural :')

    def labelling_start(self):
        root.minsize(1100, 700)
        #root.maxsize(1351, 738)
        root.resizable(1,  1)
        root.title("New rootlevel")

        self.value_label = tk.Label(root, text=self.get_current_value())

        self.Scale1 = tk.Scale(root, tickinterval=10, from_=0.0, to=100.0)
        self.Scale1.place(relx=0.60, rely=0.85, relheight=0.093
                , relwidth=0.271)
        self.Scale1.configure(length="251")
        self.Scale1.configure(orient="horizontal")
        self.Scale1.configure(troughcolor="#d9d9d9")

        self.Scale2 = tk.Scale(root, tickinterval=5, from_=10.0, to=50.0, command=self.slider_changed,   variable=self.current_value)
        self.Scale2.place(relx=0.132, rely=0.85, relheight=0.093, relwidth=0.271)
        self.Scale2.configure(length="249")
        self.Scale2.configure(orient="horizontal")
        self.Scale2.configure(troughcolor="#d9d9d9")

        self.next_icon = PhotoImage(file = r"next.png")
        self.button_right = tk.Button(root, image = self.next_icon)
        self.button_right.place(relx=0.926, rely=0.363, height=83, width=43)
        self.button_right.configure(borderwidth="2")
        #self.button_right.configure(text='>')

        self.back_icon = PhotoImage(file = r"back.png")
        self.button_left = tk.Button(root, image = self.back_icon)
        self.button_left.place(relx=0.031, rely=0.363, height=83, width=43)
        self.button_left.configure(borderwidth="2")
        self.button_left.configure(text='<')

        frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        xscroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        #xscroll.grid(row=1, column=0, sticky=tk.E+tk.W)
        yscroll = tk.Scrollbar(frame)
        #yscroll.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.canvas = tk.Canvas(frame, bd=0, width=self.iterator_x, height=self.iterator_y, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
            
        #self.canvas = tk.Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        #xscroll.config(command=self.canvas.xview)
        #yscroll.config(command=self.canvas.yview)
        frame.pack(expand=1)

        self.pencil_icon = PhotoImage(file = r"pencil.png")
        self.pencil_icon =  self.pencil_icon.subsample(1, 1)
        self.TButton1 = tk.Button(root, image = self.pencil_icon)
        self.TButton1.place(relx=0.132, rely=0.82, height=40, width=40)
        self.TButton1.configure(borderwidth="2")
        self.TButton1.bind("<Button-1>", partial(self.button_click, key="0"))

        self.transp_icon = PhotoImage(file = r"transparency.png")
        self.transp_icon =  self.transp_icon.subsample(2, 2)
        self.TButton2 = tk.Button(root, image=self.transp_icon)
        self.TButton2.place(relx=0.831, rely=0.82, height=40, width=40)
        # self.TButton2.configure(borderwidth="2")
        self.TButton2.configure(text='2')
        #self.TButton2.bind("<Button-1>", partial(self.button_click, key="0"))

        self.erase_icon = PhotoImage(file = r"erase.png")
        self.erase_icon =  self.erase_icon.subsample(2, 2)
        self.TButton3 = tk.Button(root, image=self.erase_icon)
        self.TButton3.place(relx=0.180, rely=0.82, height=40, width=40)
        #self.TButton3.configure(takefocus="")
        #self.TButton3.configure(text='3')

    def labelling_menu(self):
    
        print('Aqui')
        self.Label1.destroy()
        self.opt.destroy()

        self.Label1 = tk.Label(root)
        self.Label1.place(relx=0.115, rely=0.44, height=21, width=245)
        self.Label1.configure(activebackground="#f9f9f9")
        self.Label1.configure(text='Escolha a porcentagem de iteração :')

        self.btn_load_mosaico = tk.Button(root)
        self.btn_load_mosaico.place(relx=0.72, rely=0.44, height=28, width=123)
        self.btn_load_mosaico.configure(takefocus="")
        self.btn_load_mosaico.configure(text='Mosaico')
        self.btn_load_mosaico.bind('<Button-1>', partial(self.get_btn, key="0"))

        self.Label2 = tk.Label(root)
        self.Label2.place(relx=0.11, rely=0.52, height=21, width=200)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(text='Selecione o Shape de Base :')

        self.btn_shape_reference = tk.Button(root)
        self.btn_shape_reference.place(relx=0.72, rely=0.52, height=28, width=123)
        self.btn_shape_reference.configure(takefocus="")
        self.btn_shape_reference.configure(text='Shape de Base')
        self.btn_shape_reference.bind('<Button-1>', partial(self.get_btn, key="3"))

        self.Label3 = tk.Label(root)
        self.Label3.place(relx=0.117, rely=0.60, height=21, width=260)
        self.Label3.configure(activebackground="#f9f9f9")
        self.Label3.configure(text='Porcentagem de fundo preto permitida:')

        self.Spinbox_backg = tk.Spinbox(root, from_=5.0, to=100.0, increment=5, textvariable=self.spn_box_1)
        self.Spinbox_backg.place(relx=0.63, rely=0.60, relheight=0.046, relwidth=0.243)
        self.Spinbox_backg.configure(activebackground="#f9f9f9")
        self.Spinbox_backg.configure(background="white")
        self.Spinbox_backg.configure(font="TkDefaultFont")
        self.Spinbox_backg.configure(highlightbackground="black")
        self.Spinbox_backg.configure(selectbackground="blue")
        self.Spinbox_backg.configure(selectforeground="white")
        self.Spinbox_backg.configure(command=partial(self.get_values_spinbox, type='Draw Weeds'))

        self.btn_start = tk.Button(root)
        self.btn_start.place(relx=0.742, rely=0.871, height=48, width=123)
        self.btn_start.configure(takefocus="")
        self.btn_start.configure(text='Iniciar')
        self.btn_start.bind('<Button-1>', partial(self.get_btn, key="5"))

        self.Label4 = tk.Label(root)
        self.Label4.place(relx=0.117, rely=0.69, height=21, width=235)
        self.Label4.configure(text='Valor do comprimento da imagem :')

        self.Spinbox2 = tk.Spinbox(root, from_=256.0, to=800.0, increment=256, textvariable=self.spn_box_2)
        self.Spinbox2.place(relx=0.63, rely=0.69, relheight=0.046, relwidth=0.243)
        self.Spinbox2.configure(activebackground="#f9f9f9")
        self.Spinbox2.configure(background="white")
        self.Spinbox2.configure(font="TkDefaultFont")
        self.Spinbox2.configure(highlightbackground="black")
        self.Spinbox2.configure(selectbackground="blue")
        self.Spinbox2.configure(selectforeground="white")
        self.Spinbox2.configure(command=partial(self.get_values_spinbox, type='Draw Weeds'))

        self.Label5 = tk.Label(root)
        self.Label5.place(relx=0.117, rely=0.76, height=21, width=186)
        self.Label5.configure(text='Valor de altura da imagem :')

        self.Spinbox3 = tk.Spinbox(root, from_=256.0, to=800.0, increment=256, textvariable=self.spn_box_3)
        self.Spinbox3.place(relx=0.63, rely=0.76, relheight=0.046, relwidth=0.243)
        self.Spinbox3.configure(activebackground="#f9f9f9")
        self.Spinbox3.configure(background="white")
        self.Spinbox3.configure(font="TkDefaultFont")
        self.Spinbox3.configure(highlightbackground="black")
        self.Spinbox3.configure(selectbackground="blue")
        self.Spinbox3.configure(selectforeground="white")
        self.Spinbox3.configure(command=partial(self.get_values_spinbox, type='Draw Weeds'))

    def first_menu(self, app):

        self.logo = PhotoImage(file = r"images/Logo-Escuro.png")
        self.logo =  self.logo.subsample(5, 5)
        self.Canvas1 = tk.Canvas(root)
        self.Canvas1.place(relx=0.117, rely=0.111, relheight=0.291
                , relwidth=0.752)
        self.Canvas1.configure(borderwidth="2")
        self.Canvas1.configure(relief="ridge")
        self.Canvas1.configure(selectbackground="blue")
        self.Canvas1.configure(selectforeground="white")
        self.Canvas1.create_image(300,90,image=self.logo, anchor='center')

        self.Label1 = tk.Label(root)
        self.Label1.place(relx=0.25, rely=0.578, height=21, width=200)
        self.Label1.configure(text='Selecione uma opção : ')

        self.variable = tk.StringVar(app)
        self.variable.set('Choose a Option')

        self.opt = tk.OptionMenu(app, self.variable, *self.OptionList)
        #opt.config(width=90, font=('Helvetica', 12))
        self.opt.place(x=self.width_size*0.50, y=self.hight_size*0.57)

        self.labelTest = tk.Label(text="", font=('Helvetica', 12), fg='red')
        self.labelTest.pack(side="top")

        self.variable.trace("w", self.callback_opt)

    def remove_buttons(self, option='First Menu'):

        if option == 'Test Neural Network':
            self.btn_selet_image.destroy()
            self.btn_segmentation.destroy()
            self.btn_mask_true.destroy()
            self.btn_diff_imgs.destroy()

            self.painel_up_left.destroy()
            self.painel_up_right.destroy()
            self.painel_down_left.destroy()
            self.painel_down_right.destroy()

        elif option == 'Compare Results':
            self.TSeparator1.destroy()
            self.btn_load_mosaico.destroy()
            self.btn_shape_reference.destroy()
            self.btn_shape_neural.destroy()
            self.btn_start.destroy()
            self.Spinbox1.destroy()
            self.Spinbox2.destroy()                    
            self.Spinbox3.destroy()                    
            self.Label1.destroy()                      
            self.Radiobutton1.destroy()   
            self.Label2.destroy()
            self.Label3.destroy()                      
            self.Label4.destroy()
            self.Label5.destroy()
            self.Label6.destroy()
            self.Label7.destroy()

        elif option == 'Draw Menu':
            self.Label2.destroy()
            self.btn_shape_reference.destroy()
            self.Label3.destroy()
            self.Spinbox_backg.destroy()
            self.btn_start.destroy()
            self.btn_load_mosaico.destroy()

        else:
            
            self.Canvas1.destroy()
            self.Label1.destroy()

            self.opt.destroy()
            self.labelTest.destroy()

    def get_text(self):  
        text_val = self.entry_text.get()
        
        label_init = tk.Label(root, text=text_val)
        self.canvas_init.create_window(200, 230, window=label_init)

    def get_current_value(self):
        self.slider_pencil = self.current_value.get()
        if  self.slider_pencil < 10:
            self.slider_pencil = 10
        return '{: .2f}'.format(self.current_value.get())

    def slider_changed(self, event):
        self.value_label.configure(text=self.get_current_value())

    def get_values_spinbox(self, type=''):

        if type == 'Compare Results':

            if self.first_click_bool == False:
                self.iterator_recoil = float(int(self.Spinbox1.get())/100)
                self.iterator_x = int(self.Spinbox2.get())
                self.iterator_y = int(self.Spinbox3.get())

        elif type == 'Draw Weeds':
            self.background_percent = float(1-int(self.Spinbox_backg.get())/100)
            self.iterator_x = int(self.Spinbox2.get())
            self.iterator_y = int(self.Spinbox3.get())
            self.iterator_recoil = 1.0
            print(self.background_percent)
                    
        else:
            values1 = self.iterator_recoil * 100
            values2 = self.iterator_x
            values3 = self.iterator_y

    def get_values_radio(self):
        self.first_click_bool = not (self.first_click_bool)
        
        if self.first_click_bool:
            bool_default = bool(self.bool_value.get())
            self.spn_box_1.set('80')
            self.spn_box_2.set('500')
            self.spn_box_3.set('400')
            #self.bool_value.set(bool_default)

        else:
            bool_default = False
            self.bool_value.set(bool_default)
        
    def get_btn(self, event, key):
        self.event_btn = key
        if key=='0':
            self.name_tif = self.load_shp(0)[0]

        elif key=='1':
            self.name_reference_binary = self.load_shp(1)[1]
        
        elif key=='2':
            self.name_reference_neural = self.load_shp(2)[2]

        elif key=='3':
            self.name_reference_binary = self.load_shp(1)[1]

        elif self.name_tif != '' and self.name_reference_binary != '' and self.name_reference_neural != '' and key=='5':
            root.geometry("800x600+400+100")
            self.ready_start = True

        if self.name_tif != '' and self.name_reference_binary != '' and key=='5':
            self.remove_buttons('Draw Menu')
            self.reference_binary = self.shp_to_bin( self.name_reference_binary, self.name_tif)
            self.daninha_1 = gdal.Open(self.reference_binary)
            self.daninha_band_1 =  self.daninha_1.GetRasterBand(1)
            self.labelling_start()
            self.remove_buttons('Fisrt Menu')
            self.remove_buttons('Draw Menu')

            self.dst_img = gdal.GetDriverByName('GTiff').Create('resutado_gerado.tif', self.mosaico.RasterXSize, self.mosaico.RasterYSize, 1, gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
            self.dst_img.SetProjection(self.mosaico.GetProjectionRef())
            self.dst_img.SetGeoTransform(self.mosaico.GetGeoTransform()) 

            self.button_right.bind("<Button-1>", partial(self.button_click, key="1"))
            self.button_left.bind("<Button-1>", partial(self.button_click, key="0"))


        if self.ready_start:
            self.rm_btn()
            
            self.reference_binary = gdal.Open(self.shp_to_bin(self.name_reference_binary, self.name_tif), 1)
            self.reference_neural = gdal.Open(self.shp_to_bin(self.name_reference_neural, self.name_tif), 1)
            
            self.dst_img = gdal.GetDriverByName('GTiff').Create(self.name_reference_binary + '_out_2.tif', self.reference_binary.RasterXSize, self.reference_binary.RasterYSize, 1, gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
            self.dst_img.SetProjection(self.reference_binary.GetProjectionRef())
            self.dst_img.SetGeoTransform(self.reference_binary.GetGeoTransform()) 

            button_left = tk.Button(root, text="Back")
            button_left.place(relx=0.02, rely=0.4, height=48, width=100)
            button_left.bind("<Button-1>", partial(self.button_click, key="0"))

            button_right = tk.Button(root, text="Next")
            button_right.place(relx=0.85, rely=0.4, height=48, width=100)
            button_right.bind("<Button-1>", partial(self.button_click, key="1"))

    def run(self):
        self.labelling_start()
            
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

    def remove_buttons(self, option='First Menu'):

        if option == 'Test Neural Network':
            self.btn_selet_image.destroy()
            self.btn_segmentation.destroy()
            self.btn_mask_true.destroy()
            self.btn_diff_imgs.destroy()

            self.painel_up_left.destroy()
            self.painel_up_right.destroy()
            self.painel_down_left.destroy()
            self.painel_down_right.destroy()

        elif option == 'Compare Results':
            self.TSeparator1.destroy()
            self.btn_load_mosaico.destroy()
            self.btn_shape_reference.destroy()
            self.btn_shape_neural.destroy()
            self.btn_start.destroy()
            self.Spinbox1.destroy()
            self.Spinbox2.destroy()                    
            self.Spinbox3.destroy()                    
            self.Label1.destroy()                      
            self.Radiobutton1.destroy()   
            self.Label2.destroy()
            self.Label3.destroy()                      
            self.Label4.destroy()
            self.Label5.destroy()
            self.Label6.destroy()
            self.Label7.destroy()

        elif option == 'Draw Menu':
            self.Label2.destroy()
            self.btn_shape_reference.destroy()
            self.Label3.destroy()
            self.Spinbox_backg.destroy()
            self.btn_load_mosaico.destroy()

        else:
            
            self.Canvas1.destroy()
            self.Label1.destroy()

            self.opt.destroy()
            self.labelTest.destroy()

    def select_image(self):

        self.path_rgb = filedialog.askopenfilename()
        if self.path_rgb:
        
            self.img_rgb = cv2.imread(self.path_rgb, -1)
            self.img_rgb = cv2.resize(self.img_rgb, (256,256))

            image = Image.fromarray(self.img_rgb)
            image_tk = ImageTk.PhotoImage(image)

            self.painel_up_left.configure(image=image_tk)
            self.painel_up_left.image = image_tk

    def callback_opt(self, *args):

        if (self.old_choose == 'Test Neural Network' and self.variable.get() != 'Test Neural Network'):
            self.remove_buttons()

        if(self.variable.get() == 'Test Neural Network'):
            self.labelTest.destroy()
            self.opt.destroy()
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

            self.reference_binary = gdal.Open(self.shp_to_bin(self.name_reference_binary, self.name_tif[1]))
            self.reference_neural = gdal.Open(self.shp_to_bin(self.name_reference_neural, self.name_tif[1]))

            if self.name_tif[0]:

                button_left = tk.Button(root, text="Back")
                button_left.place(x=self.width_size*0.04, y=self.hight_size*0.5)
                button_left.bind("<Button-1>", partial(self.button_click, key="0"))

                button_right = tk.Button(root, text="Next")
                button_right.place(x=self.width_size*0.70, y=self.hight_size*0.5)
                button_right.bind("<Button-1>", partial(self.button_click, key="1"))
                
            else:
                mbox.showerror('Error', 'Nenhum Ortomosaico.tif foi Selecionado :')
                tif_loaded = False
        
        elif(self.variable.get() == 'Draw Weeds'):

            if not os.path.isdir(self.path_save_img_rgb):
                os.makedirs(self.path_save_img_rgb, exist_ok=True)
         
            if not os.path.isdir(self.path_save_img_bin):
                os.makedirs(self.path_save_img_bin, exist_ok=True)

            self.labelling_menu()

        self.old_choose = self.variable.get()
             
    def button_click(self, event=None, key=None):

        self.draw_img = PIL.Image.new("RGB",(self.iterator_x, self.iterator_y),(0,0,0))
        self.draw_line = PIL.ImageDraw.Draw(self.draw_img)
        self.cnt_validator = []
        
        if (key == "1"):
            if (self.x_crop + self.iterator_x < self.mosaico.RasterXSize and self.x_crop + self.iterator_x > 0):
                self.x_crop += self.iterator_x * self.iterator_recoil
                print('key 1 - if 0')

                if self.x_crop + self.iterator_x > self.mosaico.RasterXSize:
                    self.x_max = self.x_crop - self.iterator_x * self.iterator_recoil
                    print('entrou')

            if (self.x_crop + self.iterator_x > self.mosaico.RasterXSize):
                self.x_crop = 0
                self.y_crop += self.iterator_y * self.iterator_recoil
                print('key 1 - if 1')


            if (self.y_crop + self.iterator_y > self.mosaico.RasterYSize):
                self.x_crop = self.x_crop
                self.y_crop = self.y_crop
                print('key 1 - if 2')
                mbox.showinfo(title='Todo o Mosaico foi Percorrido!')

            self.daninha_parcela = self.daninha_band_1.ReadAsArray(self.x_crop, self.y_crop, self.iterator_x, self.iterator_y)
            while cv2.countNonZero(self.daninha_parcela) <= self.iterator_x*self.iterator_y*0.05:
                if (self.x_crop + self.iterator_x < self.mosaico.RasterXSize and self.x_crop + self.iterator_x > 0):
                    self.x_crop += self.iterator_x * self.iterator_recoil
                    print('key 1 - if 0')

                    if self.x_crop + self.iterator_x > self.mosaico.RasterXSize:
                        self.x_max = self.x_crop - self.iterator_x * self.iterator_recoil
                        print('entrou')

                if (self.x_crop + self.iterator_x > self.mosaico.RasterXSize):
                    self.x_crop = 0
                    self.y_crop += self.iterator_y * self.iterator_recoil
                    print('key 1 - if 1')

                if (self.y_crop + self.iterator_y > self.mosaico.RasterYSize):
                    self.x_crop = self.x_crop
                    self.y_crop = self.y_crop
                    print('key 1 - if 2')
                    mbox.showinfo(title='Todo o Mosaico foi Percorrido!')
                    break
                self.daninha_parcela = self.daninha_band_1.ReadAsArray(self.x_crop, self.y_crop, self.iterator_x, self.iterator_y)

        elif (key == "0"):
            if (self.x_crop - self.iterator_x < self.mosaico.RasterXSize):
                self.x_crop -= self.iterator_x * self.iterator_recoil
                print('key 0 - if 1')

                if self.x_crop <= 0:    
                    self.x_crop = self.x_max
                    self.y_crop -= self.iterator_y * self.iterator_recoil
                    print('key 0 - if 2')

            if (self.y_crop - self.iterator_y > self.mosaico.RasterYSize):
                self.x_crop =0
                self.y_crop -= self.iterator_y * self.iterator_recoil
                print('aqui2')
            
            self.daninha_parcela = self.daninha_band_1.ReadAsArray(self.x_crop, self.y_crop, self.iterator_x, self.iterator_y)
            while cv2.countNonZero(self.daninha_parcela) <= self.iterator_x*self.iterator_y*0.05:
                if (self.x_crop - self.iterator_x < self.mosaico.RasterXSize):
                    self.x_crop -= self.iterator_x * self.iterator_recoil
                    print('key 0 - if 1')

                    if self.x_crop <= 0:    
                        self.x_crop = self.x_max
                        self.y_crop -= self.iterator_y * self.iterator_recoil
                        print('key 0 - if 2')

                if (self.y_crop - self.iterator_y > self.mosaico.RasterYSize):
                    self.x_crop =0
                    self.y_crop -= self.iterator_y * self.iterator_recoil
                    print('aqui2')
                
                self.daninha_parcela = self.daninha_band_1.ReadAsArray(self.x_crop, self.y_crop, self.iterator_x, self.iterator_y)
        print('x :', self.x_crop,', y :', self.y_crop)
        self.daninha_parcela = self.daninha_band_1.ReadAsArray(self.x_crop, self.y_crop, self.iterator_x, self.iterator_y)
        #while (cv2.countNonZero(self.daninha_parcela) == 0):  
                        
        blueparcela = self.blue.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        greenparcela = self.green.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        redparcela = self.red.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        self.imgparcela = cv2.merge((blueparcela, greenparcela, redparcela))

        '''
        img_neural = self.reference_neural.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        self.img_binary   = self.reference_binary.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        union, self.dif = nf.diff_contourns(self, img_neural, self.img_binary)
    
        self.contours = nf.find_contourns(self, self.dif)
        self.draw = cv2.drawContours(self.imgparcela, self.contours, -1, (255, 0, 0), 3)
        
        img = PIL.Image.fromarray(self.draw)'''

        img = PIL.Image.fromarray(self.imgparcela)
        self.image_tk = ImageTk.PhotoImage(img)
        self.array_clicks.clear()
        self.first_click = True
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.canvas.create_image(self.iterator_x // 2, self.iterator_y // 2, image=self.image_tk, anchor=tk.CENTER)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        self.canvas.bind("<Button-1>",  self.get_x_and_y)
        self.canvas.bind("<B1-Motion>", self.draw_smth)
        #cv2.countNonZero(self.save_draw_array)
       
        
    def get_x_and_y(self, event):
        global lasx, lasy
        lasx, lasy = event.x, event.y
        self.array_clicks.append(lasx)
        self.array_clicks.append(lasy)

    def draw_smth(self, event):
        global lasx, lasy
        self.canvas.create_line((lasx, lasy, event.x, event.y), 
                                fill='red', capstyle=tk.ROUND, 
                                joinstyle=tk.ROUND, width=int(self.slider_pencil),
                                smooth=True, splinesteps=12,
                                dash=(3,5))
        '''                                             
            self.canvas_draw = self.canvas.create_polygon((self.array, event.x, event.y), 
                                                        fill='red', capstyle=tk.ROUND, 
                                                        joinstyle=tk.ROUND, width=10,
                                                        smooth=True, splinesteps=12,
                                                        dash=(3,5))
        '''
    
        self.draw_line.line((lasx, lasy, event.x, event.y), (255,255,255), width=int(self.slider_pencil), joint='curve')
        #self.draw_line.line((lasx, lasy, event.x, event.y), (255,255,255), width=int(self.slider_pencil), joint='curve')

        Offset = (int(self.slider_pencil)-1)/2
        self.draw_line.ellipse ((lasx-Offset,lasy-Offset,lasx+Offset,lasy+Offset), (255,255,255))
        lasx, lasy = event.x, event.y
        self.save_draw_array = np.asarray(self.draw_img)
        self.save_draw_array = nf.prepare_array(self, self.save_draw_array)
        #self.save_draw_array = nf.prepare_array(self, self.save_draw_array)
        #ia.imshow(self.save_draw_array[0][0]) 
        cv2.imwrite(self.path_save_img_rgb + '/daninha_{x}_{y}.png'.format(x=int(self.x_crop),y=int(self.y_crop)), self.imgparcela)
        cv2.imwrite(self.path_save_img_bin + '/daninha_{x}_{y}.png'.format(x=int(self.x_crop),y=int(self.y_crop)), self.save_draw_array)
        self.dst_img.GetRasterBand(1).WriteArray(self.save_draw_array, xoff=self.x_crop, yoff=self.y_crop)
        self.dst_img.FlushCache()
        
    def printcoords(self, event):

        cx, cy = self.event2canvas(event, self.canvas)
        self.ctn = []
        if self.first_click == True: 

            for i in range(0, len(self.contours)):
                self.cnt_validator.append(False)
                self.img_fit = cv2.fillPoly(self.dif, pts=self.contours, color=(0,0,0))

            self.first_click = False

        for i in range(0, len(self.cnt_validator)):   
            r = cv2.pointPolygonTest(self.contours[i], (cx, cy), False)
            if r > 0:
                self.cnt_validator[i] = (not self.cnt_validator[i])    
                self.ctn = self.contours[i]

                if self.cnt_validator[i] == True:
                    self.draw = cv2.drawContours(self.imgparcela, self.ctn, -1, (0, 255, 0), 3)
                    self.img_fit = cv2.fillPoly(self.dif, pts=[self.ctn], color=(255,255,255))

                    union_ref_checker = self.diff_contourns(self.img_binary, self.img_fit)[0]
                    self.dst_img.GetRasterBand(1).WriteArray(union_ref_checker, xoff=self.x_crop, yoff=self.y_crop)

                else:
                    self.draw = cv2.drawContours(self.imgparcela, self.ctn, -1, (255, 0, 0), 3)
                    self.img_fit = cv2.fillPoly(self.dif, pts=[self.ctn], color=(0,0,0))

        img = PIL.Image.fromarray(self.draw)
        image_tk = ImageTk.PhotoImage(img)
        #self.canvas.destroy()
        #self.canvas.pack(fill=tk.BOTH,expand=0)
        #self.painel_center.image=image_tk

    def save_in_reference_tif(self):

        if self.reference_neural.endswith('tif'):
            self.red = self.mosaico.GetRasterBand(1)

            self.dst_img = gdal.GetDriverByName('GTiff').Create(str(self.reference_neural), self.nx, self.ny, 1, gdal.GDT_Byte)
            self.dst_img.SetGeoTransform(self.mosaico.GetGeoTransform())
            self.srs = osr.SpatialReference()
            self.srs.ImportFromWkt(self.mosaico.GetProjection())
            self.dst_img.SetProjection(self.srs.ExportToWkt())

    def load_rgb_tif(self):

        path_rgb_shp = filedialog.askopenfilename(title='Selecione O Mosaico')
        if path_rgb_shp.endswith('tif'):

            self.mosaico = gdal.Open(path_rgb_shp)
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
        type : 0 - Carrega o Mosaico 
               1 - Representa o shape de referencia
               2 - Representa o shape da rede neural
        option : Refere-se ao tipo de operacao a ser executada
        """
        if option == 'Compare Results':
            if type_shape == 0:

                path_reference_tif = self.load_rgb_tif()[1]
                path_reference_shp = None
                path_neural_shp    = None

            elif type_shape == 1:
                path_reference_tif = None
                path_reference_shp = filedialog.askopenfilename(title='Selecione o Shape de Referência :')
                path_neural_shp    = None

            elif type_shape == 2:
                path_reference_tif = None
                path_reference_shp = None
                path_neural_shp    = filedialog.askopenfilename(title='Selecione o Shape da Rede Neural :')

            elif type_shape == 3:
                path_reference_shp = filedialog.askopenfilename(title='Selecione o Shape de Referência :')
                path_reference_tif = None
                path_neural_shp    = filedialog.askopenfilename(title='Selecione o Shape da Rede Neural :')

            return path_reference_tif, path_reference_shp, path_neural_shp

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

                pr = nf.predict_image(self, img)
                
                if (self.imgparcela.max()>0) and (self.imgparcela.min()<255):
                    write_image = pr

                else:
                    pr[pr>=255]= 0
                    write_image = pr
                    
                self.dst_img.GetRasterBand(1).WriteArray(write_image, xoff=x, yoff=y)
                self.dst_img.self.dst_img.FlushCache()

    def generate_shape(self):
        src_band = self.dst_img.GetRasterBand(1)
        dst_layername = 'daninhas'
        drv = ogr.GetDriverByName("ESRI Shapefile")
        dst_ds = drv.CreateDataSource(str(self.out_file))
        dst_layer = dst_ds.CreateLayer(dst_layername, srs = self.srs)

        gdal.Polygonize(src_band, src_band, dst_layer, -1, [], callback=None )
        dst_ds.Destroy()
        mbox.showinfo("Information", "Shape Gerado com Sucesso!: ")

    def destroy_aplication(self):
        string_text = 'x_' + str(self.x_crop) + '_y_' + str(self.y_crop) + '_name_' + str(self.name_tif)
        with open("log_progress.txt", "ab") as f:
            f.write(string_text.encode('utf-8', 'ignore'))
        root.destroy()

if __name__ == "__main__":

    root = tk.Tk()
    obj = Interface(root)
    root.title('Semantic Segmetation Tools')
    root.resizable(False,False)
    obj.first_menu(root)
    #Interface(root).run()
    root.geometry("800x800+400+400")
    Exit1 = tk.Button(root, text="Sair", command=obj.destroy_aplication)
    Exit1.place(relx=0.019, rely=0.871, height=48, width=100)
    root.mainloop()