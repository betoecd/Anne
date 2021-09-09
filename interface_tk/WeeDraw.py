import numpy as np
import os
import pathlib
import PIL
import cv2
import sys

try:

    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True
    import shutil

from typing import Text
from functools import partial
from tkinter import PhotoImage, messagebox as mbox
from PIL import Image, ImageDraw, ImageTk
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

        self.screen_width = 512
        self.screen_height = 512

        self.count_img = 0
        self.count_feature = 0
        self.iterator_recoil = 1.0
        self.cnt_validator = []
        self.background_percent = 0.8
        self.array_clicks = []
        self.current_points = []
        self.current_points_bkp = []
        self.draw_lines_array = [[]]
        self.features_polygons = [[]]
        self.polygons_ids_array = []
        self.vertices_ids_array = []

        self.save_draw_array = None
        self.name_tif = ''
        self.name_reference_binary = ''
        self.name_reference_neural = ''
        self.slider_pencil = 10
        self.slider_opacity = 5
        self.pencil_draw = True
        self.polygon_draw = False
        self.opacity = False
        self.bool_draw = False
        self.path_save_img_rgb = 'dataset/rgb'
        self.path_save_img_bin = 'dataset/binario'
        self.directory_saved = ''
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
        self.current_value_draw = tk.DoubleVar()
        self.current_value_opacity = tk.DoubleVar()
        self.btn_int = tk.IntVar()

        self.var = tk.IntVar()
        self.old_choose = '' 
        self.OptionList = ["Efetuar Marcacoes"] 

        img = ImageTk.PhotoImage(file='icons/icone_sensix.png')
        root.call('wm', 'iconphoto', root._w, img)

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
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        
        root.configure(highlightcolor="black")

        self.TSeparator1 = ttk.Separator(root)
        self.TSeparator1.place(relx=0.021, rely=0.535,  relwidth=0.962)

        self.menubar = tk.Menu(root,font="TkMenuFont",bg=_bgcolor,fg=_fgcolor)
        root.configure(menu = self.menubar)

        self.btn_load_mosaico = tk.Button(root)
        self.btn_load_mosaico.place(relx=0.74, rely=0.572, height=28, width=123)
        self.btn_load_mosaico.configure(takefocus="", text='Mosaico')
        self.btn_load_mosaico.bind('<Button-1>', partial(self.get_btn, key="0"))

        self.btn_shape_reference = tk.Button(root)
        self.btn_shape_reference.place(relx=0.74, rely=0.659, height=28, width=123)
        self.btn_shape_reference.configure(takefocus="", text='Shape de Refer')
        self.btn_shape_reference.bind('<Button-1>', partial(self.get_btn, key="1"))

        self.btn_shape_neural = tk.Button(root)
        self.btn_shape_neural.place(relx=0.74, rely=0.747, height=28, width=123)
        self.btn_shape_neural.configure(takefocus="", text='Shape da Rede')
        self.btn_shape_neural.bind('<Button-1>', partial(self.get_btn, key="2"))

        self.btn_start = tk.Button(root)
        self.btn_start.place(relx=0.742, rely=0.871, height=48, width=123)
        self.btn_start.configure(takefocus="", text='Iniciar')
        self.btn_start.bind('<Button-1>', partial(self.get_btn, key="5"))

        self.spinbox1 = tk.Spinbox(root, from_=10.0, to=800.0, increment=10, textvariable=self.spn_box_1)
        self.spinbox1.place(relx=0.74, rely=0.178, relheight=0.046, relwidth=0.243)
        self.spinbox1.configure(activebackground="#f9f9f9", background="white", font="TkDefaultFont", highlightbackground="black", selectbackground="blue", selectforeground="white", command=self.get_values_spinbox)

        self.spinbox2 = tk.Spinbox(root, from_=100.0, to=800.0, increment=100, textvariable=self.spn_box_2)
        self.spinbox2.place(relx=0.74, rely=0.271, relheight=0.046, relwidth=0.243)
        self.spinbox2.configure(activebackground="#f9f9f9", background="white", font="TkDefaultFont", highlightbackground="black", selectbackground="blue", selectforeground="white", command=self.get_values_spinbox)

        self.spinbox3 = tk.Spinbox(root, from_=100.0, to=800.0, increment=100, textvariable=self.spn_box_3)
        self.spinbox3.place(relx=0.74, rely=0.364, relheight=0.046, relwidth=0.243)
        self.spinbox3.configure(activebackground="#f9f9f9", background="white", font="TkDefaultFont", highlightbackground="black", selectbackground="blue", selectforeground="white", command=self.get_values_spinbox)
        
        self.label1 = tk.Label(root)
        self.label1.place(relx=0.015, rely=0.178, height=21, width=245)
        self.label1.configure(activebackground="#f9f9f9", text='Escolha a porcentagem de iteração :')

        self.Radiobutton1 = tk.Radiobutton(root)
        self.Radiobutton1.place(relx=0.721, rely=0.455, relheight=0.046, relwidth=0.132)
        self.Radiobutton1.configure(justify='left', text='Padrão', value=True, variable=self.bool_value, command=self.get_values_radio,)

        self.label2 = tk.Label(root)
        self.label2.place(relx=0.021, rely=0.269, height=21, width=235)
        self.label2.configure(text='Valor do comprimento da imagem :')

        self.label3 = tk.Label(root)
        self.label3.place(relx=0.021, rely=0.36, height=21, width=186)
        self.label3.configure(text='Valor de altura da imagem :')

        self.label4 = tk.Label(root)
        self.label4.place(relx=0.014, rely=0.446, height=21, width=186)
        self.label4.configure(text='Usar configuração padrão :')

        self.label5 = tk.Label(root)
        self.label5.place(relx=0.023, rely=0.58, height=21, width=138)
        self.label5.configure(text='Selecionar Mosaico :')

        self.label6 = tk.Label(root)
        self.label6.place(relx=0.019, rely=0.66, height=21, width=213)
        self.label6.configure(text='Selecionar shape de referencia :')

        self.label7 = tk.Label(root)
        self.label7.place(relx=0.019, rely=0.74, height=38, width=221)
        self.label7.configure(text='Selecionar shape da rede neural :')

    def labelling_start(self):
        root.minsize(1100, 700)
        root.resizable(1,  1)
        root.title("WeeDraw")

        self.value_label = tk.Label(root, text=self.get_current_value_draw())
        self.value_label_opacity = tk.Label(root, text=self.get_current_value_opacity())

        self.Scale1 = tk.Scale(root, tickinterval=1, from_=1.0, to=5.0, command=self.slider_changed_opacity,   variable=self.current_value_opacity)
        self.Scale1.place(relx=0.60, rely=0.88, relheight=0.093, relwidth=0.271)
        self.Scale1.configure(length="251", orient="horizontal", troughcolor="#d9d9d9")

        
        self.Scale2 = tk.Scale(root, tickinterval=5, from_=10.0, to=50.0, command=self.slider_changed_draw,   variable=self.current_value_draw)
        self.Scale2.place(relx=0.132, rely=0.88, relheight=0.093, relwidth=0.271)
        self.Scale2.configure(length="249", orient="horizontal", troughcolor="#d9d9d9")
        
        self.next_icon = PhotoImage(file = r"icons/next.png")
        self.button_right = tk.Button(root, image = self.next_icon)
        self.button_right.place(relx=0.926, rely=0.363, height=83, width=43)
        self.button_right.configure(borderwidth="2")

        self.back_icon = PhotoImage(file = r"icons/back.png")
        self.button_left = tk.Button(root, image = self.back_icon)
        self.button_left.place(relx=0.031, rely=0.363, height=83, width=43)
        self.button_left.configure(borderwidth="2")

        frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        xscroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        yscroll = tk.Scrollbar(frame)
        self.canvas = tk.Canvas(frame, bd=0, width=self.screen_width, height=self.screen_height, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
            
        #self.canvas = tk.Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        #xscroll.config(command=self.canvas.xview)
        #yscroll.config(command=self.canvas.yview)
        frame.pack(expand=1)

        self.pencil_icon = PhotoImage(file = r"icons/pencil_2.png")
        self.pencil_icon =  self.pencil_icon.subsample(1, 1)
        self.pencil_btn = tk.Button(root, image = self.pencil_icon)
        self.pencil_btn.place(relx=0.132, rely=0.82, height=40, width=40)
        self.pencil_btn.bind("<Button-1>", partial(self.get_btn, key='6'))

        self.polygon_icon = PhotoImage(file = r"icons/polygon.png")
        self.polygon_icon =  self.polygon_icon.subsample(1, 1)
        self.polygon_btn = tk.Button(root, image = self.polygon_icon)
        self.polygon_btn.place(relx=0.176, rely=0.82, height=40, width=40)
        self.polygon_btn.bind("<Button-1>", partial(self.get_btn, key='9'))

        self.transp_icon = PhotoImage(file = r"icons/transparency.png")
        self.transp_icon =  self.transp_icon.subsample(2, 2)
        self.transparency_btn = tk.Button(root, image=self.transp_icon)
        self.transparency_btn.place(relx=0.831, rely=0.82, height=40, width=40)
        self.transparency_btn.bind("<Button-1>", partial(self.get_btn, key='8'))
        
        self.erase_icon = PhotoImage(file = r"icons/erase.png")
        self.erase_icon =  self.erase_icon.subsample(1, 1)
        self.erase_btn = tk.Button(root, image=self.erase_icon)
        self.erase_btn.place(relx=0.222, rely=0.82, height=40, width=40)
        self.erase_btn.bind("<Button-1>", partial(self.get_btn, key='7'))

        self.percent_txt = tk.Label(root, text = '',  font=('Helvetica', 18), fg='black')
        self.percent_txt.place(relx=0.46, rely=0.06, height=21, width=80)
        self.percent_txt.configure(activebackground="#f9f9f9")

        self.current_value_opacity.set(5.0)

    def labelling_menu(self):

        self.label1.destroy()
        self.opt.destroy()

        self.label1 = tk.Label(root)
        self.label1.place(relx=0.090, rely=0.52, height=21, width=200)
        self.label1.configure(activebackground="#f9f9f9")
        self.label1.configure(text='Selecione o Mosaico :')

        self.btn_load_mosaico = tk.Button(root)
        self.btn_load_mosaico.place(relx=0.72, rely=0.52, height=28, width=123)
        self.btn_load_mosaico.configure(takefocus="")
        self.btn_load_mosaico.configure(text='Mosaico')
        self.btn_load_mosaico.bind('<Button-1>', partial(self.get_btn, key="0"))

        self.label2 = tk.Label(root)
        self.label2.place(relx=0.11, rely=0.60, height=21, width=200)
        self.label2.configure(activebackground="#f9f9f9")
        self.label2.configure(text='Selecione o Shape de Base :')

        self.btn_shape_reference = tk.Button(root)
        self.btn_shape_reference.place(relx=0.72, rely=0.60, height=28, width=123)
        self.btn_shape_reference.configure(takefocus="")
        self.btn_shape_reference.configure(text='Shape de Base')
        self.btn_shape_reference.bind('<Button-1>', partial(self.get_btn, key="3"))

        self.label3 = tk.Label(root)
        self.label3.place(relx=0.117, rely=0.68, height=21, width=260)
        self.label3.configure(activebackground="#f9f9f9")
        self.label3.configure(text='Porcentagem de fundo preto permitida:')

        self.spinbox_backg = tk.Spinbox(root, from_=5.0, to=100.0, increment=5, textvariable=self.spn_box_1)
        self.spinbox_backg.place(relx=0.63, rely=0.69, relheight=0.046, relwidth=0.243)
        self.spinbox_backg.configure(activebackground="#f9f9f9")
        self.spinbox_backg.configure(background="white")
        self.spinbox_backg.configure(font="TkDefaultFont")
        self.spinbox_backg.configure(highlightbackground="black")
        self.spinbox_backg.configure(selectbackground="blue")
        self.spinbox_backg.configure(selectforeground="white")
        self.spinbox_backg.configure(command=partial(self.get_values_spinbox, type='Efetuar Marcacoes'))

        self.btn_start = tk.Button(root)
        self.btn_start.place(relx=0.742, rely=0.871, height=48, width=123)
        self.btn_start.configure(takefocus="")
        self.btn_start.configure(text='Iniciar')
        self.btn_start.bind('<Button-1>', partial(self.get_btn, key="5"))

        '''
        self.label4 = tk.Label(root)
        self.label4.place(relx=0.117, rely=0.69, height=21, width=235)
        self.label4.configure(text='Valor do comprimento da imagem :')

        self.spinbox2 = tk.Spinbox(root, from_=256.0, to=768.0, increment=256, textvariable=self.spn_box_2)
        self.spinbox2.place(relx=0.63, rely=0.69, relheight=0.046, relwidth=0.243)
        self.spinbox2.configure(activebackground="#f9f9f9")
        self.spinbox2.configure(background="white")
        self.spinbox2.configure(font="TkDefaultFont")
        self.spinbox2.configure(highlightbackground="black")
        self.spinbox2.configure(selectbackground="blue")
        self.spinbox2.configure(selectforeground="white")
        self.spinbox2.configure(command=partial(self.get_values_spinbox, type='Efetuar Marcacoes'))

        self.label5 = tk.Label(root)
        self.label5.place(relx=0.117, rely=0.76, height=21, width=186)
        self.label5.configure(text='Valor de altura da imagem :')

        self.spinbox3 = tk.Spinbox(root, from_=256.0, to=400.0, increment=256, textvariable=self.spn_box_3)
        self.spinbox3.place(relx=0.63, rely=0.76, relheight=0.046, relwidth=0.243)
        self.spinbox3.configure(activebackground="#f9f9f9")
        self.spinbox3.configure(background="white")
        self.spinbox3.configure(font="TkDefaultFont")
        self.spinbox3.configure(highlightbackground="black")
        self.spinbox3.configure(selectbackground="blue")
        self.spinbox3.configure(selectforeground="white")
        self.spinbox3.configure(command=partial(self.get_values_spinbox, type='Efetuar Marcacoes'))
        '''

    def first_menu(self, app):

        self.logo = PhotoImage(file = r"icons/Logo-Escuro.png")
        self.logo =  self.logo.subsample(5, 5)
        self.canvas1 = tk.Canvas(root)
        self.canvas1.place(relx=0.117, rely=0.111, relheight=0.291, relwidth=0.752)
        self.canvas1.configure(borderwidth="2")
        self.canvas1.configure(relief="ridge")
        self.canvas1.configure(selectbackground="blue")
        self.canvas1.configure(selectforeground="white")
        self.canvas1.create_image(300,90,image=self.logo, anchor='center')

        self.label1 = tk.Label(root)
        self.label1.place(relx=0.25, rely=0.578, height=21, width=200)
        self.label1.configure(text='Selecione uma opção : ')

        self.variable = tk.StringVar(app)
        self.variable.set('Ainda Nao Selecionado')

        self.opt = tk.OptionMenu(app, self.variable, *self.OptionList)
        #opt.config(width=90, font=('Helvetica', 12))
        self.opt.place(x=self.width_size*0.50, y=self.hight_size*0.57)

        self.label_opt = tk.Label(text="", font=('Helvetica', 12), fg='red')
        self.label_opt.pack(side="top")

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

        elif option == 'Comparar Resultados':
            self.TSeparator1.destroy()
            self.btn_load_mosaico.destroy()
            self.btn_shape_reference.destroy()
            self.btn_shape_neural.destroy()
            self.btn_start.destroy()
            self.spinbox1.destroy()
            self.spinbox2.destroy()                    
            self.spinbox3.destroy()                    
            self.label1.destroy()                      
            self.Radiobutton1.destroy()   
            self.label2.destroy()
            self.label3.destroy()                      
            self.label4.destroy()
            self.label5.destroy()
            self.label6.destroy()
            self.label7.destroy()

        elif option == 'Draw Menu':
            self.label2.destroy()
            self.btn_shape_reference.destroy()
            self.label3.destroy()
            self.spinbox_backg.destroy()
            self.btn_start.destroy()
            self.btn_load_mosaico.destroy()
            #self.spinbox3.destroy()
            #self.spinbox2.destroy()
            #self.label4.destroy()
            #self.label5.destroy()

        elif option ==  'Start':
            self.TSeparator1.destroy()
            self.btn_load_mosaico.destroy()
            self.btn_shape_reference.destroy()
            self.btn_shape_neural.destroy()
            self.btn_start.destroy()
            self.spinbox1.destroy()
            self.spinbox2.destroy()
            self.spinbox3.destroy()
            self.Radiobutton1.destroy()
            self.label1.destroy()
            self.label2.destroy()
            self.label3.destroy()
            self.label4.destroy()
            self.label5.destroy()
            self.label6.destroy()
            self.label7.destroy()

        else:
            
            self.canvas1.destroy()
            self.label1.destroy()

            self.opt.destroy()
            self.label_opt.destroy()

    def get_text(self):  
        text_val = self.entry_text.get()
        
        label_init = tk.Label(root, text=text_val)
        self.canvas_init.create_window(200, 230, window=label_init)

    def get_current_value_draw(self):
        self.slider_pencil = self.current_value_draw.get()
        if  self.slider_pencil < 10:
            self.slider_pencil = 10
        
        return '{: .2f}'.format(self.current_value_draw.get())

    def get_current_value_opacity(self):

        if  self.current_value_opacity.get() == 1:
            self.slider_opacity = 'gray12'

        elif self.current_value_opacity.get() == 2:
            self.slider_opacity = 'gray25'

        elif self.current_value_opacity.get() == 3:
            self.slider_opacity = 'gray50'

        elif self.current_value_opacity.get() == 4:
            self.slider_opacity = 'gray75'

        else:
            self.slider_opacity = None
        
        return '{: .2f}'.format(self.current_value_opacity.get())

    def slider_changed_opacity(self, event):
        self.value_label_opacity.configure(text=self.get_current_value_opacity())

    def slider_changed_draw(self, event):
        self.value_label.configure(text=self.get_current_value_draw())

    def change_dir(self, event=None, key=None):
        self.cnt_validator = []   

        if (key == "1"):
            self.count_img += 1
       
        else:
            self.count_img += -1

        #print(self.name_dir[self.count_img])
        
        self.img_rgb = cv2.imread(self.diretorio_rgb + self.name_dir[self.count_img], 1)
        self.img_neural = cv2.imread(self.diretorio_pred + self.name_dir[self.count_img], 1)
        self.img_neural = cv2.cvtColor(self.img_neural, cv2.COLOR_BGR2GRAY)
        self.img_neural = nf.adjust_pixels(self, self.img_neural, 50)
        #self.img_neural = nf.ellipse_overlap(self, self.img_neural, )
        
        self.img_binary = cv2.imread(self.diretorio_true + self.name_dir[self.count_img], 1)
        self.img_binary = cv2.cvtColor(self.img_binary, cv2.COLOR_BGR2GRAY) 

        self.union, self.dif = nf.diff_contourns(self, self.img_binary, self.img_neural,)
        self.contours = nf.find_contourns(self, self.dif)
        self.draw = cv2.drawContours(self.img_rgb, self.contours, -1, (255, 0, 0), 3)
        
        img = PIL.Image.fromarray(self.draw)
        img = cv2.resize(self.img_rgb, (self.screen_width, self.screen_height))
        img = PIL.Image.fromarray(img)
        self.image_tk = ImageTk.PhotoImage(img)

        self.img_canvas_id = self.canvas.create_image(self.screen_width// 2, self.screen_height // 2, image=self.image_tk, anchor=tk.CENTER)
        #print(self.img_canvas_id)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

        self.first_click = True      
        self.canvas.bind("<ButtonPress-1>",self.printcoords)


    def get_values_spinbox(self, type=''):

        if type == 'Comparar Resultados':

            if self.first_click_bool == False:
                self.iterator_recoil = float(int(self.spinbox1.get())/100)
                self.iterator_x = int(self.spinbox2.get())
                self.iterator_y = int(self.spinbox3.get())

        elif type == 'Efetuar Marcacoes':
            self.background_percent = float(1-int(self.spinbox_backg.get())/100)
            #self.iterator_x = int(self.spinbox2.get())
            #self.iterator_y = int(self.spinbox3.get())
            self.iterator_recoil = 1.0
            #print(self.background_percent)
                    
        else:
            values1 = self.iterator_recoil * 100
            values2 = self.iterator_x
            values3 = self.iterator_y

    def get_values_radio(self):
        self.first_click_bool = not (self.first_click_bool)
        
        if self.first_click_bool:
            bool_default = bool(self.bool_value.get())
            self.spn_box_1.set('80')
            self.spn_box_2.set('256')
            self.spn_box_3.set('256')
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

        elif key=='6':
            self.pencil_draw = True
            self.polygon_draw = False
            self.opacity = False

        elif key=='7':
            self.pencil_draw = False
            self.polygon_draw = False
            self.opacity = False
        
        elif key=='9':
            self.pencil_draw = False
            self.polygon_draw = True
            self.opacity = False

        elif key=='8':
            self.opacity = not self.opacity
            if self.opacity:
                option_img = 'normal'
                #print('No if')
                self.canvas.tag_lower(self.img_canvas_id)
                self.canvas.update()
            else:
                self.canvas.tag_raise(self.img_canvas_id)
                self.canvas.update()
            

        elif self.name_tif != '' and self.name_reference_binary != '' and self.name_reference_neural != '' and key=='5':
            root.geometry("800x600+50+10")
            self.ready_start = True

        elif self.name_tif != '' and self.name_reference_binary != '' and key=='5':
            #self.remove_buttons('Draw Menu')
            self.reference_binary = self.shp_to_bin(self.name_reference_binary, self.name_tif)
            self.remove_buttons('Fisrt Menu')
            self.labelling_start()
            self.remove_buttons('Draw Menu')
            self.name_tif = '"' + self.name_tif + '"'
            if(self.load_progress()):

                if(str(self.directory_saved) == str((self.name_tif))):
                    mbox.showinfo('Information','O Progresso Anterior foi Carregado!')
                    self.dst_img = gdal.Open('resutado_gerado.tif', gdal.GA_Update)
                    
                else:
                    self.x_crop = 0.0
                    self.y_crop = 0.0
                    self.dst_img = gdal.GetDriverByName('GTiff').Create('resutado_gerado.tif', self.mosaico.RasterXSize, self.mosaico.RasterYSize, 1, gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
                    self.dst_img.SetProjection(self.mosaico.GetProjectionRef())
                    self.dst_img.SetGeoTransform(self.mosaico.GetGeoTransform())
          
            else:
                self.x_crop = 0.0
                self.y_crop = 0.0
                self.dst_img = gdal.GetDriverByName('GTiff').Create('resutado_gerado.tif', self.mosaico.RasterXSize, self.mosaico.RasterYSize, 1, gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
                self.dst_img.SetProjection(self.mosaico.GetProjectionRef())
                self.dst_img.SetGeoTransform(self.mosaico.GetGeoTransform()) 

            self.daninha_1 = gdal.Open(self.reference_binary)
            self.daninha_band_1 =  self.daninha_1.GetRasterBand(1)

            self.draw_img = PIL.Image.new("RGBA",(self.screen_width, self.screen_height),(0,0,0,0))
            self.draw_line = ImageDraw.Draw(self.draw_img)
            self.cnt_validator = []   

            self.button_right.bind("<Button-1>", partial(self.button_click, key="1"))
            self.button_left.bind("<Button-1>", partial(self.button_click, key="0"))

        if self.ready_start:
            self.remove_buttons('Start')
            
            self.reference_binary = gdal.Open(self.shp_to_bin(self.name_reference_binary, self.name_tif), 1)
            self.reference_neural = gdal.Open(self.shp_to_bin(self.name_reference_neural, self.name_tif), 1)
            
            self.dst_img = gdal.GetDriverByName('GTiff').Create(self.name_reference_binary + '_out.tif', self.reference_binary.RasterXSize, self.reference_binary.RasterYSize, 1, gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
            self.dst_img.SetProjection(self.reference_binary.GetProjectionRef())
            self.dst_img.SetGeoTransform(self.reference_binary.GetGeoTransform()) 

            frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            xscroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
            yscroll = tk.Scrollbar(frame)
            self.canvas = tk.Canvas(frame, bd=0, width=self.screen_width, height=self.screen_height, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
                
            #self.canvas = tk.Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
            self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
            #xscroll.config(command=self.canvas.xview)
            #yscroll.config(command=self.canvas.yview)
            frame.pack(expand=1)

            self.next_icon = PhotoImage(file = r"icons/next.png")
            self.button_right = tk.Button(root, image = self.next_icon)
            self.button_right.place(relx=0.926, rely=0.363, height=83, width=43)
            self.button_right.configure(borderwidth="2")
            self.button_right.bind("<Button-1>", partial(self.change_dir, key="1"))

            self.back_icon = PhotoImage(file = r"icons/back.png")
            self.button_left = tk.Button(root, image = self.back_icon)
            self.button_left.place(relx=0.031, rely=0.363, height=83, width=43)
            self.button_left.configure(borderwidth="2")
            self.button_left.bind("<Button-1>", partial(self.change_dir, key="0"))


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
            self.label_opt.destroy()
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

        elif(self.variable.get() == 'Comparar Resultados'):
            self.remove_buttons()
            self.start()
            
        
        elif(self.variable.get() == 'Comparar por Diretorios'):
            self.remove_buttons()

            '''
            self.diretorio_rgb = filedialog.askdirectory(title='Selecione o diretorio de imagens RGB :') + '/'
            self.diretorio_true = filedialog.askdirectory(title='Selecione o diretorio de imagens de marcacoes manuais :') + '/'
            self.diretorio_pred = filedialog.askdirectory(title='Selecione o diretorio de imagens de predicao :') + '/'
            '''
            self.diretorio_rgb = "../../daninhas/ortomosaicos/pre-emergente/190837/rgb/"
            self.diretorio_true = "../../daninhas/ortomosaicos/pre-emergente/190837/marcacoes_manuais/"
            self.diretorio_pred = "../../daninhas/ortomosaicos/pre-emergente/190837/resultados_rede_manuais/"

            #(self.diretorio_rgb, self.diretorio_true, self.diretorio_pred)
            self.name_dir = os.listdir(self.diretorio_rgb)
            frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            xscroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
            yscroll = tk.Scrollbar(frame)
            self.canvas = tk.Canvas(frame, bd=0, width=self.screen_width, height=self.screen_height, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
                
            #self.canvas = tk.Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
            self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
            #xscroll.config(command=self.canvas.xview)
            #yscroll.config(command=self.canvas.yview)
            frame.pack(expand=1)
            self.next_icon = PhotoImage(file = r"icons/next.png")
            self.button_right = tk.Button(root, image = self.next_icon)
            self.button_right.place(relx=0.926, rely=0.363, height=83, width=43)
            self.button_right.configure(borderwidth="2")
            self.button_right.bind("<Button-1>", partial(self.change_dir, key="1"))

            self.back_icon = PhotoImage(file = r"icons/back.png")
            self.button_left = tk.Button(root, image = self.back_icon)
            self.button_left.place(relx=0.031, rely=0.363, height=83, width=43)
            self.button_left.configure(borderwidth="2")
            self.button_left.bind("<Button-1>", partial(self.change_dir, key="0"))

        elif(self.variable.get() == 'Efetuar Marcacoes'):

            if not os.path.isdir(self.path_save_img_rgb):
                os.makedirs(self.path_save_img_rgb, exist_ok=True)
         
            if not os.path.isdir(self.path_save_img_bin):
                os.makedirs(self.path_save_img_bin, exist_ok=True)

            self.labelling_menu()

        self.old_choose = self.variable.get()

    def percent_progress(self, x, y, total_x, total_y):
        current_position = y/total_y 
        current_position = round(current_position*100, 2)
        return current_position
             
    def button_click(self, event=None, key=None):
        if (self.bool_draw):
            self.count_feature = 0
            data_polygons = []

            for i in range(0, len(self.features_polygons)-1, 1):
                try:
                    if (self.features_polygons[i]   == []):
                        continue
                    data_polygons.append((self.features_polygons[i][2]))
                    #print(i, len(self.features_polygons))
                    #print('Features : ', self.features_polygons)
                    #print('Features Array : ', self.features_polygons[i][2])

                    if (self.features_polygons[i+1][0] != self.features_polygons[i][0]):
                        #print(data_polygons)
                        #print('-----------------------------------------------------')
                        self.draw_line.polygon((data_polygons), fill='white', outline='white')
                        data_polygons.clear()

                    if (i+3) == len(self.features_polygons):
                        #print('Ultimo :')
                        data_polygons.append(self.features_polygons[i+1][2])
                        data_polygons.append(self.features_polygons[i+2][2])
                        self.draw_line.polygon((data_polygons), fill='white', outline='white')
                        #print(data_polygons)
                        data_polygons.clear()
                except:
                    pass

            self.save_draw_array = np.asarray(self.draw_img)
            self.save_draw_array = nf.prepare_array(self, self.save_draw_array, self.iterator_x, self.iterator_y)

            self.canvas.delete(self.img_canvas_id)
            cv2.imwrite(self.path_save_img_rgb + '/daninha_{x}_{y}.png'.format(x=int(self.x_crop),y=int(self.y_crop)), self.imgparcela)
            cv2.imwrite(self.path_save_img_bin + '/daninha_{x}_{y}.png'.format(x=int(self.x_crop),y=int(self.y_crop)), self.save_draw_array)
            self.dst_img.GetRasterBand(1).WriteArray(self.save_draw_array, xoff=self.x_crop, yoff=self.y_crop)
            self.dst_img.FlushCache()
            self.bool_draw = False
        
            self.draw_img = PIL.Image.new("RGB",(self.screen_width, self.screen_height),(0,0,0))
            self.draw_line = ImageDraw.Draw(self.draw_img)
            self.current_points_bkp.clear()
            self.current_points.clear()
            self.canvas.delete('oval')
            self.canvas.delete('line')
            self.canvas.delete('simple_line')
            self.canvas.delete('poly')
            self.cnt_validator = []
            #self.canvas.delete(self.line_obj)

            self.draw_lines_array.clear()
            self.features_polygons.clear()
            
            del self.draw_lines_array
            del self.features_polygons
            self.draw_lines_array = []
            self.features_polygons = [[]]

        if (key == "1"):
            if (self.x_crop + self.iterator_x < self.mosaico.RasterXSize and self.x_crop + self.iterator_x > 0):
                self.x_crop += self.iterator_x * self.iterator_recoil
                #print('key 1 - if 0')

                if self.x_crop + self.iterator_x > self.mosaico.RasterXSize:
                    self.x_max = self.x_crop - self.iterator_x * self.iterator_recoil
                    #print('entrou')

            if (self.x_crop + self.iterator_x > self.mosaico.RasterXSize):
                self.x_crop = 0
                self.y_crop += self.iterator_y * self.iterator_recoil
                #print('key 1 - if 1')


            if (self.y_crop + self.iterator_y > self.mosaico.RasterYSize):
                self.x_crop = self.x_crop
                self.y_crop = self.y_crop
                #print('key 1 - if 2')
                mbox.showinfo("Information", "Todo o Mosaico foi Percorrido!")
                self.destroy_aplication()

            self.daninha_parcela = self.daninha_band_1.ReadAsArray(self.x_crop, self.y_crop, self.iterator_x, self.iterator_y)
            while cv2.countNonZero(self.daninha_parcela) <= self.iterator_x*self.iterator_y*self.background_percent:
                if (self.x_crop + self.iterator_x < self.mosaico.RasterXSize and self.x_crop + self.iterator_x > 0):
                    self.x_crop += self.iterator_x * self.iterator_recoil
                    #print('key 1 - if 0')

                    if self.x_crop + self.iterator_x > self.mosaico.RasterXSize:
                        self.x_max = self.x_crop - self.iterator_x * self.iterator_recoil
                        #print('entrou')

                if (self.x_crop + self.iterator_x > self.mosaico.RasterXSize):
                    self.x_crop = 0
                    self.y_crop += self.iterator_y * self.iterator_recoil
                    #print('key 1 - if 1')

                if (self.y_crop + self.iterator_y > self.mosaico.RasterYSize):
                    self.x_crop = self.x_crop
                    self.y_crop = self.y_crop
                    #print('key 1 - if 2')
                    mbox.showinfo("Information", "Todo o Mosaico foi Percorrido!")
                    self.destroy_aplication()
                    break

                self.daninha_parcela = self.daninha_band_1.ReadAsArray(self.x_crop, self.y_crop, self.iterator_x, self.iterator_y)

        elif (key == "0"):
            if (self.x_crop - self.iterator_x < self.mosaico.RasterXSize):
                self.x_crop -= self.iterator_x * self.iterator_recoil
                #print('key 0 - if 1')

                if self.x_crop <= 0:    
                    self.x_crop = self.x_max
                    self.y_crop -= self.iterator_y * self.iterator_recoil
                    #print('key 0 - if 2')

            if (self.y_crop - self.iterator_y > self.mosaico.RasterYSize):
                self.x_crop =0
                self.y_crop -= self.iterator_y * self.iterator_recoil
                #print('aqui2')
            
            self.daninha_parcela = self.daninha_band_1.ReadAsArray(self.x_crop, self.y_crop, self.iterator_x, self.iterator_y)
            while cv2.countNonZero(self.daninha_parcela) <= self.iterator_x*self.iterator_y*self.background_percent:
                if (self.x_crop - self.iterator_x < self.mosaico.RasterXSize):
                    self.x_crop -= self.iterator_x * self.iterator_recoil
                    #print('key 0 - if 1')

                    if self.x_crop <= 0:    
                        self.x_crop = self.x_max
                        self.y_crop -= self.iterator_y * self.iterator_recoil
                        #print('key 0 - if 2')

                if (self.y_crop - self.iterator_y > self.mosaico.RasterYSize):
                    self.x_crop =0
                    self.y_crop -= self.iterator_y * self.iterator_recoil
                    #print('aqui2')
                
                self.daninha_parcela = self.daninha_band_1.ReadAsArray(self.x_crop, self.y_crop, self.iterator_x, self.iterator_y)

        self.daninha_parcela = self.daninha_band_1.ReadAsArray(self.x_crop, self.y_crop, self.iterator_x, self.iterator_y)
        self.percent_txt['text'] = str(self.percent_progress(self.x_crop, self.y_crop, self.mosaico.RasterXSize, self.mosaico.RasterYSize)) + '%'
        blueparcela = self.blue.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        greenparcela = self.green.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        redparcela = self.red.ReadAsArray(self.x_crop, self.y_crop,self.iterator_x, self.iterator_y)
        self.imgparcela = cv2.merge((blueparcela, greenparcela, redparcela))
        self.imgparcela[self.daninha_parcela == 0] = 0

        self.img_array_tk = cv2.resize(self.imgparcela, (self.screen_width, self.screen_height))
        self.img_array_tk = PIL.Image.fromarray(self.img_array_tk)
        self.image_tk = ImageTk.PhotoImage(self.img_array_tk)
        self.first_click = True
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.img_canvas_id = self.canvas.create_image(self.screen_width// 2, self.screen_height // 2, image=self.image_tk, anchor=tk.CENTER)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        self.canvas.bind("<Button-1>",  self.get_x_and_y)
        self.canvas.bind("<Button 3>", self.right_click)
        self.canvas.bind("<B1-Motion>", self.draw_smth)
        #cv2.countNonZero(self.save_draw_array)
          
    def right_click(self, event):
        self.current_points.clear()
        self.count_feature += 1

    def get_x_and_y(self, event):
        self.lasx, self.lasy = event.x, event.y
        if(self.polygon_draw):            
            self.current_points.append((self.lasx, self.lasy))
            self.current_points_bkp.append((self.lasx, self.lasy))
            #print(self.current_points)
            for pt in self.current_points:
                x, y =  pt
                x1, y1 = (x - 1), (y - 1)
                x2, y2 = (x + 1), (y + 1)
                self.vertices_ids = self.canvas.create_oval(x1, y1, x2, y2, fill='blue', outline='blue', width=3, tags='oval')
                self.vertices_ids_array.append(self.vertices_ids)

            number_points=len(self.current_points)
            if number_points>2:
                self.polygons_ids=self.canvas.create_polygon(self.current_points, fill='red', outline='', width=2, stipple=self.slider_opacity, tags='poly')
                #self.draw_line.polygon((self.current_points), fill='white', outline='white')
                self.features_polygons.extend([[self.count_feature, self.polygons_ids, ((self.lasx, self.lasy))]])
                self.polygons_ids_array.append(self.polygons_ids)
                #print(self.features_polygons)

            elif number_points==2 :
                self.polygons_ids= self.canvas.create_line(self.current_points, fill='', tags='simple_line')
                self.features_polygons.extend([[self.count_feature, self.polygons_ids, ((self.lasx, self.lasy))]])  
                self.polygons_ids_array.append(self.polygons_ids)

            elif number_points<2:
                #self.polygons_ids= self.canvas.create_line(self.current_points)
                self.features_polygons.extend([[self.count_feature, 1, ((self.lasx, self.lasy))]])  
                self.polygons_ids_array.append(self.vertices_ids_array)

            self.bool_draw = True

    def draw_smth(self, event):
        if(self.pencil_draw):
            self.lasx, self.lasy = event.x, event.y
            self.line_obj = self.canvas.create_line((self.lasx, self.lasy, event.x, event.y), 
                                    fill='red', capstyle=tk.ROUND, 
                                    joinstyle=tk.ROUND, width=int(self.slider_pencil),
                                    smooth=True, splinesteps=12,
                                    dash=(3,5), stipple=self.slider_opacity, tags='line')
            
            self.draw_line.line((self.lasx, self.lasy, event.x, event.y), (255,255,255), width=int(self.slider_pencil), joint='curve')
            Offset = (int(self.slider_pencil))/2
            self.draw_line.ellipse ((self.lasx-Offset,self.lasy-Offset,self.lasx+Offset,self.lasy+Offset), (255,255,255))
            
            clicks_ids = [[self.line_obj, self.lasx, self.lasy]]
            self.draw_lines_array.extend(clicks_ids)
            #print(self.draw_lines_array)

        elif(not self.pencil_draw and not self.polygon_draw):
            self.lasx, self.lasy = event.x, event.y
            for i in range(1, len(self.draw_lines_array[:][:]), 1):
                if  self.lasx - self.slider_pencil <= self.draw_lines_array[:][i][1] and self.lasx + self.slider_pencil > self.draw_lines_array[:][i][1] and \
                    self.lasy - self.slider_pencil <= self.draw_lines_array[:][i][2] and self.lasy + self.slider_pencil > self.draw_lines_array[:][i][2]:
                    
                    try:
                        self.canvas.delete(self.draw_lines_array[:][i][0])
                        self.draw_line.line((self.lasx, self.lasy, event.x, event.y), (0,0,0), width=int(self.slider_pencil), joint='curve')
                        Offset = (int(self.slider_pencil))/2
                        self.draw_line.ellipse((self.lasx-Offset,self.lasy-Offset,self.lasx+Offset,self.lasy+Offset), (0,0,0))
                   
                    except:
                        pass

            for i in range(1, len(self.features_polygons), 1):
                #print('valor poly : ', self.features_polygons, self.features_polygons[i], end='\n')
                if  self.lasx - 10 <= self.features_polygons[i][2][0] and self.lasx + 10 > self.features_polygons[i][2][0] and \
                    self.lasy - 10 <= self.features_polygons[i][2][1] and self.lasy + 10 > self.features_polygons[i][2][1]:
                    self.canvas.delete(self.features_polygons[i][1])
                    self.canvas.delete('oval')
                    self.canvas.delete('simple_line')
                    self.canvas.delete(self.features_polygons[i][0])
                    self.features_polygons.pop(i) 
        #self.save_draw_array = np.asarray(self.draw_img)
        #self.save_draw_array = nf.prepare_array(self, self.save_draw_array, self.iterator_x, self.iterator_y)
        self.bool_draw = True


    def printcoords(self, event):

        cx, cy = self.event2canvas(event, self.canvas)
        self.ctn = []
        if self.first_click == True: 

            for i in range(0, len(self.contours)):
                self.cnt_validator.append(False)
                self.img_fit = cv2.fillPoly(self.dif, pts=self.contours, color=(0,0,0))
            
            #print("False")
            self.first_click = False

        for i in range(0, len(self.cnt_validator)):   
            r = cv2.pointPolygonTest(self.contours[i], (cx, cy), False)
            #print(r)
            if r > 0:
                self.cnt_validator[i] = (not self.cnt_validator[i])    
                #print("Selected contour ", i)   
                self.ctn = self.contours[i]

                if self.cnt_validator[i] == True:
                    self.draw = cv2.drawContours(self.img_rgb, self.ctn, -1, (0, 255, 0), 3)
                    self.img_fit = cv2.fillPoly(self.dif, pts=[self.ctn], color=(255,255,255))

                    union_ref_checker = nf.diff_contourns(self, self.img_binary, self.img_fit)[0]
                    #self.dst_img.GetRasterBand(1).WriteArray(union_ref_checker, xoff=self.x_crop, yoff=self.y_crop)
                    #self.dst_img.FlushCache()
                else:
                    self.draw = cv2.drawContours(self.img_rgb, self.ctn, -1, (255, 0, 0), 3)
                    self.img_fit = cv2.fillPoly(self.dif, pts=[self.ctn], color=(0,0,0))

        #print('validator :', self.cnt_validator)
        img = PIL.Image.fromarray(self.draw)
        self.image_tk = ImageTk.PhotoImage(img)

        self.img_canvas_id = self.canvas.create_image(self.screen_width// 2, self.screen_height // 2, image=self.image_tk, anchor=tk.CENTER)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        

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

    def load_shp(self, type_shape=0, option='Comparar Resultados'):
        """
        Carrega o shp que sera utilido nas comparacoes
        type : 0 - Carrega o Mosaico 
               1 - Representa o shape de referencia
               2 - Representa o shape da rede neural
               3 - Representa o shape de ambos
        option : Refere-se ao tipo de operacao a ser executada
        """
        if option == 'Comparar Resultados':
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

    def load_progress(self):
        
        try:   
            f = open('log_progress.txt', encoding="utf-8")
            for lines in f:
                pass
            values = lines.split(',')
            self.x_crop = float(values[0])
            self.y_crop = float(values[1])
            self.directory_saved = str(values[2])

            bool_check_dir = True
            f.close()

        except:
            f = open('log_progress.txt','x', encoding="utf-8")
            bool_check_dir = False
            f.close()
        return(bool_check_dir)

    def destroy_aplication(self):

        if (self.x_crop >= 0 and self.y_crop >= 0):
            string_text = str(self.x_crop) + ',' + str(self.y_crop) + ',' + str(self.name_tif) + ', \n'
            with open("log_progress.txt", "ab") as f:
                f.write(string_text.encode('utf-8', 'ignore'))

        root.destroy()

if __name__ == "__main__":

    root = tk.Tk()
    obj = Interface(root)
    root.title('WeeDraw')
    root.resizable(False,False)
    obj.first_menu(root)
    #Interface(root).run()
    root.geometry("800x800+50+10")
    Exit1 = tk.Button(root, text="Sair", command=obj.destroy_aplication)
    Exit1.place(relx=0.019, rely=0.871, height=48, width=100)
    root.mainloop()