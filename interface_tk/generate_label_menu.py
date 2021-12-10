#! /usr/bin/env python
#  -*- coding: utf-8 -*-

import sys
import generate_label_menu_support
from PIL import Image, ImageDraw, ImageTk
from tkinter import PhotoImage
from tkSliderWidget.tkSliderWidget import Slider 

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

def vp_start_gui():
    global val, w, root
    root = tk.Tk()
    top = MenuGenerateLabel (root)
    generate_label_menu_support.init(root, top)
    root.mainloop()

w = None

def create_menu_interface(rt, *args, **kwargs):
    global w, w_win, root
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    top = MenuGenerateLabel (w)
    generate_label_menu_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_label_menu():
    global w
    w.destroy()
    w = None

class MenuGenerateLabel:
    def __init__(self, top=None):

        top.geometry('1290x700+60+20')
        top.maxsize(1350, 740)
        top.resizable(0, 0)
        top.title('New Toplevel')
        self.menu_screen(top)


    def menu_screen(self, top):
        
        self.color_frame_options = '#414851'
        self.color_background = '#262930'
        self.color_frame_over_center = 'black' 
        self.color_buttons_center = 'white'
        self.background_slider = '#414851'
        self.intern_slider = '#5a636f'

        self.frame_ground = tk.Frame(top)
        self.frame_ground.place(relx=0.0, rely=0.0, relheight=1350, relwidth=740)
        self.frame_ground.configure(relief='groove', borderwidth='0', background=self.color_background)

        #Tela inferior com botoes
        self.frame_below_center = tk.Frame(top)
        self.frame_below_center.place(relx=0.178, rely=0.013, relheight=0.966, relwidth=0.817)
        self.frame_below_center.configure(relief='groove', borderwidth='0', background=self.color_frame_over_center)

        #Tela superior, onde a imagem principal é mostrada
        self.frame_over_center = tk.Frame(self.frame_below_center)
        self.frame_over_center.place(relx=0.051, rely=0.0, relheight=0.999, relwidth=0.898)
        self.frame_over_center.configure(relief='groove', borderwidth='0', background=self.color_frame_over_center)


        #self.canvas_main = tk.Canvas(self.frame_below_center)
        #self.logo = PhotoImage(file = r"icons/Logo-Escuro.png")
        #self.canvas_main.place(relx=0.05, rely=0.0, relheight=0.999, relwidth=0.898)
        #self.canvas_main.create_image(512, 512, image=self.logo, anchor='center')



        self.logo = PhotoImage(file = r"icons/Logo-Escuro.png")
        self.frame_of_options = tk.Frame(top)
        self.frame_of_options.place(relx=0.008, rely=0.014, relheight=0.964, relwidth=0.159)
        self.frame_of_options.configure(borderwidth='0', background=self.color_frame_options)

        self.pencil_icon = PhotoImage(file = r"icons/pencil_black_white.png")
        self.pencil_icon = self.pencil_icon.subsample(2, 2)
        self.pencil_btn = tk.Button(self.frame_below_center, image=self.pencil_icon)
        self.pencil_btn.place(relx=0.006, rely=0.004, height=43, width=43)
        self.pencil_btn.configure(borderwidth='2', text='Button', background=self.color_buttons_center)

        self.erase_icon = ImageTk.PhotoImage(file = r"icons/eraser_black.png")
        self.erase_btn = tk.Button(self.frame_below_center, image=self.erase_icon)
        self.erase_btn.place(relx=0.006, rely=0.135, height=43, width=43)
        self.erase_btn.configure(borderwidth='2', text='Button', background=self.color_buttons_center)

        self.polygon_icon = ImageTk.PhotoImage(file = r"icons/polygon.png")
        self.polygon_btn = tk.Button(self.frame_below_center, image=self.polygon_icon)
        self.polygon_btn.place(relx=0.006, rely=0.07, height=43, width=43)
        self.polygon_btn.configure(borderwidth='2', text='Button', background=self.color_buttons_center)

        self.hide_layer_icon = ImageTk.PhotoImage(file = r"icons/hide_layer.png")
        self.hide_layer_btn = tk.Button(self.frame_below_center, image=self.hide_layer_icon)
        self.hide_layer_btn.place(relx=0.006, rely=0.2, height=43, width=43)
        self.hide_layer_btn.configure(activebackground='#f9f9f9', borderwidth='2', text='Button', background=self.color_buttons_center)

        self.next_icon = PhotoImage(file = r"icons/next.png")
        self.next_btn = tk.Button(root, image = self.next_icon)
        self.next_btn.place(relx=0.957, rely=0.43, height=70, width=43)
        self.next_btn.configure(borderwidth="2", background='white')

        self.back_icon = PhotoImage(file = r"icons/back.png")
        self.back_btn = tk.Button(root, image = self.back_icon)
        self.back_btn.place(relx=0.183, rely=0.43, height=70, width=43)
        self.back_btn.configure(borderwidth="2", background='white')

        self.slider_thickness = tk.Scale(self.frame_of_options, from_=0.0, to=100.0)
        self.slider_thickness.place(relx=0.098, rely=0.0705, relheight=0.062, relwidth=0.8)
        self.slider_thickness.configure(length='164', orient='horizontal', borderwidth='0', troughcolor=self.intern_slider, fg='white',
                                        bg=self.background_slider, highlightbackground=self.background_slider)
       
        self.slider_opacity = tk.Scale(self.frame_of_options, from_=0.0, to=100.0)
        self.slider_opacity.place(relx=0.098, rely=0.178, relheight=0.062, relwidth=0.8)
        self.slider_opacity.configure(length='164', orient='horizontal', borderwidth='0', troughcolor=self.intern_slider, fg='white',
                                      bg=self.background_slider, highlightbackground=self.background_slider)

        self.slider_contourn = tk.Scale(self.frame_of_options, from_=0.0, to=100.0)
        self.slider_contourn.place(relx=0.098, rely=0.279, relheight=0.062, relwidth=0.8)
        self.slider_contourn.configure(length='164', orient='horizontal', borderwidth='0', troughcolor=self.intern_slider, fg='white',
                                       bg=self.background_slider, highlightbackground=self.background_slider)
                              
        self.slider_tolerancy = tk.Scale(self.frame_of_options, from_=0.0, to=100.0)
        self.slider_tolerancy.place(relx=0.098, rely=0.49, relheight=0.062, relwidth=0.8)
        self.slider_tolerancy.configure(length='164', orient='horizontal', borderwidth='0', troughcolor=self.intern_slider, fg='white',
                                        bg=self.background_slider, highlightbackground=self.background_slider)

        self.slider_saturation = tk.Scale(self.frame_of_options, from_=0.0, to=100.0)
        self.slider_saturation.place(relx=0.098, rely=0.385, relheight=0.062, relwidth=0.8)
        self.slider_saturation.configure(length='164', orient='horizontal', borderwidth='0', troughcolor=self.intern_slider, fg='white',
                                        bg=self.background_slider, highlightbackground=self.background_slider)

        self.label_thickness = tk.Label(self.frame_of_options)
        self.label_thickness.place(relx=0.059, rely=0.03, height=21, width=79)
        self.label_thickness.configure(text='Espessura:', background=self.color_frame_options, fg=self.color_buttons_center)

        self.label_opacity = tk.Label(self.frame_of_options)
        self.label_opacity.place(relx=0.059, rely=0.138, height=31, width=79)
        self.label_opacity.configure(text='Opacidade:', fg=self.color_buttons_center, bg=self.color_frame_options)

        self.label_contourn = tk.Label(self.frame_of_options)
        self.label_contourn.place(relx=0.059, rely=0.247, height=21, width=79)
        self.label_contourn.configure(text='Contorno:', fg=self.color_buttons_center, bg=self.color_frame_options)

        self.label_saturation = tk.Label(self.frame_of_options)
        self.label_saturation.place(relx=0.059, rely=0.356, height=21, width=79)
        self.label_saturation.configure(text='Saturação:', fg=self.color_buttons_center, bg=self.color_frame_options)

        self.label_saturation = tk.Label(self.frame_of_options)
        self.label_saturation.place(relx=1.093, rely=0.684, height=21, width=79)
        self.label_saturation.configure(activebackground='#f9f9f9', text='Espessura:', background=self.color_buttons_center)

        self.label_contourn_tolerancy = tk.Label(self.frame_of_options)
        self.label_contourn_tolerancy.place(relx=0.059, rely=0.462, height=21, width=79)
        self.label_contourn_tolerancy.configure(text='Tolerância:', fg=self.color_buttons_center, bg=self.color_frame_options)

        self.logo =  self.logo.subsample(15, 15)
        self.canvas_logo = tk.Canvas(self.frame_of_options)
        self.canvas_logo.place(relx=0.04, rely=0.910, relheight=0.08, relwidth=0.94)
        self.canvas_logo.create_image(92, 30,image=self.logo, anchor='center')
        
        #self.menubar = tk.Menu(top,font='TkMenuFont',bg='black',fg='black')
        #top.configure(menu = self.menubar)
    
        
if __name__ == '__main__':
    vp_start_gui()





