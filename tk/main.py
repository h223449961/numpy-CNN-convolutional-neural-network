# -*- coding: utf-8 -*-
from ImageLoader import Img_load
from canny import Canny_edge_detector
from utilities import image_utility
import cv2
import numpy as np
from nnet import net
import random
import urllib.request
from tkinter import filedialog
import tkinter as tk
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from itertools import zip_longest
class recognition:
    def __init__(self):
        self.img_load=Img_load()
        self.util=image_utility()
        self.neural_net=None
    def load_and_preprocess(self, path):
        img_path=path
        images=self.img_load.load_img(img_path)
        y_images=self.util.convert_rgb_gray(images)
        gray_images=self.util.convert_rgb_gray(images)
        return gray_images
    def get_hog_features(self, images):
        hog_feats=[]
        eps=0
        for image in images:
            hog_feats.append(self.util.generate_hog(image))
            eps+=1
        return hog_feats
    def prep_data_to_train(self, pos, neg, pos_img_feats, neg_img_feats):
        x=[]
        y=[]
        imgs=[]
        for i in range(len(pos_img_feats)):
            x.append(pos_img_feats[i].flatten())
            y.append(1)
            imgs.append(pos[i])
        for j in range(len(neg_img_feats)):
            x.append(neg_img_feats[i].flatten())
            y.append(0)
            imgs.append(neg[i])
        temp=list(zip(x, y, imgs))
        np.random.shuffle(temp)
        x, y, imgs=zip(*temp)
        return x, y, imgs
    def real_time_exe(self, url):
        '''
        headers = {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE"}
        req = urllib.request.Request(url=url,headers=headers)
        file = urllib.request.urlopen(req)
        imagefromweb = np.array(bytearray(file.read()), dtype=np.int8)
        image = cv2.imdecode(imagefromweb, -1)
        '''
        url = cv2.imdecode(np.fromfile(url,dtype=np.uint8),-1)
        imageresiz = cv2.resize(url, (500, 250))
        imagetoneuralnetwork = cv2.resize(imageresiz, (64,64), interpolation=cv2.INTER_AREA)
        gray = 0.114 * imagetoneuralnetwork[:,:,0] + 0.587 * imagetoneuralnetwork[:,:,1] + 0.299 * imagetoneuralnetwork[:,:,2]
        grays = []
        grays.append(gray)
        hogs = self.get_hog_features(grays)
        #print(hogs)
        volume = hogs[0].flatten()
        imageproce = []
        imageproce.append(volume)
        imageneuron = np.array(imageproce, np.float32)
        y_pred=self.neural_net.test(imageneuron)
        if(y_pred[0][1]<0.53):
            acpre_label.configure(text ='並不是積水容器',font=(None,24,'bold'),fg = "#107c10")
        else:
            acpre_label.configure(text ='潛在的積水容器',font=(None,24,'bold'),fg = "#C13E43")
        cvphoto = Image.fromarray(imageresiz)
        imgtk = ImageTk.PhotoImage(image=cvphoto)
        media.imgtk = imgtk
        media.configure(image = imgtk)
    def run_model(self):
        xtrain_neg_path = 'C:/Users/E418/Desktop/train_negative'
        xtrain_pos_path = 'C:/Users/E418/Desktop/train_positive'
        xtest_neg_path = 'C:/Users/E418/Desktop/test_negative'
        xtest_pos_path = 'C:/Users/E418/Desktop/test_positive'
        xtrain_pos=self.load_and_preprocess(xtrain_pos_path)
        xtrain_neg=self.load_and_preprocess(xtrain_neg_path)
        xtest_pos=self.load_and_preprocess(xtest_pos_path)
        xtest_neg=self.load_and_preprocess(xtest_neg_path)
        train_pos_hogs=self.get_hog_features(xtrain_pos)
        train_neg_hogs=self.get_hog_features(xtrain_neg)
        test_pos_hogs=self.get_hog_features(xtest_pos)
        test_neg_hogs=self.get_hog_features(xtest_neg)
        a, b, c=train_pos_hogs[0].shape
        input_dim=a*b*c
        x_train, y_train, train_imgs=self.prep_data_to_train(xtrain_pos, xtrain_neg, train_pos_hogs, train_neg_hogs)
        x_test, y_test, test_imgs=self.prep_data_to_train(xtest_pos, xtest_neg, test_pos_hogs, test_neg_hogs)
        self.neural_net=net(1, [300, 450, 600], input_dim)
        x_train=np.array(x_train, np.float32)
        y_train=np.array(y_train, np.int32)
        x_test=np.array(x_test, np.float32)
        y_test=np.array(y_test, np.int32)
        self.neural_net.train(x_train, y_train, 999, 3, 0.05)
        y_pred=self.neural_net.test(x_test)	
        return y_test, y_pred, train_imgs, test_imgs
def opfile():
    sfname = filedialog.askopenfilename(title='選擇',filetypes=[('All Files','*'),("jpeg files","*.jpg"),("png files","*.png"),("gif files","*.gif")])
    return sfname
def oand():
    filename = opfile()
    global a
    a = recognition()
    actuals, predictions, train_imgs, test_imgs = a.run_model()
    a.real_time_exe(filename)
def main():
    global root
    root = tk.Tk()
    mediaFrame = tk.Frame(root).pack()
    global media
    media = tk.Label(mediaFrame)
    media.pack()
    b1 = tk.Button(root, text="打開",command = oand).pack()
    global acpre_frame
    acpre_frame = tk.Frame(root)
    acpre_frame.pack(side=tk.TOP)
    global acpre_label
    acpre_label = tk.Label(acpre_frame, text='辨識結果',font=(None,24,'bold'),fg = "#00adee")
    acpre_label.pack(side=tk.RIGHT)
    #global a
    #a = recognition()
    #actuals, predictions, train_imgs, test_imgs = a.run_model()
    #a.real_time_exe('2.jpg')
    root.mainloop()
if __name__=='__main__':
    main()
