import os
import cv2
import random
from sklearn.utils import shuffle
import numpy as np
import pickle
import pandas as pd
import argparse
from PIL import Image

from datetime import datetime


def load_pickle(func, format, fold):
    file_path = f'../data/5Folds/{func}_{format}{fold}.pickle'
    pickle_in = open(file_path,"rb")
    data = pickle.load(pickle_in)
    return data

def shuffle_data(data1, data2):
    data1, data2 = shuffle(data1, data2)
    return data1, data2

def get_XandY(data, IMG_SIZE = 224):
    X = []
    y = []

    for features,label in data:
            X.append(features)
            y.append(label)
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,3)
    return X, y

def normalize_image(image):
    # convert image to float type
    image = image.astype(np.float32)
    # rescale pixel values to range 0-1
    image -= image.min()
    image /= image.max() - image.min()
    return image

def create_training_data(CATEGORIES=["train"], DATADIR="",IMG_SIZE = 224):
    
    training_data = []
    for category in CATEGORIES:  # do dogs and cats
        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category) #+1 if rsua  # get the classification  (0 or a 1). 0=dog 1=cat
        
        for img in os.listdir(path):  # iterate over each image per dogs and cats
            if ".jpg" in img or ".png" in img or "PNG" in img or "jpeg" in img:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)  # convert to array
                # img_array = crop_image(img_array,0)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE),interpolation=cv2.INTER_AREA)
                new_array = new_array/255
                new_array = normalize_image(new_array)
            else:
                gif = cv2.VideoCapture(os.path.join(path,img))
                ret,frame = gif.read() # ret=True if it finds a frame else False.
                # print(path+'/'+img)
                img_array = Image.fromarray(frame)
                img_array = np.array(img_array.convert('RGB')) #harus dikasi np.array karena tipe data float32
                # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                # img_array = crop_image(img_array,0)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE),interpolation=cv2.INTER_AREA)
                new_array = new_array/255

            training_data.append([new_array, class_num])  # add this to our training_data
            
    return training_data