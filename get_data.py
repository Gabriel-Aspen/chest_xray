import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = 'data/train'
CATEGORIES = ['NORMAL', 'PNEUMONIA']

def get_data(datadir, categories):
  DATADIR = datadir
  CATEGORIES = categories
  IMG_SIZE = 224

  training_data = []
  for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path)[:]:
      try:
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #grayscale
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])
      except Exception as e:
        pass
  random.shuffle(training_data)

  X = []
  y = []
  for features, label in training_data:
    X.append(features)
    y.append(label)
  X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #the last one is because its in grayscale
  with open('pickle_files/X_train.pickle, 'wb') as pickle_out: #name the X!
    pickle.dump(X, pickle_out)
    pickle_out.close()
  with open('pickle_files/y_train.pickle', 'wb') as pickle_out: #name the y!
      pickle.dump(y, pickle_out)
      pickle_out.close()
get_data(DATADIR, CATEGORIES)
