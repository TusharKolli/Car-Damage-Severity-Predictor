import numpy as np
import keras
import keras.utils as image
import matplotlib.pyplot as plt
from keras.layers import Dense,Flatten,Input,Dropout
from keras.models import Model
from keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import load_model

model = load_model("C:\MObile Net V2.model\MObile Net V2.model\MobileNet_Model_Final.keras")

import pickle 
img=image.load_img(r"C:\Users\Tushar\Downloads\minor-car-accidents.jpg",target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
output=np.argmax(model.predict(img_data),axis=1)
index=['1-Minor', '2-Moderate', '3-Severe', '4-Good']
result=str(index[output[0]])
print(result)