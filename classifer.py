import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

x = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]
# print(pd.Series(y).value_counts())
classes = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K','L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0,train_size=3500,test_size = 500)
xtrainscaled = x_train/255.0
xtestscaled = x_test/255.0

lr = LogisticRegression(solver = "saga",multi_class = "multinomial")
lr.fit(xtrainscaled,y_train)

y_pred = lr.predict(xtestscaled)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

def getPrediction(image):
        im_pil = Image.open(image)    
        # converting the image to pil format
        # L it means - each pixel is represented as 0 to 255
        # antialias - it is use to minimize the distortion of the image
        img_bw = im_pil.convert("L")
        img_bw_resized = img_bw.resize((28,28),Image.ANTIALIAS)
        
        pixel_filter = 20
        min_pixel = np.percentile(img_bw_resized,pixel_filter)
        img_bw_resized_inverted_scaled = np.clip(img_bw_resized-min_pixel,0,255)
        max_pixel= np.max(img_bw_resized)
        # converting the data into an array and passing it to the model to predict
        img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = lr.predict(test_sample)
        return test_pred[0]