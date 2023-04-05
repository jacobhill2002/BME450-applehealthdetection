# BME450-lungcancerdetection #
Lung Cancer CT Detection Imaging and Analysis # Jacob Hill, Jett Stad, and John Morris
# We plan to use a dataset of medical CT images that show different human lung diagrams. We will use a neural network in order to train the computer on how to detect if the patient has a tumor from their picture and determine what type of stage of cancer progression it is in. We will do this by training the network to visualize different colors (black and white) and sizes in order to make a prognosis similar to that of a specialized medical worker. By doing this we hope to be able to aid and improve the accuracy of future prognosis of CT scans for this type of disease. #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
 
from sklearn.model_selection import train_test_split
from sklearn import metrics
 
import cv2
import gc
import os
 
import tensorflow as tf
from tensorflow import keras
from keras import layers
 
import warnings
warnings.filterwarnings('ignore')

#Import dataset 
from zipfile import ZipFile
 
data_path = 'lung-and-colon-cancer-histopathological-images.zip'
 
with ZipFile(data_path,'r') as zip:
  zip.extractall()
  print('The data set has been extracted.')
  
#Data Visualization
path = 'lung_colon_image_set/lung_image_sets'
classes = os.listdir(path)
classes

path = '/lung_colon_image_set/lung_image_sets'
 
#Data Preparation For Training
for cat in classes:
    image_dir = f'{path}/{cat}'
    images = os.listdir(image_dir)
 
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Images for {cat} category . . . .', fontsize=20)
 
    for i in range(3):
        k = np.random.randint(0, len(images))
        img = np.array(Image.open(f'{path}/{cat}/{images[k]}'))
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.show()
    
#Hyperparameters
IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 64

#Training
X = []
Y = []
 
for i, cat in enumerate(classes):
  images = glob(f'{path}/{cat}/*.jpeg')
 
  for image in images:
    img = cv2.imread(image)
     
    X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    Y.append(i)
 
X = np.asarray(X)

X_train, X_val, Y_train, Y_val = train_test_split(X, one_hot_encoded_Y, test_size = SPLIT, random_state = 2022)
print(X_train.shape, X_val.shape)
one_hot_encoded_Y = pd.get_dummies(Y).values

#Model Development
model = keras.models.Sequential([
    layers.Conv2D(filters=32,
                  kernel_size=(5, 5),
                  activation='relu',
                  input_shape=(IMG_SIZE,
                               IMG_SIZE,
                               3),
                  padding='same'),
    layers.MaxPooling2D(2, 2),
 
    layers.Conv2D(filters=64,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same'),
    layers.MaxPooling2D(2, 2),
 
    layers.Conv2D(filters=128,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same'),
    layers.MaxPooling2D(2, 2),
 
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax')])
    
    model.summary()
    
    keras.utils.plot_model(
    model,
    show_shapes = True,
    show_dtype = True,
    show_layer_activations = True)
   
#Optimization
    model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])
    
#Callback
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
 
 
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.90:
            print('\n Validation accuracy has reached upto \
                      90% so, stopping further training.')
            self.model.stop_training = True
 
 
es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True) 
lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)

history = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1, callbacks = [es, lr, myCallback()])

#Plotting
history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['accuracy','val_accuracy']].plot() 
plt.show()

#Model Evaluation
Y_pred = model.predict(X_val)
Y_val = np.argmax(Y_val, axis=1)
Y_pred = np.argmax(Y_pred, axis=1)

#Confusion Metrics 
metrics.confusion_matrix(Y_val, Y_pred)

#Print Metrics
print(metrics.classification_report(Y_val, Y_pred, target_names=classes))
   
