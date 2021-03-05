#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

# Confirm that we're using Python 3
assert sys.version_info.major is 3, 'Oops, not running Python 3. Use Runtime > Change runtime type'


# In[2]:


# TensorFlow and tf.keras
print("Installing dependencies for Colab environment")
get_ipython().system('pip install -Uq grpcio==1.32.0')

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

print('TensorFlow version: {}'.format(tf.__version__))


# In[3]:


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))


# In[4]:


model = keras.Sequential([
  keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
                      strides=2, activation='relu', name='Conv1'),
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
])
model.summary()

testing = False
epochs = 15

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=epochs)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))


# In[5]:


# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors,
# and stored with the default serving key
import tempfile

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model(
    model,
    "API/Fashion/1",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

print('\nSaved model:')
get_ipython().system('ls -l {export_path}')


# In[6]:


tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True, show_dtype=True,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)


# In[7]:


get_ipython().system('saved_model_cli show --dir {export_path} --all')


# In[8]:


def show(idx, title):
  plt.figure()
  plt.imshow(test_images[idx].reshape(28,28))
  plt.axis('off')
  plt.title('\n\n{}'.format(title), fontdict={'size': 16})

import random
rando = random.randint(0,len(test_images)-1)
show(rando, 'An Example Image: {}'.format(class_names[test_labels[rando]]))


# In[9]:


import json

with open('request.json', 'w') as f:
    json.dump({"signature_name": "serving_default", "instances": test_images[0:3].tolist()}, f)


# In[10]:


get_ipython().system('pip install sklearn')
from skimage import transform, io
from sklearn.preprocessing import MinMaxScaler

img_array = io.imread("https://shop.tate.org.uk/dw/image/v2/BBPB_PRD/on/demandware.static/-/Sites-TateMasterShop/default/dwaa107262/tate-logo-black--tshirt-back-g1086.jpg", as_gray=True)
small_grey = transform.resize(
  img_array, (28, 28), mode='symmetric', preserve_range=True)

small_grey = (small_grey * -1)
small_grey = small_grey / 255.0
plt.imshow(small_grey)

scaler = MinMaxScaler()
attempt2 = small_grey
scaler.fit_transform(attempt2)
attempt2.reshape(28,28,1)

small_grey.reshape(28,28,1)

data_array = np.ndarray((1,28,28,1), dtype=float)

np.append(data_array,attempt2)


# In[11]:


model.predict(data_array)


# In[12]:


prediction_array = model.predict(data_array)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
result = {
    "prediction": class_names[np.argmax(prediction_array)],
    "confidence": '{:2.0f}%'.format(100*np.max(prediction_array))
}

print(result)


# In[ ]:




