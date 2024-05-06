#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pandas as pd


from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, Reshape
from tensorflow.keras import layers, Sequential

# In[2]:


dataset, dataset_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
dataset_info

test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']


# In[4]:


num_training_examples = 0
num_validation_examples = 0
num_test_examples = 0

for example in training_set:
  num_training_examples += 1

for example in validation_set:
  num_validation_examples += 1

for example in test_set:
  num_test_examples += 1

print('Total Number of Training Images: {}'.format(num_training_examples))
print('Total Number of Validation Images: {}'.format(num_validation_examples))
print('Total Number of Test Images: {} \n'.format(num_test_examples))

num_classes = dataset_info.features['label'].num_classes
print('Total Number of Classes: {}'.format(num_classes))


# In[4]:


desired_training_size = 6552
desired_validation_size = 819
desired_test_size = 818

test_set = test_set.shuffle(buffer_size=6149)
remaining_test_instances = test_set.skip(5532).take(617)
instances_for_training = test_set.take(5532)

training_set = training_set.concatenate(instances_for_training)
training_set = training_set.shuffle(buffer_size=desired_training_size)

validation_set = validation_set.shuffle(buffer_size=1020)
instances_for_test = validation_set.skip(819).take(201)

validation_set = validation_set.take(819)

test_set = remaining_test_instances.concatenate(instances_for_test)
test_set = test_set.shuffle(818)


IMAGE_RES = 224
BATCH_SIZE = 32
EPOCHS = 30

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

training_set = training_set.map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_set = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_set = test_set.map(format_image).batch(BATCH_SIZE).prefetch(1)

num_training_batches = tf.data.experimental.cardinality(training_set).numpy()
num_validation_batches = tf.data.experimental.cardinality(validation_set).numpy()
num_test_batches = tf.data.experimental.cardinality(test_set).numpy()

print("Number of batches in the training set:", num_training_batches)
print("Number of batches in the validation set:", num_validation_batches)
print("Number of batches in the test set:", num_test_batches)


input_shape = (224, 224, 3)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.4),
])

from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, Reshape
from tensorflow.keras import layers, Sequential

def tensorflow_based_model():
    model = Sequential() 
    
    model.add(data_augmentation) 
    
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(224,224,3), padding='same'))  
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=2))  

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))  
    model.add(MaxPooling2D(pool_size=2))  
    
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))  
    model.add(MaxPooling2D(pool_size=2)) 
    
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same')) 
    model.add(MaxPooling2D(pool_size=2)) 
    
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))  
    model.add(MaxPooling2D(pool_size=2))  
    
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))  
    model.add(MaxPooling2D(pool_size=2)) 
    
    model.add(Conv2D(filters=256, kernel_size=2, activation='relu', padding='same'))  
    model.add(MaxPooling2D(pool_size=2))  

    model.add(Conv2D(filters=256, kernel_size=2, activation='relu', padding='same'))  
    model.add(MaxPooling2D(pool_size=2)) 
    
    model.add(Conv2D(filters=512, kernel_size=2, activation='relu', padding='same'))  
    model.add(MaxPooling2D(pool_size=2))  
    
    model.add(Conv2D(filters=512, kernel_size=2, activation='relu', padding='same')) 
    model.add(MaxPooling2D(pool_size=2)) 
    
    model.add(Dropout(0.2))  
    model.add(Flatten())  
    model.add(Dense(150))  
    model.add(Activation('relu'))  
    model.add(Dropout(0.2)) 
    model.add(Dense(num_classes, activation='softmax')) 

    return model

model = tensorflow_based_model()
base_learning_rate = 0.001
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), metrics=['accuracy']) 


# In[ ]:

#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)

history = model.fit(training_set,
        batch_size = 32,
        epochs=50,
        validation_data=validation_set)


# In[150]:


model.save('/gcs/plant-buddy-bucket/model_oxford/model12_10conv17')
pd.DataFrame.from_dict(history.history).to_csv('/gcs/plant-buddy-bucket/model_oxford/history12_10conv17.csv',index=False)