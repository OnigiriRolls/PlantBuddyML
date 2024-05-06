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
from codecarbon import OfflineEmissionsTracker

# In[2]:

training_set=tf.data.Dataset.load('/gcs/plant_buddy_eu/oxford_dataset/training_set')
validation_set=tf.data.Dataset.load('/gcs/plant_buddy_eu/oxford_dataset/validation_set')
test_set=tf.data.Dataset.load('/gcs/plant_buddy_eu/oxford_dataset/test_set')

training_class_counts = {}
validation_class_counts = {}
test_class_counts = {}

for images, labels in training_set:
    for label in labels.numpy():
        label = label.item()  
        if label in training_class_counts:
            training_class_counts[label] += 1
        else:
            training_class_counts[label] = 1

for images, labels in validation_set:
    for label in labels.numpy():
        label = label.item() 
        if label in validation_class_counts:
            validation_class_counts[label] += 1
        else:
            validation_class_counts[label] = 1


for images, labels in test_set:
    for label in labels.numpy():
        label = label.item() 
        if label in test_class_counts:
            test_class_counts[label] += 1
        else:
            test_class_counts[label] = 1

print("Training set labels:")
for label, count in training_class_counts.items():
    print("{}".format(label))

print("Training set:")
for label, count in training_class_counts.items():
    print("{}".format(count))

print("\nValidation set labels:")
for label, count in validation_class_counts.items():
    print("{}".format(label))

print("\nValidation set:")
for label, count in validation_class_counts.items():
    print("{}".format(count))

print("\nTest set labels:")
for label, count in test_class_counts.items():
    print("{}".format(label))

print("\nTest set:")
for label, count in test_class_counts.items():
    print("{}".format(count))
            
            
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

num_training_batches = tf.data.experimental.cardinality(training_set).numpy()
num_validation_batches = tf.data.experimental.cardinality(validation_set).numpy()
num_test_batches = tf.data.experimental.cardinality(test_set).numpy()

print("Number of batches in the training set:", num_training_batches)
print("Number of batches in the validation set:", num_validation_batches)
print("Number of batches in the test set:", num_test_batches)


# In[5]:


input_shape = (224, 224, 3)

# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip('horizontal'),
#     tf.keras.layers.RandomRotation(0.2),
# ])

# from tensorflow.keras.layers import Conv2D,MaxPooling2D
# from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, Reshape
# from tensorflow.keras import layers, Sequential

# def tensorflow_based_model():
#     model = Sequential()  # Step 1
    
#     model.add(data_augmentation) 
    
#     model.add(Conv2D(filters=16, kernel_size=2, input_shape=(224,224,3), padding='same'))  # Step 2
#     model.add(Activation('relu'))  # Step 3
#     model.add(MaxPooling2D(pool_size=2))  # Step 4

#     model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))  # Repeating Step 2 and Step 3
#     model.add(MaxPooling2D(pool_size=2))  # Repeating Step 4
    
#     model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))  # Repeating Step 2 and Step 3
#     model.add(MaxPooling2D(pool_size=2))  # Repeating Step 4
    
#     model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))  # Repeating Step 2 and Step 3
#     model.add(MaxPooling2D(pool_size=2))  # Repeating Step 4

#     model.add(Conv2D(filters=256, kernel_size=2, activation='relu', padding='same'))  # Repeating Step 2 and Step 3
#     model.add(MaxPooling2D(pool_size=2))  # Repeating Step 4
    
#     model.add(Conv2D(filters=256, kernel_size=2, activation='relu', padding='same'))  # Repeating Step 2 and Step 3
#     model.add(MaxPooling2D(pool_size=2))  # Repeating Step 4
    
#     model.add(Conv2D(filters=256, kernel_size=2, activation='relu', padding='same'))  # Repeating Step 2 and Step 3
#     model.add(MaxPooling2D(pool_size=2))  # Repeating Step 4

#     model.add(Flatten())  # Step 6
#     model.add(Dense(150))  # Step 7
#     model.add(Activation('relu'))  # Step 3
#     model.add(Dropout(0.2))  # Step 5
#     model.add(Dense(102, activation='softmax'))  # Step 3 and Step 7 with softmax activation

#     return model

# model = tensorflow_based_model()
model = tf.keras.models.load_model('gcs/plant_buddy_eu/models/model12_8conv_17', compile=False)

# model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy']) 
base_learning_rate = 0.001
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), metrics=['accuracy']) 


# In[ ]:

tracker = OfflineEmissionsTracker(country_iso_code="BEL", measure_power_secs=120, output_dir="/gcs/plant_buddy_eu/energy", output_file="emissions12_8conv_17_re.csv")
tracker.start()

history = model.fit(training_set,
        batch_size = 32,
        epochs=70,
        validation_data=validation_set)

emissions: float = tracker.stop()
print(emissions)

model.save('/gcs/plant_buddy_eu/models/model12_8conv_17_re')
pd.DataFrame.from_dict(history.history).to_csv('/gcs/plant_buddy_eu/histories/history12_8conv_17_re.csv',index=False)

tracker = OfflineEmissionsTracker(country_iso_code="BEL", measure_power_secs=30, output_dir="/gcs/plant_buddy_eu/energy", output_file="emissions12_8conv_17_re_eval.csv")
tracker.start()

loss, accuracy = model.evaluate(test_set)
print('Test accuracy :', accuracy)
print('Test loss :', loss)

emissions: float = tracker.stop()
print(emissions)