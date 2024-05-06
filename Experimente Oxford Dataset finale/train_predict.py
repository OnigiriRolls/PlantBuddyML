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


saved_model = tf.keras.models.load_model('gcs/plant_buddy/models/models/model11_retrain', compile=False)

base_learning_rate = 0.001
saved_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), metrics=['accuracy']) 

# model.build(input_shape=(None, 224, 224, 3))

loss, accuracy = saved_model.evaluate(test_set)
print('Test accuracy :', accuracy)
print('Test loss :', loss)
