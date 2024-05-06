#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


# In[5]:


dataset, dataset_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
dataset_info

test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']

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


# In[7]:



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


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

# %%
from tensorflow.keras import layers

# URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
# feature_extractor = hub.KerasLayer(URL,
#                                    input_shape=(IMAGE_RES, IMAGE_RES, 3))

base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_RES, IMAGE_RES, 3),
                                               include_top=False,
                                               weights='imagenet')


for image_batch, label_batch in training_set.take(1):
    pass
image_batch.shape

feature_batch = base_model(image_batch)
print(feature_batch.shape)


base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(102, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

dropout_rate = 0.2

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = layers.Dropout(dropout_rate)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# %%


import pandas as pd

model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.summary()

import pandas as pd
base_learning_rate=0.001

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

EPOCHS = 35

history = model.fit(training_set,
                    epochs=EPOCHS,
                    validation_data=validation_set)

model.save('/gcs/plant-buddy-bucket/model_oxford/model9_3001_finetuned_before')
pd.DataFrame.from_dict(history.history).to_csv('/gcs/plant-buddy-bucket/model_oxford/history9_3001_finetuned_before.csv',index=False)


base_model.trainable = True

print("Number of layers in the base model: ", len(base_model.layers))

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

history_fine = model.fit(training_set,
                    epochs=EPOCHS,
                    validation_data=validation_set)

model.save('/gcs/plant-buddy-bucket/model_oxford/model9_3001_finetuned_after')
pd.DataFrame.from_dict(history_fine.history).to_csv('/gcs/plant-buddy-bucket/model_oxford/history9_3001_finetuned_after.csv',index=False)