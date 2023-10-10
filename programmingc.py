#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 19:04:25 2022

@author: arpanrajpurohit
"""

import numpy as np
import pandas
import matplotlib.pyplot as plot
import tensorflow as tf
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape

dataset = tf.keras.datasets.fashion_mnist
(train_input, train_output), (test_input, test_output) = dataset.load_data()
del train_output
del test_output

EPOCHS = 100
BATCH_SIZE = 100
NOICE_SHAPE=100
train_input=train_input.reshape(len(train_input),28,28,1)
train_input =  train_input.astype('float32')
train_input = train_input/255
train_input = train_input*2 - 1.

generator = Sequential()
generator.add(Dense(784,input_shape=[100]))
generator.add(LeakyReLU())
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(392))
generator.add(LeakyReLU())
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(196))
generator.add(LeakyReLU())
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(784))
generator.add(Reshape([28,28,1]))

discriminator = Sequential()
discriminator.add(Dense(1,input_shape=[28,28,1]))
discriminator.add(Flatten())
discriminator.add(Dense(392))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.5))
discriminator.add(Dense(196))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.5))
discriminator.add(Dense(98))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.5))
discriminator.add(Dense(1,activation='sigmoid'))

GAN =Sequential([generator,discriminator])
discriminator.compile(optimizer='adam',loss='binary_crossentropy')
discriminator.trainable = False
GAN.compile(optimizer='adam',loss='binary_crossentropy')

epochs = 30
batch_size = 100
noise_shape=100

with tf.device('/gpu:0'):
 for epoch in range(epochs):
    print(f"Currently on Epoch {epoch+1}")
    
    
    for i in range(train_input.shape[0]//batch_size):
        
        if (i+1)%50 == 0:
            print(f"\tCurrently on batch number {i+1} of {train_input.shape[0]//batch_size}")
            
        
        
        train_dataset = train_input[i*batch_size:(i+1)*batch_size]
       
        #training discriminator on real images
        train_label=np.ones(shape=(batch_size,1))
        discriminator.trainable = True
        d_loss_real=discriminator.train_on_batch(train_dataset,train_label)
        
        #training discriminator on fake images
        noise=np.random.normal(size=[batch_size,noise_shape])
        gen_image = generator.predict_on_batch(noise)
        train_label=np.zeros(shape=(batch_size,1))
        d_loss_fake=discriminator.train_on_batch(gen_image,train_label)
        
        
        #training generator 
        noise=np.random.normal(size=[batch_size,noise_shape])
        train_label=np.ones(shape=(batch_size,1))
        discriminator.trainable = False
        
        d_g_loss_batch =GAN.train_on_batch(noise, train_label)
        
        
        
       
    #plotting generated images at the start and then after every 10 epoch
    if epoch % 2 == 0:
        samples = 10
        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, 100)))

        for k in range(samples):
            plot.subplot(2, 5, k+1)
            plot.imshow(x_fake[k].reshape(28, 28), cmap='gray')
            plot.xticks([])
            plot.yticks([])

        plot.tight_layout()
        plot.show()