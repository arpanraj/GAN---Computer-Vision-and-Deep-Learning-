# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy
import pandas
import matplotlib.pyplot as plot
import tensorflow as tf
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape

SIZE = 28
FLAT_SIZE = 784
CHANNEL = 1

# step 1 Design the GAN

gen_sub_network = Sequential()
gen_sub_network.add(Dense(784,input_shape=[100]))
gen_sub_network.add(LeakyReLU())
gen_sub_network.add(BatchNormalization(momentum=0.7))
gen_sub_network.add(Dense(392))
gen_sub_network.add(LeakyReLU())
gen_sub_network.add(BatchNormalization(momentum=0.7))
gen_sub_network.add(Dense(196))
gen_sub_network.add(LeakyReLU())
gen_sub_network.add(BatchNormalization(momentum=0.7))
gen_sub_network.add(Dense(FLAT_SIZE))
gen_sub_network.add(Reshape([SIZE,SIZE,CHANNEL]))
gen_sub_network.summary()

dis_sub_network = Sequential()
dis_sub_network.add(Dense(1,input_shape=[SIZE,SIZE,CHANNEL]))
dis_sub_network.add(Flatten())
dis_sub_network.add(Dense(392))
dis_sub_network.add(LeakyReLU())
dis_sub_network.add(Dropout(0.5))
dis_sub_network.add(Dense(196))
dis_sub_network.add(LeakyReLU())
dis_sub_network.add(Dropout(0.5))
dis_sub_network.add(Dense(98))
dis_sub_network.add(LeakyReLU())
dis_sub_network.add(Dropout(0.5))
dis_sub_network.add(Dense(1,activation='sigmoid'))
dis_sub_network.summary()

GAN_network =Sequential([gen_sub_network,dis_sub_network])
dis_sub_network.compile(optimizer='adam',loss='binary_crossentropy')
dis_sub_network.trainable = False
GAN_network.compile(optimizer='adam',loss='binary_crossentropy')


# Step - 2 Preprocessing 

# def standard_channel(channel):
#     mean = numpy.mean(channel)
#     std = numpy.std(channel)
#     if std != 0:
#         channel = (channel - mean) / std
#     return channel

dataset = tf.keras.datasets.fashion_mnist
(train_input, train_output), (test_input, test_output) = dataset.load_data()
del train_output
del test_output

# proc_train_input = []
# for image in train_input:
#     image = standard_channel(image)
#     proc_train_input.append(image)
# train_input = numpy.array(proc_train_input)
# del image
# del proc_train_input

# proc_test_input = []
# for image in test_input:
#     image = standard_channel(image)
#     proc_test_input.append(image)
# test_input = numpy.array(proc_test_input)
# del image
# del proc_test_input


EPOCHS = 100
BATCH_SIZE = 100
NOICE_SHAPE=100
train_input=train_input.reshape(len(train_input),28,28,1)
train_input =  train_input.astype('float32')
train_input = train_input/255 #normalisation

losses = []
with tf.device('/gpu:0'):
    for epoch in range(EPOCHS):
        print("Epoch" + str(epoch))
        for batch in range(train_input.shape[0]//BATCH_SIZE):
            
            #real training
            real_images = train_input[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            real_output=numpy.ones(shape=(BATCH_SIZE,1))
            dis_sub_network.trainable = True
            d_loss_real=dis_sub_network.train_on_batch(real_images,real_output)
            
            #fake training
            noisey_images=numpy.random.normal(size=[BATCH_SIZE,NOICE_SHAPE])
            fake_images = gen_sub_network.predict_on_batch(noisey_images)
            fake_output=numpy.zeros(shape=(BATCH_SIZE,1))
            d_loss_fake=dis_sub_network.train_on_batch(fake_images,fake_output)
            
            #training generator 
            noisey_images=numpy.random.normal(size=[BATCH_SIZE,NOICE_SHAPE])
            output=numpy.ones(shape=(BATCH_SIZE,1))
            dis_sub_network.trainable = False
        
            loss = GAN_network.train_on_batch(noisey_images, output)
            losses.append(loss)
            
            
           
        #plotting generated images at the start and then after every 10 epoch
        if epoch % 2 == 0:
            image_count = 5
            fake_images2 = gen_sub_network.predict(numpy.random.normal(loc=0, scale=1, size=(image_count, 100)))
        
            for image_id in range(image_count):
                plot.subplot(1, 5, image_id+1)
                plot.imshow(fake_images2[image_id].reshape(28, 28), cmap='gray')
                plot.xticks([])
                plot.yticks([])
            plot.show()

noisey_images=numpy.random.normal(size=[2,NOICE_SHAPE])
generated_images = gen_sub_network.predict(noisey_images)

fig,axe=plot.subplots(2,2)
arr_ind=0
for i in range(2):
    for j in range(2):
         axe[i,j].imshow(generated_images[arr_ind].reshape(28,28),cmap='gray')
         arr_ind+=1
