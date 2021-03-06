# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 21:04:19 2017

@author: freeze
"""

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
import numpy as np
model = Sequential()
model.add(Dense(input_dim=2,output_dim=2))
model.add(Activation("relu"))
model.add(Dense(input_dim=2,output_dim=1))
model.add(Activation("relu"))
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.11, nesterov=True)
model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])
print "begin to train"
list1 = [1,1]
label1 = [0]
list2 = [1,0]
label2 = [1]
list3 = [0,0]
label3 = [0]
list4 = [0,1]
label4 = [1]
train_data = np.array((list1,list2,list3,list4))
label = np.array((label1,label2,label3,label4))
 
model.fit(train_data,label,nb_epoch = 1000,batch_size = 4,verbose = 1,shuffle=True)
list_test = [0,1]
test = np.array((list_test,list1))
classes = model.predict(test)
print classes