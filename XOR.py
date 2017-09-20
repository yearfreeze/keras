from keras.layers import Dense,Input
from keras.models import Model
from keras.optimizers import SGD

#from numpy import *
import numpy as np

x_train=np.array([[0,0],[0,1],[1,0],[1,1]])
y_train=np.array([0,1,1,0])
x_test=np.array([[1,1],[1,0]])
y_test=np.array([0,1])

#print x_train,y_test
x=Input(shape=(2,))
h=Dense(2,activation='relu')(x)
#y=Dense(12,activation='relu')(b)
#d=Dense(12,activation='relu')(c)
y=Dense(1)(h)
model=Model(inputs=x,outputs=y)

sgd=SGD(lr=0.01,decay=1e-6)

model.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy'])
			
k=model.fit(x_train,y_train,epochs=10,batch_size=3) #k是一个history对象 用来画图

result=model.predict(x_test)
print 'fuck the answer',result