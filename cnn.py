from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
import keras
from keras.utils import plot_model
#generate dummy data
import numpy as np
import matplotlib.pyplot as plt

x_train=np.random.random((1000,20))
y_train=keras.utils.to_categorical(np.random.randint(10,size=(1000,1)),num_classes=10)
x_test=np.random.random((100,20))
y_test=keras.utils.to_categorical(np.random.randint(10,size=(100,1)),num_classes=10)

model=Sequential()


model.add(Dense(64,activation='relu',input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
r_fit=model.fit(x_train,y_train,epochs=20,batch_size=128)

score=model.evaluate(x_test,y_test,batch_size=128)
print ('total loss on Testing set:',score[0])
print ('accurary of Testing set:',score[1])

#model.save('f:\\first_model.h5')
#plot_model(model)
plt.plot(r_fit.epoch,r_fit.history['acc'],label="acc")


result= model.predict(x_test)
print 'fuck the model predict'
print result