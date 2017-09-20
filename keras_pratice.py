# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:14:01 2017

@author: freeze
"""

import numpy
import theano
import theano.tensor as T

rng=numpy.random
N=400
feats=784
D_size=rng.randn(N,feats)   #对于设计矩阵 行代表维数 400行*784列
#代表400个784维度的向量
D_feats=rng.randint(size=N,low=0,high=2)
#大于等于0 小于2的N个特征
D=(D_size,D_feats)

training_steps=10000
x=T.dmatrix("x")
y=T.dvector("y")

w=theano.shared(rng.randn(feats),name="w")  #weight
b=theano.shared(0.,name="b")                #bias

print 'Initial model:'
print w.get_value()
print b.get_value()
#construct theano expression graph
p_1=1/(1+T.exp(-T.dot(x,w)-b))
prediction=p_1>0.5
xent=-y*T.log(p_1)-(1-y)*T.log(1-p_1)
cost=xent.mean()+0.01*(w**2).sum()
gw,gb=T.grad(cost,[w,b])

#compile
train=theano.function(inputs=[x,y],outputs=[prediction,xent],updates=((w,w-0.1*gw),(b,b-0.1*gb)))
predict=theano.function(inputs=[x],outputs=prediction)
#train
for i in range(training_steps):
        pred,err=train(D[0],D[1])
    
print 'Final model'
print w.get_value()
print b.get_value()
print 'target values for D:'
print D[1]
print 'predict on D:'
print (predict(D[0]))