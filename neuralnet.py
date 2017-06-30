# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:45:23 2017

@author: Mecit Yaman, mecityaman@gmail.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import pandas as pd
import os, pickle, time
# use companian file setup.py
import setup

tf.logging.set_verbosity(tf.logging.INFO)

# load hyperparameters

with open ('working_model.mdl', 'rb') as fp:
    folder = pickle.load(fp)
print('\n Working in folder /\t\n',folder)
for i in os.listdir(os.getcwd()+'/'+folder):
    print('\t\t\t-',i)
with open (folder+'/hyperparameters.b', 'rb') as fp:
    h_params = pickle.load(fp)

# number of hidden layers and units
# single hidden layer with 100 nodes is used.
# see here https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

h_units         = [100]
fit_steps       = 3000
ev_steps        = 1000
p_drop          = 0.05
wait            = 5

n_fibers            = h_params['n_fibers']
n_chemical_classes  = h_params['n_test_chemicals'][1]-h_params['n_test_chemicals'][0]
n_chemicals         = h_params['n_train_chemicals'][1]-h_params['n_train_chemicals'][0]
n_measurements      = h_params['n_train']


print('\n\n')
for i in sorted(h_params):
    print('{:>20}  {:<10}'.format(i, str(h_params[i])))
print()
print('{:>20}  {:<10}'.format('hidden units', str(h_units)))
print('{:>20}  {:<10}'.format('steps', fit_steps))
print('{:>20}  {:<10}'.format('ev_steps', ev_steps))
print('{:>20}  {:<10}'.format('dropout', p_drop))
print()
print('{:>20}  {:.2f}'. format('suggested nodes', n_measurements / (2*(n_fibers+n_chemicals))))
print('{:>20}  {:.2f}'. format('avr measurement', n_measurements / n_chemicals))
print('\n\n')

for _ in range(wait):
    time.sleep(1)
    print('.', end=' ')

#load datasets
training_file   = folder+'/train.csv'
eval_file       = folder+'/eval.csv'

training_set    = tf.contrib.learn.datasets.base.load_csv_without_header(
                filename        = training_file,
                target_dtype    = np.int,
                features_dtype  = np.float32)
  
eval_set        = tf.contrib.learn.datasets.base.load_csv_without_header(
                filename        = eval_file,
                target_dtype    = np.int,
                features_dtype  = np.float32)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension = n_fibers)]

classifier      = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=h_units,
                                            dropout=p_drop,
                                            n_classes=n_chemical_classes)

accs=[]
for _ in range(int(fit_steps/ev_steps)):

    xdata = training_set.data
    ydata = training_set.target
    
    classifier.fit(x=xdata, y=ydata, steps=ev_steps)
    print('global step: ', classifier.get_variable_value('global_step'))

    ev = classifier.evaluate(x=eval_set.data, y=eval_set.target, steps=1)
    ev_2 = classifier.evaluate(x=training_set.data, y=training_set.target, steps=1)

    print('\n\n\n\n Accuracy:',ev['accuracy'], ev_2['accuracy'],_)
    accs.append(ev['accuracy'])
    print('\a')
    print('\n\n\n')

plt.plot(accs)

# various input functions for classifier evaluation.
def f1():
    return training_set.data

def f2():
    return eval_set.data

# sorted dataset for graphical presentation of the fit.

def f3():
    ed=eval_set.data[eval_set.target.argsort()]
    et=eval_set.target[eval_set.target.argsort()]
    return ed

# binary mixture input file

def f_mix():
    mix_data = mm['array']
    mix_labels = mm['labels']
    return mix_data
    
    

l=list(classifier.predict_proba(input_fn=f3))
plt.figure()
plt.title('Evaluation')
plt.xlabel('Analytes')
plt.ylabel('Sorted measurements')
plt.imshow(l, aspect='auto', cmap='cool')




# binary mixture investogation
# will integrate this part later.
'''
mm=setup.mix_measurements(chemicals,
                    fibers,
                    n_train_chemicals=n_train_chemicals,
                    sample_size=20,
                    noise=0.05,
                    dir=folder,
                    fname='mix.csv')    
l=list(classifier.predict_proba(input_fn=f_mix))
plt.figure()
plt.imshow(l, aspect='auto', cmap='gray', interpolation='nearest' )

d=np.array(np.exp(mm['array'])).argsort()
print(d[:,-4:])
print(mm['labels'])
for i,j in enumerate(d):
    print(i,j[-3:], mm['labels'][i])


plt.yscale('log')
plt.plot(r[0])



def in_f():
    i_=eval_set.target.argsort()
    return eval_set.data[i_]

pre=list(classifier.predict(input_fn=in_f))
pro=list(classifier.predict_proba(input_fn=in_f))

print(pre)

for i in range(0):
    plt.yscale('log')
    plt.plot(pro[i])



plt.figure()
for i in range(0):
    if eval_set.target[i] !=pre[i]:
        print(i, eval_set.target[i], pre[i])
    
    
plt.figure()
plt.imshow(pro, cmap='cool', aspect='auto', interpolation='nearest')
        


mm=setup.mix_measurements(chemicals,
                            fibers,
                            n_train_chemicals=n_train_chemicals,
                            sample_size=2,
                            noise=0.01,
                            dir=folder,
                            fname='mix.csv')     
data=mm['array']
'''
