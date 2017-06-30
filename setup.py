# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:35:24 2017
@author: Mecit Yaman, mecityaman@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import pandas as pd

import random

import os, time, pickle


'''
FTIR absorption files for fiber materials PES, PEI, PE
As2S5, As2Se3 are IR-transparent in the interested region.
'''

def read_polymers_file():
    polymers = pd.read_csv('repo/polymer_data', index_col=0)
    return polymers

    
'''
Fibers that are similar in peak profile, ie. bandwith and absorption 
characteristics, to real thermally drawn fibers are generated.
w1                  : initial fiber peak position
w2                  : final fiber peak position
band_with_factor    : 1 is the same as fibers reported in 
                      A. Yildirim et al. Advanced Materials 23, 1263 (2011) and
                      M. Yaman et al., Analytical Chemistry 84, 83 (2012)  
n                   : number of fibers to be generated.
dir                 : directory to record fiber panda database files.
'''
    
def generate_fibers(w1=400.0, w2=4000.0, bandwith_factor=1.0, n=6, dir=''):
    
    polymers = read_polymers_file()

    # polymer and chemical data index column
    # is in the form np.linspace(400.0, 4000.0, 901)
    
    x = np.linspace(400.0, 4000.0, 901)

    # drawn fiber characteristic gaussian parameters
    # A. Yildirim et al. Advanced Materials 23, 1263 (2011) and
    # M. Yaman et al., Analytical Chemistry 84, 83 (2012)   
   
    x_lit = np.array([750, 920, 1010, 1650, 3100, 3630], dtype = np.float32)
    y_lit = np.array([1200, 1500, 2800, 19600, 49600, 85600], dtype = np.float32)

    #fiber spectral positions and bandwidths
    
    x_new  = np.linspace(w1, w2, n)
    y_new  = np.exp(np.interp(x_new,x_lit, np.log(y_lit)))
    
    fibers = pd.DataFrame(index=x)
    
    for i, (a,b) in enumerate(zip(x_new,y_new)):

        polymer_absorption  = polymers['PES']
        
        spectral_position   = a        
        bandwith            = bandwith_factor * b        
       
        fibers[i]           = polymer_absorption * np.exp (
                            -((x-spectral_position)**2)/bandwith)
    
    filename = str(dir)+'/fibers '+str(n)+'.csv'
    fibers.to_csv(filename)
    print('fibers    ready: ',filename)
    return fibers


'''
Functions to return, and draws a numpy array that is the Euclid distance
between to FTIR profiles of chemicals, used as a proxy for chemical similarity.
'''

def similarity(i,j):
    s=sum((chemicals[i]-chemicals[j])**2)
    return s


def draw_similarity_matrix(n=117):
    m=np.array([[similarity(i,j) for i in range(n)] for j in range(n)])
    plt.imshow(m, cmap='gray')
    return m


'''
Reads chemicals FTIR absorption pandas database from a set directory
Total number of volatile organic chemicals is 118, selected from a 
commercial library, Bruker Tensor 27 FT-IR & OPUS Library.
'''

def read_chemicals_file(a=0, b=117):
    
    chemicals   = pd.read_csv('repo/chopped', index_col='wavenumber')
    chemicals   = chemicals.rename(columns={j:int(i) for i,j in enumerate(chemicals.columns)})
    new = chemicals[list(range(a,b))].copy(deep=False)
    new = new.rename(columns={j:i for i,j in enumerate(new.columns)})

    folder      = time.ctime().replace(':','_').replace(' ','_')
    filename    = folder+'/chemicals '+ "{}-{}".format(a,b)+'.csv'    
    os.mkdir(folder)
    new.to_csv(filename)
    print('chemicals ready: ',filename)
    return new, folder

'''
Function to record single analyte measurement simulations with the fiber array.
Tests a randomly chosen chemical with a random concentration within 0-100
dynamic range.

n_train_chemicals       : # of chemicals used in training/evaluation phase. 
mixed                   : is it a mixture?
noise                   : random noise to add to the measurements.
sample_size             : # of measurement recordings
plotgraph               : graphical output of measurements.
                          should be used with sample size 1.
dir                     : directory to record measurement panda database files.
fname                   : filenam to record pandas measurment files
'''
                              
def single_molecule_measurement_simulation(chemicals,
                                           fibers,
                                           n_train_chemicals='',
                                           is_test_data = False,
                                           mixed = False,
                                           noise = 0.01,
                                           sample_size = 1,
                                           plotgraph = False,
                                           dir = '',
                                           fname=''):
    
    qr_array    = []
    labels      = []
    print('\n\n')        
                                           
    for m in range(sample_size):
        
        if is_test_data:
            # choose from the universal set
            chem_no = random.choice(chemicals.columns)

        else:
            # choose from a smaller training set
            chem_no = random.choice(list(range(*n_train_chemicals)))

        # concentration dynamic range (0.1-100 relative units)
        c       = np.random.rand()*100 + 1.001 

        noise_  = pd.DataFrame.copy(chemicals, deep=False)    
        
        noise_['noise'] = np.random.randn(len(chemicals))*noise    
        noisy_chemical  = chemicals[chem_no]+noise_['noise']

        # if mixed=True then occasionally (p=0.9) add another chemical and stir.

        if mixed:
            if np.random.rand()>0.90:
                # chemicals no 15 and 11 used for test purposes.
                
                chem_no_2   = 15 #random.choice(list(range(*n_train_chemicals)))
                chem_no     = 11 #random.choice(list(range(*n_train_chemicals)))
                noisy_chemical  = (chemicals[chem_no]*chemicals[chem_no_2])+noise_['noise']
                print('mixing', end=' ')
                if np.random.rand()>0.5:
                    chem_no = chem_no_2
                    print('swapping', end='')
                print()
        # list to store quenching ratios
        qrs = []

        for i in fibers:
            
            #random noise
            noise_['noise'] = np.random.randn(len(chemicals))*noise
            noisy_fiber = noise_['noise'] + fibers[i]
            
            # measurement signal calculation
            signal = noisy_fiber*(noisy_chemical**np.log(c))
            
            if m==sample_size-1:            
        
                plt.plot(signal,alpha=0.5)
                plt.plot(noisy_fiber, alpha=0.5)
                plt.plot(noisy_chemical+1, alpha=0.5)
            intensity_i = np.sum(noisy_fiber)
            intensity_q = np.sum(signal)
            qr          = intensity_q / intensity_i
            qrs.append(qr)
            
            
                        
        
        #inset
        labels.append(chem_no)
        
        print(m,chem_no, end='  ') # len(qrs), qrs)
        qr_array.append(qrs)
    
    # array for quenching ratios, one for each fiber
    
    res = np.array(qr_array)
    
    # Graphic presentation of chemicals, fibers and measurement signal.
    
    if plotgraph:
        plt.figure()
        plt.plot(noisy_chemical+1, alpha=0.3)
        
        for i in fibers:
            noise_['noise'] = np.random.randn(len(chemicals))*noise
            noisy_fiber = noise_['noise'] + fibers[i]
        
            plt.plot(noisy_fiber, alpha=0.8)
            signal = noisy_fiber*(noisy_chemical**np.log(c))

            plt.plot(signal,alpha=0.6)            

        plt.plot(noise_['noise']-0.5, alpha=0.4)


        inset = plt.axes([1.0, .40, .5, .45])
        plt.xticks([])
        plt.yticks([])
        inset.axis('off')
        plt.bar(range(len(qrs)), qrs, alpha=0.3)
        plt.plot(range(len(qrs)), np.ones_like(range(len(qrs))), '--', alpha=0.7)
         
        plt.figure()
        plt.title('measurment array for training set')
        plt.imshow(res, interpolation='nearest', cmap='gray', vmin=0, vmax=1, aspect = 'auto')
    
   

    pp = pd.DataFrame(res)
    pp['labels']=labels
    filename = dir+'/'+fname
    pp.to_csv(filename, index=False, header=False)
    
    #returns a numopy matrix row: measurments, column: fiber quenching ratio
    
    return {'tensorflow': pp,
            'array': res,
            'labels': labels}

'''
Function to record binary mixture measurement simulations with the fiber array.
Tests two randomly chosen chemicals with a random concentration within 0-100
dynamic range.

n_train_chemicals       : # of chemicals used in training/evaluation phase. 
mixed                   : is it a mixture?
noise                   : random noise to add to the measurements.
sample_size             : # of measurement recordings
plotgraph               : graphical output of measurements.
                          should be used with sample size 1.
dir                     : directory to record measurement panda database files.
fname                   : filenam to record pandas measurment files

for more information see above, single_molecule_measurement_simulation() definition
'''
      
def mix_measurements(chemicals,
                     fibers,
                     n_train_chemicals='',
                     sample_size=1,
                     noise=0.01,
                     dir='f',
                     fname=''):

    qr_array=[]
    labels=[]
    
    for m in range(sample_size):
        chem_no_1 = random.choice(list(range(*n_train_chemicals)))
        chem_no_2 = random.choice(list(range(*n_train_chemicals)))
        
        #concentration
        c       = np.random.rand()*100 + 1.001

        noise_  = pd.DataFrame.copy(chemicals, deep=False)
        noise_['noise'] = np.random.randn(len(chemicals))*noise    
        
        noisy_chemical  = (chemicals[chem_no_1]+ chemicals[chem_no_2])/2+noise_['noise']
        
        qrs = []
        for i in fibers:
            noise_['noise'] = np.random.randn(len(chemicals))*noise
            noisy_fiber = noise_['noise'] + fibers[i]
            signal = noisy_fiber*(noisy_chemical**np.log(c))

            intensity_i = np.sum(noisy_fiber)
            intensity_q = np.sum(signal)
            qr          = intensity_q / intensity_i
            qrs.append(qr)
        
        qr_array.append(qrs)
        # binary labels are recorden        
        labels.append((chem_no_1, chem_no_2))
        print(m,'[',chem_no_1,chem_no_2,']', end='\n')

    res = np.array(qr_array)
    
    pp = pd.DataFrame(res)
    pp['labels'] = labels
    
    filename = dir+'/'+fname
    pp.to_csv(filename, index=False, header=False)
    
    return {'tensorflow': pp,
            'array': res,
            'labels': labels}
   


if __name__=='__main__':
    
    n_test_chemicals    = 0, 100        # analyte superset used for testing
    n_train_chemicals   = 0, 85         # training set
    
    n_train, n_test     = 10000, 10000        # number of training/testing measurement sets
    
    w1, w2              = 1000, 3500    # fiber wavenumber scale
    n_fibers            = 100           # number of fibers
    bandwith_factor     = 1             # regular fibers

    noise_              = 0.05           # random noise 
    
    
    # load universal chemical set
    # gives back a unique foldername for measurment data
    # folder name is used below to record measurments
    
    chemicals, folder  = read_chemicals_file(*n_test_chemicals)   
    
    #generate simulation fibers
    fibers              = generate_fibers(w1=w1,
                                          w2=w2,
                                          n=n_fibers,
                                          bandwith_factor=bandwith_factor,
                                          dir=folder)
    
    #training measurements
    
    train_ = single_molecule_measurement_simulation(
                           chemicals,
                           fibers,
                           n_train_chemicals    = n_train_chemicals,
                           is_test_data         = False,
                           noise                = noise_, 
                           sample_size          = n_train,
                           mixed                = False,
                           plotgraph            = True,
                           dir                  = folder,
                           fname                = 'train.csv')
    
    eval_ = single_molecule_measurement_simulation(
                           chemicals,
                           fibers,
                           n_train_chemicals    = n_train_chemicals,
                           is_test_data         = True,
                           noise                = noise_, 
                           sample_size          = n_test,
                           mixed                = False,
                           plotgraph            = False,
                           dir                  = folder,
                           fname                = 'eval.csv')
                                                   
        
 
    
    h_params = {'n_train_chemicals' : n_train_chemicals,
                'n_test_chemicals'  : n_test_chemicals,
               'n_fibers'           : n_fibers,
               'noise_'             : noise_,
               'w1-w2'              : (w1,w2),
               'bandwith_factor'    : bandwith_factor,
               'n_train'            : n_train,
               'n_test'             : n_test
               }
 

    print('\n\n')
    for i in sorted(h_params):
        print('{:>20}  {:<10}'.format(i, str(h_params[i])))
    print()
    
    n_fibers        = h_params['n_fibers']
    n_chemicals     = h_params['n_train_chemicals'][1]-h_params['n_train_chemicals'][0]
    n_measurements  = h_params['n_train']

    print('{:>20}  {:.2f}'. format(
            'suggested nodes', n_measurements / ( 5*(n_fibers+n_chemicals))))

    plt.figure()
    plt.title('Anlayte similarity matrix')
    draw_similarity_matrix(n_chemicals)

    print('\n\n')
    print('{:>20}  {:<20}'. format('training set:',str(set(train_['labels']))))
    print('{:>20}  {:<20}'. format('evaluation set:',str(set(eval_['labels']))))
    
    #records hyperparameters for the neural network trainer/tester.
    
    with open(folder+'/hyperparameters.b','wb') as fp:
        pickle.dump(h_params,fp)

    with open('working_model.mdl','wb') as fp:
        pickle.dump(folder,fp)
        
