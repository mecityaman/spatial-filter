# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 23:03:34 2018

@author: myaman
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

debug_mode = False

def get_filter_matrix(n):
    ### create broad filters, sigma = 300 cm-1
    ### n samples from 400 to 40000 cm-1.
    wavelength = np.linspace(400.0,4000.0,901)
    filter_matrix = np.zeros((901,n))
    for _,cavity_pos in enumerate(np.linspace(650,3750,n)):
        intensity = np.exp(-(wavelength-cavity_pos)**2/200000)
        filter_matrix[:,_]=intensity
        if debug_mode:
            print(cavity_pos)
            plt.imshow(filter_matrix, aspect='auto')
            plt.show()
            plt.imshow(filter_matrix, aspect='auto')
            plt.show()
    return filter_matrix


def get_chemicals_matrix(n, m):
    ### n, number of measurement data
    ### m, number of unique chemicals
    ### Bruker Tensor 27 FT-IR & OPUS Library
    ### 117 FTIR spectrums
    ###
    chemicals_matrix = np.zeros((n,901))
    labels = np.random.randint(0,m,n)
    #one hot representation
    lmatrix = np.zeros((n,m))
    for i,j in enumerate(labels):
        lmatrix[i,j]=1
    
    for _,label in enumerate(labels):
        chemicals_matrix[_,:] = chemicals[label].values
        if debug_mode:
            print(_)
            plt.imshow(chemicals_matrix, aspect='auto')
            plt.show()

            plt.imshow(chemicals_matrix, aspect='auto')
            plt.show()
    return chemicals_matrix, lmatrix, labels


def get_measurements_matrix(chemicals_matrix, filter_matrix, noise_factor=0.1):
    
    ### measurement data
    ### factor 0.0  signal-to-noise ratio 12.0
    ### factor 0.1  signal-to-noise ratio  7.6
    ### factor 0.2  signal-to-noise ratio  4.5
    
    measurements = np.matmul(chemicals_matrix,filter_matrix)
    noise = np.random.randn(*measurements.shape)*noise_factor
    measurements = measurements + noise
    if debug_mode:
        plt.title('measurements')
        plt.imshow(measurements, aspect='auto')
        plt.show()
    return measurements.astype('float32') #required dtype for TensorFlow
        

def get_signal_to_noise_ratio(noise_factor):
    l = [] #store signal to noise ratios for each analyte
    for i in range(116):    
        a_chemical = chemicals[i]
        a_chemical = a_chemical+np.random.randn(901)*noise_factor
        signal_to_noise =  a_chemical.mean()/a_chemical.std()
        #print('factor %3.2f  signal-to-noise ratio %4.1f'%( noise_factor,signal_to_noise))
        l.append(signal_to_noise)
    #plt.plot(a_chemical,'c',alpha=.5) #plot a sample
    #plt.show()
    #plt.plot(l)
    return np.array(l).mean()



num_pixels          = 100
num_chemicals       = 100
num_measurements    = 500
# noise factors 0.0, 0.1, 0.2 correspond to
# signal to noise ratios 10.0, 7.0, 4.0.
noise_factor        = 1.0


# read chemical FTIR library
chemicals = pd.read_pickle('data\chemicals 0-117.pkl')

# imaging array pixels
filter_matrix = get_filter_matrix(100)

# generate and store measurement and test data
for i in ['train','test']:
    
    directory ='data/'+str(noise_factor)+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    chemicals_matrix, lmatrix, labels = get_chemicals_matrix(num_measurements,num_chemicals)
    measurements = get_measurements_matrix(chemicals_matrix,filter_matrix, noise_factor=noise_factor)

    np.save(directory+i+'_measurements',measurements)
    np.save(directory+i+'_labels', labels)
    np.save(directory+i+'_lmatrix', lmatrix)

    plt.title(i+' measurements')
    plt.imshow(measurements,aspect='auto')
    plt.show()
    plt.title(i+' one hot matrix')
    plt.imshow(lmatrix, cmap='gray', aspect='auto')
    plt.show()

print('factor %3.2f  signal-to-noise ratio %4.1f'%( noise_factor,get_signal_to_noise_ratio(noise_factor)))

