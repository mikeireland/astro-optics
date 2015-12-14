from __future__  import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import os
import pylab as pl

def ft_filter(grid_size=4,alpha=0.2):
    """
    Parameters
    ----------
    alpha: float
        The minimum value of amplitude reduction calculated. "Wiggles" will get
        1/alpha times bigger.
    """
    # Read in the lightforge doc
    dir = os.path.dirname(__file__)
    filename = dir + "\\piaa_grid_" + str(grid_size) + ".dat"
    
    dd = np.loadtxt(filename) 
    dx = 0.01
    dd = dd[1:,1:]
    nx = len(dd)

    #One cycle per 0.1m is equivalent to how many cycles per FOV?
    fov = dx *nx
    x = (np.arange(nx)-nx/2)
    xy = np.meshgrid(x,x) #in mm

    #Create a Fourier amplitude filter that matches our knowledge that
    #"at about 10 cycles/mm, amplitudes are reduced by a factor of 2"
    gauss_sigma = fov/0.1/2.35
    amp_reduction = np.exp(-(xy[0]**2 + xy[1]**2)/2.0/gauss_sigma**2)
#    amp_reduction = np.fft.fftshift( (amp_reduction + 0.1)/1.1 ) #Dodgy non divide by zero
    amp_reduction = np.fft.fftshift( np.maximum(amp_reduction,alpha) ) #Dodgy non divide by zero
    dd_filtered = np.real(np.fft.ifft2(np.fft.fft2(dd)/amp_reduction))
    
    pl.imshow(dd_filtered)
    
    return dd_filtered


