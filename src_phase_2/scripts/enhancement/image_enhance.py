# -*- coding: utf-8 -*-

from enhancement.ridge_segment import ridge_segment
from enhancement.ridge_orient import ridge_orient
from enhancement.ridge_freq import ridge_freq
from enhancement.ridge_filter import ridge_filter
import numpy as np

def image_enhance(img):
    blksze = 16;
    thresh = 0.1;
    normim,mask = ridge_segment(img,blksze,thresh);             # normalise the image and find a ROI

    gradientsigma = 1;
    blocksigma = 7;
    orientsmoothsigma = 7;
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma);              # find orientation of every pixel

    blksze = 38;
    windsze = 5;
    minWaveLength = 5;
    maxWaveLength = 15;
    freq,medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,maxWaveLength);    # find the overall frequency of ridges
    
    freq = medfreq*mask;
    kx = 0.65;ky = 0.65;
    newim = ridge_filter(normim, orientim, freq, kx, ky);       # create gabor filter and do the actual filtering
    newim = newim < -3
    newim = newim.astype(np.uint8)
    print(newim)
    return(newim)
