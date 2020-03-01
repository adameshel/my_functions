#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:00:39 2018

@author: adameshel
"""
import numpy as np

def power_law_a_b(f_GHz, pol, approx_type='ITU'):
    ### function from pycomlink library: https://github.com/pycomlink ###
    """Approximation of parameters for A-R relationship
    
    f_GHz : int, float or np.array of these Frequency of the microwave link in GHz (1GHz to 100GHz)
    pol : str Polarization of the microwave link 'h'/'H'/'v'/'V'
    approx_type : str, optional
            Approximation type (the default is 'ITU', which implies parameter
            approximation using a table recommanded by ITU)
            
    Returns a,b : float Parameters of A-R relationship      
    References
    ----------
    .. [4] ITU, "ITU-R: Specific attenuation model for rain for use in 
        prediction methods", International Telecommunication Union, 2013 
         
    """
    from scipy.interpolate import interp1d

    f_GHz = np.asarray(f_GHz)

    if f_GHz.min() < 1 or f_GHz.max() > 100:
        raise ValueError('Frequency must be between 1 Ghz and 100 GHz.')
    else:
        if pol == 'V' or pol == 'v':
            f_a = interp1d(ITU_table[0, :], ITU_table[2, :], kind='cubic')
            f_b = interp1d(ITU_table[0, :], ITU_table[4, :], kind='cubic')
        elif pol == 'H' or pol == 'h':
            f_a = interp1d(ITU_table[0, :], ITU_table[1, :], kind='cubic')
            f_b = interp1d(ITU_table[0, :], ITU_table[3, :], kind='cubic')
        else:
            ValueError('Polarization must be V, v, H or h.')
        a = f_a(f_GHz)
        b = f_b(f_GHz)
    return a, b

ITU_table = np.array([
  [1.000e+0, 2.000e+0, 4.000e+0, 6.000e+0, 7.000e+0, 8.000e+0, 1.000e+1, 
   1.200e+1, 1.500e+1, 2.000e+1, 2.500e+1, 3.000e+1, 3.500e+1, 4.000e+1, 
   4.500e+1, 5.000e+1, 6.000e+1, 7.000e+1, 8.000e+1, 9.000e+1, 1.000e+2],
  [3.870e-5, 2.000e-4, 6.000e-4, 1.800e-3, 3.000e-3, 4.500e-3, 1.010e-2,
   1.880e-2, 3.670e-2, 7.510e-2, 1.240e-1, 1.870e-1, 2.630e-1, 3.500e-1, 
   4.420e-1, 5.360e-1, 7.070e-1, 8.510e-1, 9.750e-1, 1.060e+0, 1.120e+0],
  [3.520e-5, 1.000e-4, 6.000e-4, 1.600e-3, 2.600e-3, 4.000e-3, 8.900e-3,
   1.680e-2, 3.350e-2, 6.910e-2, 1.130e-1, 1.670e-1, 2.330e-1, 3.100e-1,
   3.930e-1, 4.790e-1, 6.420e-1, 7.840e-1, 9.060e-1, 9.990e-1, 1.060e+0],
  [9.120e-1, 9.630e-1, 1.121e+0, 1.308e+0, 1.332e+0, 1.327e+0, 1.276e+0,
   1.217e+0, 1.154e+0, 1.099e+0, 1.061e+0, 1.021e+0, 9.790e-1, 9.390e-1,
   9.030e-1, 8.730e-1, 8.260e-1, 7.930e-1, 7.690e-1, 7.530e-1, 7.430e-1],
  [8.800e-1, 9.230e-1, 1.075e+0, 1.265e+0, 1.312e+0, 1.310e+0, 1.264e+0, 
   1.200e+0, 1.128e+0, 1.065e+0, 1.030e+0, 1.000e+0, 9.630e-1, 9.290e-1,
   8.970e-1, 8.680e-1, 8.240e-1, 7.930e-1, 7.690e-1, 7.540e-1, 7.440e-1]])
    
    
  
    
def _calc_A_min_max(tx_min, tx_max, rx_min, rx_max, gT=1.0, gR=0.6, window=7):
    """Calculate rain rate from attenuation using the A-R Relationship
    Parameters
    ----------
    gT : float, optional
        induced bias
    gR : float, optional
        induced bias
    window: int, optional
        number of previous measurements to use for zero-level calculation
    Returns
    -------
    float or iterable of float
        Ar_max
    Note
    ----
    Based on: "Empirical Study of the Quantization Bias Effects in
    Commercial Microwave Links Min/Max Attenuation
    Measurements for Rain Monitoring" by OSTROMETZKY J., ESHEL A.
    """
     # quantization bias correction
    Ac_max = tx_max - rx_min + (gT + gR) / 2
    Ac_min = tx_min - rx_max - (gT + gR) / 2
    Ac_max[np.isnan(Ac_max)] = np.rint(np.nanmean(Ac_max))
    Ac_min[np.isnan(Ac_min)] = np.rint(np.nanmean(Ac_min))
     # zero-level calculation
    Ar_max = np.full(Ac_max.shape, 0)
    for i in range(len(Ac_max)):
        Ar_max[i] = Ac_max[i] - Ac_min[max(0, i-window):i+1].min()
    return Ar_max
    
