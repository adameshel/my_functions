#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:00:58 2019

@author: adameshel
"""
import numpy as np
import pandas as pd
import xarray as xr



def rmse(predicted, observed):
    """Returns the Root Mean Square Error of 2 sets."""
    p = np.array(predicted)
    o = np.array(observed)
    RMSE = np.sqrt(np.nanmean((p - o) ** 2))
    return RMSE



def relative_diff(predicted, observed):
    """Returns the Root Mean Square Error of 2 sets.
    Will only work for comparing one snapshot to another, and not for multiple
    timestamps."""
    Difference = np.abs(predicted - observed)
    return Difference
    

def spatialCorrelation(ds_1,ds_2):
    """Calculate the spatial correlation of timestamps in xarray dataset.
    A 3D array of multiple spatial (2D) Correlation maps is returned.
    The function's input must have an attribute called `raindepth`"""
    
    if np.shape(ds_1.raindepth)[2] == 1: # if there is only one timestamp
        x = ds_1.raindepth.values[1,:].size
        y = ds_1.raindepth.values[:,1].size

    else:
        x = ds_1.raindepth.values[1,:,1].size
        y = ds_1.raindepth.values[:,1,1].size
        
    spat_corr_mat = np.zeros([y, x])
    yi=0
    xi=0
    for yi in range(y): #lat -> rows
        for xi in range(x): #lon -> columns
            spat_corr_mat[yi,xi] = np.corrcoef(
                ds_1['raindepth'].sel(y=yi, x=xi).values,
                ds_2['raindepth'].sel(y=yi, x=xi).values)[0,1]
    return spat_corr_mat



def spatialRMSE(ds_1,ds_2):
    """Calculates the spatial RMSE of timestamps in xarray dataset.
    A 3D array of multiple spatial (2D) RMSE maps is returned.
    The function's input must have an attribute `raindepth` """
    if np.shape(ds_1.raindepth)[2] == 1: # if there is only one timestamp
        x = ds_1.raindepth.values[1,:].size
        y = ds_1.raindepth.values[:,1].size

    else:
        x = ds_1.raindepth.values[1,:,1].size
        y = ds_1.raindepth.values[:,1,1].size
    spat_rmse_mat = np.zeros([y, x])
    yi=0
    xi=0
    for yi in range(y): #lat -> rows
        for xi in range(x): #lon -> columns
            spat_rmse_mat[yi,xi] = rmse(
                ds_1['raindepth'].sel(y=yi, x=xi).values,
                ds_2['raindepth'].sel(y=yi, x=xi).values)
    return spat_rmse_mat



def time_corr_spatSnapshots(groundTruth, 
                            estimation, 
                            threshold=0.0, 
                            equal_to_thresh=False, 
                            rounding=False,
                            nan_thresh=0):
    '''Ave. corr of grid for a chosen timestamp
    threshold: get rid of values not exceeding this rain intensity in both 
    GT and reconst.
    equal_to_thresh: True if you want to discard numbers smaller OR EQUAL to 
    threshold.
    rounding: assign True to round values to 3 decimals.'''
    time_corr_vec = np.zeros(groundTruth.time.size) # number of timestamps
    i=0
    for i in range(len(time_corr_vec)):
        # import pdb; pdb.set_trace()
#        import pdb; pdb.set_trace()
        timestamp = str(groundTruth['time'].values[i])
        if groundTruth['raindepth'].sel(time=timestamp).max() < nan_thresh:
            # exclude very weak rain cells
            time_corr_vec[i] = np.nan
        else:
            nan_pairs = ~np.logical_or(
                np.isnan(groundTruth['raindepth'].sel(
                    time=timestamp).values.flatten()), 
                np.isnan(estimation['raindepth'].sel(
                    time=timestamp).values.flatten()))
            
            groundTruth_for_corr = np.compress(
                nan_pairs,
                groundTruth['raindepth'].sel(time=timestamp).values.flatten())
            est_for_corr = np.compress(
                nan_pairs,
                estimation['raindepth'].sel(time=timestamp).values.flatten())
            if rounding is True:
                groundTruth_for_corr = np.round(groundTruth_for_corr, 3)
                est_for_corr = np.round(est_for_corr, 3)
                
            ################
            # get rid of values smaller than Threshold for the correlation
            if equal_to_thresh is False:
                pairs = ~np.logical_and(groundTruth_for_corr < threshold,
                                        est_for_corr < threshold) 
            else:
                pairs = ~np.logical_and(groundTruth_for_corr <= threshold,
                                        est_for_corr <= threshold)
                
            groundTruth_for_corr = np.compress(pairs,
                                               groundTruth_for_corr)
            est_for_corr = np.compress(pairs,
                                       est_for_corr)
            ################
            
    #        import pdb; pdb.set_trace()
            time_corr_vec[i] = \
                np.corrcoef(groundTruth_for_corr, est_for_corr)[0,1]
    return time_corr_vec



def time_RMSE_spatSnapshots(groundTruth, 
                            estimation, 
                            threshold=0.0, 
                            equal_to_thresh=False, 
                            normalize=False, 
                            rounding=False,
                            nan_thresh=0):
    '''Ave. corr of grid for a chosen timestamp
    threshold: get rid of values not exceeding this rain intensity in 
    both GT and reconst.
    normalize: boolian indicatind if you wish to devide rmse by mean rainfall.
    equal_to_thresh: True if you want to discard numbers smaller OR EQUAL to 
    threshold.
    rounding: assign True to round values to 3 decimals.'''
    time_RMSE_vec = np.zeros(groundTruth.time.size) # number of timestamps
    # import pdb; pdb.set_trace()
    i=0

    for i in range(len(time_RMSE_vec)):
#        import pdb; pdb.set_trace()
        timestamp = str(groundTruth['time'].values[i])
        if groundTruth['raindepth'].sel(time=timestamp).max() < nan_thresh:
            # exclude very weak rain cells
            time_RMSE_vec[i] = np.nan
        else:
    #        import pdb; pdb.set_trace()
            nan_pairs = ~np.logical_or(
                np.isnan(groundTruth['raindepth'].sel(
                    time=timestamp).values.flatten()), 
                np.isnan(estimation['raindepth'].sel(
                    time=timestamp).values.flatten()))
            
            groundTruth_for_RMSE = np.compress(
                nan_pairs,
                groundTruth['raindepth'].sel(time=timestamp).values.flatten())
            est_for_RMSE = np.compress(
                nan_pairs,
                estimation['raindepth'].sel(time=timestamp).values.flatten())
            
            if rounding is True:
                groundTruth_for_RMSE = np.round(groundTruth_for_RMSE, 3)
                est_for_RMSE = np.round(est_for_RMSE, 3)
            
            ################
    #         get rid of values smaller than Threshold for the correlation
    #        import pdb; pdb.set_trace()
            if equal_to_thresh is False:
                pairs = ~np.logical_and(groundTruth_for_RMSE < threshold,
                                        est_for_RMSE < threshold) 
            else:
                pairs = ~np.logical_and(groundTruth_for_RMSE <= threshold,
                                        est_for_RMSE <= threshold) 
            groundTruth_for_RMSE = np.compress(pairs,
                                               groundTruth_for_RMSE)
            est_for_RMSE = np.compress(pairs,
                                       est_for_RMSE)
    #        import pdb; pdb.set_trace()
            ################
            if normalize is False:
                time_RMSE_vec[i] = rmse(groundTruth_for_RMSE, est_for_RMSE)
            else:
                time_RMSE_vec[i] = \
                    rmse(groundTruth_for_RMSE, est_for_RMSE)/np.nanmean(
                        groundTruth_for_RMSE)
            #print('rmse normalized by the domains mean rain intensity')
    return time_RMSE_vec


def time_RMSE_spatSnapshots_bins(groundTruth, 
                                estimation, 
                                threshold=0.0, 
                                equal_to_thresh=False, 
                                normalize=False, 
                                rounding=False,
                                nan_thresh=0):
    '''
    threshold: get rid of values not exceeding this rain intensity in both GT 
    and reconst.
    normalize: boolian indicatind if you wish to devide rmse by mean rainfall.
    equal_to_thresh: True if you want to discard numbers smaller OR EQUAL to 
    threshold.
    rounding: assign True to round values to 3 decimals.'''
    time_RMSE_vec = np.zeros(groundTruth.time.size) # number of timestamps
    # import pdb; pdb.set_trace()
    i=0

    for i in range(len(time_RMSE_vec)):
#        import pdb; pdb.set_trace()
        timestamp = str(groundTruth['time'].values[i])
        if groundTruth.sel(time=timestamp).max().values < nan_thresh:
            # exclude very weak rain cells
            time_RMSE_vec[i] = np.nan
        else:
    #        import pdb; pdb.set_trace()
            nan_pairs = ~np.logical_or(
                np.isnan(groundTruth.sel(time=timestamp).values.flatten()), 
                np.isnan(estimation.sel(time=timestamp).values.flatten()))
            
            groundTruth_for_RMSE = np.compress(
                nan_pairs,
                groundTruth.sel(time=timestamp).values.flatten())
            est_for_RMSE = np.compress(
                nan_pairs,
                estimation.sel(time=timestamp).values.flatten())
            
            if rounding is True:
                groundTruth_for_RMSE = np.round(groundTruth_for_RMSE, 3)
                est_for_RMSE = np.round(est_for_RMSE, 3)
            
            ################
    #         get rid of values smaller than Threshold for the correlation
    #        import pdb; pdb.set_trace()
            if equal_to_thresh is False:
                pairs = ~np.logical_and(groundTruth_for_RMSE < threshold,
                                        est_for_RMSE < threshold) 
            else:
                pairs = ~np.logical_and(groundTruth_for_RMSE <= threshold,
                                        est_for_RMSE <= threshold) 
            groundTruth_for_RMSE = np.compress(pairs,
                                               groundTruth_for_RMSE)
            est_for_RMSE = np.compress(pairs,
                                       est_for_RMSE)
    #        import pdb; pdb.set_trace()
            ################
            if normalize is False:
                time_RMSE_vec[i] = rmse(groundTruth_for_RMSE, est_for_RMSE)
            else:
                time_RMSE_vec[i] = \
                    rmse(groundTruth_for_RMSE, est_for_RMSE)/np.nanmean(
                        groundTruth_for_RMSE)
            #print('rmse normalized by the domains mean rain intensity')
    return time_RMSE_vec
