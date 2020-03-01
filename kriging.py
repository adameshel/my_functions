#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Dec 11 12:33:17 2018

@author: adameshel
'''
# All from :  http://connor-johnson.com/2014/03/20/simple-kriging-in-python/
#from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
import sys
 
def SVh( P, h, bw ):
    '''
    Experimental semivariogram for a single lag
    '''
    p_d = squareform( pdist( P[:,:2] ) )
    N = p_d.shape[0]
    Z = list()
    for i in range(N):
        for j in range(i,N):
            if( p_d[i,j] >= h-bw )and( p_d[i,j] <= h+bw ):
                Z.append( ( P[i,2] - P[j,2] )**2.0 )
    if len(Z)==0:
        return -1
    return np.sum( Z ) / ( 2.0 * len( Z ) )
 
def SV( P, hs, bw ):
    '''
    Experimental variogram for a collection of lags
    '''
    sv = list()
    for h in hs:
#         sv_temp =  SVh( P, h, bw )
        sv.append( SVh( P, h, bw ) )
    sv = [ [ hs[i], sv[i] ] for i in range( len( hs ) ) if sv[i] != -1 ]
    return np.array( sv ).T
 
def C( P, h, bw ):
    '''
    Calculate the sill
    '''
    c0 = np.var( P[:,2] )
    if h == 0:
        return c0
    return c0 - SVh( P, h, bw )

#def func_to_opt(h, C0, a):
#    '''
#    Gaussian or Spherical model
#    '''
#    # Gaussian
##    return C0 * (1.0 - np.exp(-(3.0 * h / a) ** 2))
#    # Spherical
#    h[h>=a]=a
#    return C0*( 1.5*h/a - 0.5*(h/a)**3.0 )

def func_to_opt(h, C0, a):
    '''
    Select: Gaussian or Spherical model
    The Spherical is truncated to lag/range < 1 to include the rising part of
    the function only. The nugget effect was set to be 0.1 of the sill, as
    calculated by the Dutch.
    '''
    nugget = C0*0.1
#     # Gaussian
##    func = C0 * (1.0 - np.exp(-(3.0 * h / a) ** 2)) + nugget
#     
#    # Spherical
    func = C0*( 1.5*h/a - 0.5*(h/a)**3.0 ) + nugget
    func[h>=a] = C0 + nugget # Dutch metodology of truncating Spherical func.
#    func[func>=C0] = C0
    
    return func


def func_with_noise(h, C0, a):
    '''
    Select: Gaussian or Spherical model
    The Spherical is truncated to lag/range < 1 to include the rising part of
    the function only. The nugget effect was set to be 0.1 of the sill, as
    calculated by the Dutch.
    '''
#    nugget = C0*0.1
#     # Gaussian
##    func = C0 * (1.0 - np.exp(-(3.0 * h / a) ** 2)) + nugget
#     
#    # Spherical
#    func = C0*( 1.5*h/a - 0.5*(h/a)**3.0 ) + nugget
#    func[h>=a] = C0 + nugget # Dutch metodology of truncating Spherical func.
    
    ## trying new nugget option from 
    ## https://scikit-learn.org/0.17/auto_examples/gaussian_process/plot_gp_regression.html
    # Spherical
    func_no_nugget = C0*( 1.5*h/a - 0.5*(h/a)**3.0 ) + C0*0.1
    y = func_no_nugget.ravel()
    dy = 0.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    nugget = (dy / y) ** 2
    func = C0*( 1.5*h/a - 0.5*(h/a)**3.0 ) + C0*0.1 + nugget
    func[h>=a] = C0 + C0*0.1
    # the next row was added without checking it
    func[func>=C0+ C0*0.1] = C0
    
    
    return func


def krige( P, u, N, covfct, Kinverse ):
    '''
    Input  (P)     ndarray, data
           (model) modeling function
                    - spherical
                    - exponential
                    - gaussian
           (hs)    kriging distances
           (bw)    kriging bandwidth
           (u)     unsampled point
           (N)     number of neighboring
                   points to consider
    '''
 
    # mean of the variable
    mu = np.mean( P[:,2] )
 
    # distance between location u and each data point in P
    distances = np.sqrt( ( P[:,0]-u[0] )**2.0 + ( P[:,1]-u[1] )**2.0 )
    # apply the covariance model to the distances
    k = covfct( distances )
 
    # calculate the kriging weights
#     import pdb; pdb.set_trace()
    weights = k.dot(Kinverse)
 
    # calculate the residuals
    residuals = P[:,2] - mu
 
    # calculate the estimation
    estimation = weights.dot(residuals) + mu
 
    return estimation*(estimation>0), weights



def kriging_execute(df_krg, x_grid_utm, y_grid_utm, 
                    sill_guess=None, range_guess=25, range_cal=False,
                    timestamp=0, unit_length='m', delete_links=True, 
                    change_bw=False, bandwidth_in_km=0.3, 
                    doy=1, duration_agg=186):
    '''df_krg: dataframe which is in the form of the output of
`create_virtual_gauges.py`.
range_guess: a first guess of the decorrelation distance (in km).
range_cal: if True, the first guess of range is derived from doy and 
duration_agg.

unit_length: the units of x_grid_utm, y_grid_utm (usually 'm').
delete_links: if True- deletes links by distances from one another
while the cov. condition number is too large.
change_bw: if True- increases the band width while the
cov. condition number is too large.
doy: the Day Of Year Parameter from Van De Beek 2012.
duration_agg: duration of aggrigation time of rainfall (h). Van De Beek 2012.
'''
    if unit_length == 'm':
        unit_helper = 1e3
    else:
        unit_helper = 1
    xMax = np.max([df_krg.xa.max(), df_krg.xb.max()])
    xMin = np.min([df_krg.xa.min(), df_krg.xb.min()])
    yMax = np.max([df_krg.ya.max(), df_krg.yb.max()])
    yMin = np.min([df_krg.ya.min(), df_krg.yb.min()])

    p_prep = np.array( df_krg[['x','y','z']] )
    max_num_of_rg_in_row = len(p_prep[0,2])
    ## A loop for defining the VRGs per link where the number  
    ## of VRGs per link is not the same
    for row in range(len(p_prep[:,0])):
        num_of_rg_in_row = len(p_prep[row,0])
        p_prep[row,2] = p_prep[row,2][:num_of_rg_in_row]
        
    p = np.zeros([len(p_prep[:,0]) * max_num_of_rg_in_row,
                  len(p_prep[0,:])])
    link_num_prep = []
    link_L_prep = []
    p_row = 0
    element = 0
    for row in range(len(p_prep[:,0])):
        if row != 0:
            p_row = p_row + element +1
        for col in range(len(p_prep[0,:])):
            for element in range(len(p_prep[row,2])):
                p[p_row+element,col] = p_prep[row,col][element]
    for cut_point in range(len(p[:,0])): 
        if p[cut_point,0] == 0: 
            break
    if len(p[:,0]) == (cut_point + 1):
        p_filtered = p
    else:
        p_filtered = np.delete(p, 
                               range(cut_point,len(p_prep[:,0]) *\
                                     max_num_of_rg_in_row), 
                               0)

    for row in range(len(p_prep[:,0])):
        for element in range(len(p_prep[row,2])):
            link_num_prep.append(df_krg['Link_num'][row])
            link_L_prep.append(df_krg['L'][row])

#    import pdb; pdb.set_trace()
    df_p = pd.DataFrame(p_filtered, columns=['x','y','z'])
    df_p['l_num'] = link_num_prep
    df_p['L'] = link_L_prep
    del p
    del p_filtered
#    import pdb; pdb.set_trace()
    
#    range_guess_original = range_guess

    # Deleting one of 2 links which their centers are too close
    K_matCond = 1e9
    min_dist = [0.0]
#     dist_spacer = sys.float_info.epsilon
    dist_spacer = sys.float_info.epsilon # meters
    link_num_list_to_exclude = []
    Cond = 0
    bw_helper = 0 # km
    cond_num_thresh = 3e5
    while K_matCond >= cond_num_thresh:
        bw = bandwidth_in_km * unit_helper + bw_helper * unit_helper # bandwidth in km
        hs_factor = 0.75
    #    hs = np.arange(0,np.max([xMax-xMin,yMax-yMin]),bw)
        # hs = np.arange(0,np.std(p[:,2]),bw)
        hs = np.arange(0, np.min([np.max([xMax-xMin,yMax-yMin]), 50*unit_helper])\
                       * hs_factor, bw*2.0)
        print(Cond)
        print('bw: ' + str(bw))
        if delete_links==True:
    #        dist_spacer = (np.min(pdist( df_p[['x','y']] )) - \
    #                       np.min(pdist( df_p[['x','y']] )) / 2) * \
    #                       unit_helper + dist_spacer # 10 meters 
            dist_spacer = (np.partition(pdist( df_p[['x','y']] ), Cond)[Cond] - \
                           np.partition(pdist( df_p[['x','y']] ), Cond)[Cond] / 2)\
                           + dist_spacer  
            print('dist_spacer: ' + str(dist_spacer))
            # minimum distance between virtual gauges to allow inversion of K
            # (avoid ill-condition of K)
            min_dist = pdist( [[np.abs(x_grid_utm[1,0]-x_grid_utm[0,0]) + \
                                np.mean(x_grid_utm[:,0]),
                                np.abs(y_grid_utm[1,0]-y_grid_utm[0,0]) + \
                                np.mean(y_grid_utm[:,0])],
                              [np.abs(x_grid_utm[1,0]-x_grid_utm[0,0]) + \
                               dist_spacer + np.mean(x_grid_utm[:,0]),
                                np.abs(y_grid_utm[1,0]-y_grid_utm[0,0]) + \
                                dist_spacer + np.mean(y_grid_utm[:,0])]] )
    
            dist_temp = squareform(pdist( df_p[['x','y']] ))
            # loop over the upper part of the dist matrix
            for row in range(len(df_p)-1):
                for col in range(row+1,len(df_p)):
                    if np.max(dist_temp[row,col]) <= min_dist[0]: 
                        # name of smaller link to exclude
                        pair = [link_L_prep[row],link_L_prep[col]]
                        idx = np.argmin(pair)
#                        import pdb; pdb.set_trace()
                        if idx == 0:
                            link_num_list_to_exclude.append(link_num_prep[row])
                        else:
                            link_num_list_to_exclude.append(link_num_prep[col])
                        break
            df_p = df_p[~df_p['l_num'].isin(link_num_list_to_exclude)]
            df_p = df_p.reset_index(drop=True)

        else:
            min_dist = [0,0]

        p = df_p[['x','y','z']].values
        sv = SV( p, hs, bw )
        ## optimization of params
        if range_cal is True: # use the Van Der Beek range parameter
            range_guess = ((15.51*duration_agg**0.09+2.06*\
                            duration_agg**(-0.12)*\
                            np.cos((2*np.pi*(doy-7.37*duration_agg**0.22))/\
                                   365))**4)/1000
        if sill_guess is None:
#            sill_guess = np.median(sv[1]) + np.median(sv[1]) / 2
            sill_guess = np.var(sv[1])
            if sill_guess < 0.5:
                sill_guess = 0.5
        ## normalizing the initial guesses and the semivariance values
        magnitude_sill = 10 ** (int(np.log10(sill_guess)))
        magnitude_range = 10 ** (int(np.log10(range_guess)))
        
        sv[0] = sv[0] / unit_helper 
        sv[0] = sv[0] / magnitude_range
#        range_guess = np.min([range_guess / magnitude_range, sv[0][-1]])
        range_guess = range_guess / magnitude_range
        
        sv[1] = sv[1] / magnitude_sill
        sill_guess = sill_guess / magnitude_sill
 
        try:
            popt, pcov = curve_fit(f=func_to_opt, xdata=sv[0], ydata=sv[1], 
                                   bounds=(0, [sill_guess, range_guess])) 
            C0, a_krg = popt # C0 is the sill and a_krg is the range

        except:
            a_krg = range_guess
            C0 = sill_guess
            print('OPT func. could not converge. First guesses were used for'\
                  + ' Sill and Range.')
        a_krg = a_krg * unit_helper * magnitude_range
        sv[0] = sv[0] * unit_helper * magnitude_range
        
        C0 = C0 * magnitude_sill
        sv[1] = sv[1] * magnitude_sill
#        import pdb; pdb.set_trace()
######################################################################
######################################################################
#         C0=SILL; a_krg=RANGE
######################################################################
######################################################################
#             import pdb; pdb.set_trace()
        noise_in_function = False
        cov_func = lambda h: C0 - func_to_opt( h, C0, a_krg )
#        cov_func = lambda h: 0 if (h > a_krg) else (C0 - func_to_opt( h, C0, a_krg ))
#        import pdb; pdb.set_trace()
        # number of neighboring points to consider
        N = np.shape(p)[0]
        # form a matrix of distances between existing data points
        K = squareform( pdist( df_p[['x','y']] ) )
        #apply the covariance model to these distances
#        import pdb; pdb.set_trace()
        K = cov_func( K.ravel() )
        K = np.array( K )
        K = K.reshape( N,N )
        K_matCond = np.linalg.cond( K ) # condition number
        
        if K_matCond > cond_num_thresh:
            noise_in_function = True
            cov_func = lambda h: C0 - func_with_noise( h, C0, a_krg )
    #        import pdb; pdb.set_trace()
            # number of neighboring points to consider
            N = np.shape(p)[0]
            # form a matrix of distances between existing data points
            K = squareform( pdist( df_p[['x','y']] ) )
            #apply the covariance model to these distances
            K = cov_func( K.ravel() )
            K = np.array( K )
            K = K.reshape( N,N )
            K_matCond = np.linalg.cond( K ) # condition number
            
        Cond = Cond + 1
        bw_helper += 0.05 * change_bw
        print('K_matCond: ' + str(K_matCond))

    K_inv = np.linalg.inv( K )
    
    df_exclude_link = pd.DataFrame({'l_num': link_num_list_to_exclude})
    df_exclude_link.l_num.unique()    
    print('condition number = ' + str(K_matCond) + '\n', 'dist_spacer = ' + \
          str(dist_spacer) + '\n', 'C0 = ' + str(C0) + '\n', 'a_krg = ' + \
          str(a_krg) + '\n', 'min_dist = ' + str(min_dist[0]) + '\n',\
          'bandwidth (bw) = ' + str(bw) + '\n', 'noise_in_cov_function = ' +\
          str(noise_in_function) + '\n',\
          'exclude links = ' + str(df_exclude_link.l_num.values) )
    ## plotting the semivariogram
    plt.fig, ax = plt.subplots(figsize=(10,5))
    ax.plot( sv[0], sv[1], '.-' )
    plt.xlim(0,np.max(sv[0]))
    ax.plot( sv[0], func_to_opt( sv[0], C0, a_krg ) )
    ax.plot( sv[0], cov_func(sv[0]) )
    ax.set_title('Spherical Model, timeframe: ' + str(timestamp))
    ax.set_ylabel('Semivariance')
    ax.set_xlabel('Lag [m]')
    
    rainfield_flat = np.zeros(np.shape(x_grid_utm.flatten()))
#         temp_weights = np.zeros(np.shape(py_xgrid_UTM.flatten()))
    for i, (xi, yj) in enumerate(zip(x_grid_utm.flatten(),
                                 y_grid_utm.flatten())):
        rainfield_flat[i], _ = krige( p, (xi,yj), N, cov_func, K_inv)
    gridded_snapshot = rainfield_flat.reshape(np.shape(x_grid_utm))

    print(' max rain intensity estimated = ' + str(np.max(gridded_snapshot)))


    df_krg = df_krg[~df_krg['Link_num'].isin(df_exclude_link.l_num.values)]
    if 'index' in df_krg.columns:
        df_krg.drop('index', axis=1, inplace=True)
    if 'level_0' in df_krg.columns:
        df_krg.drop('level_0', axis=1, inplace=True)
    df_krg = df_krg.reset_index()
    print('Processing finished')
    
    return gridded_snapshot, df_krg
#######################################################
#######################################################
