'''
Based on paper "Rain Rate Estimation Using Measurements From
Commercial Telecommunications Links"
by Oren Goldshtein, Hagit Messer and Artem Zinevich
'''
from __future__ import print_function
import pandas as pd
import numpy as np


def apply_inverse_power_law(A, L, a, b):
    ''' variance of the attenuation measurement error.
    Formula (1) from paper.
    A - attenuation
    L - cml length in KM
    a, b - ITU power law parameters
    '''
    return (A/(a*L))**(1.0/b)


def apply_power_law(R, L, a, b):
    ''' variance of the attenuation measurement error.
    Formula (1) from paper.
    R - rain
    L - cml length in KM
    a, b - ITU power law parameters
    '''
    return (R**b)*a*L


def calc_rain_from_atten(df):
    '''calculate the average rain rate at each cml.
    df should contain the following columns:
    A - attenuation due to rain
    L - length of the cmls in KM
    a, b - ITU power law parameters
    '''
    df['R'] = df.apply(lambda cml: apply_inverse_power_law(cml['A'],
                                                           cml['L'],
                                                           cml['a'],
                                                           cml['b']),
                       axis=1)
    return df


def calc_atten_from_rain(df):
    '''calculate the attenuation from the avg. rain.
    df should contain the following columns:
    R - rain
    L - length of the cmls in KM
    a, b - ITU power law parameters
    '''
    df['A'] = df.apply(lambda cml: apply_power_law(cml['R'],
                                                   cml['L'],
                                                   cml['a'],
                                                   cml['b']),
                       axis=1)
    return df


def create_virtual_gauges(df, gauge_length=0.5, num_gauges=None):
    ''' split each cml (a single row of df) into several virtual
    rain gauges.
    gauge_length - the distance between two gauges in KM.
    num_gauges - number of virtual gauges per cml (overrides gauge_length)
    df should contain the following columns:
    xa - longitude of site A of the cml
    ya - latitude of site A of the cml
    xb - longitude of site B of the cml
    yb - latitude of site B of the cml
    L - length of the cml
    '''
    df = df.copy()
    x_gauges = []
    y_gauges = []
    z_gauges = []

    # split each cml into several virtual rain gauges
    for i, cml in df.iterrows():
        L = cml['L']
        if num_gauges is None:
            num_gauges_along_cml = int(np.ceil(L / float(gauge_length)))
        else:
            num_gauges_along_cml = num_gauges
        x, y = get_gauges_lon_lat(cml['xa'], cml['ya'],
                                  cml['xb'], cml['yb'],
                                  G=num_gauges_along_cml)

        # initial rain value for each gauge
        z = tuple([cml['R']] * num_gauges_along_cml)

        x_gauges.append(x)
        y_gauges.append(y)
        z_gauges.append(z)

    # add x, y locations of the virtual rain gauges of each cml
    df['x'] = x_gauges
    df['y'] = y_gauges

    # add initial z (rain rate) of each virtual rain gauge of each cml
    df['z'] = z_gauges
    return df


def get_gauges_lon_lat(a_lon, a_lat, b_lon, b_lat, G=2):
    """ Calculate and return longitude and latitude
    of G gauges along the cml """
    def GaugeCoords(t):
        return a_lon + t*(b_lon - a_lon), a_lat + t*(b_lat - a_lat)

    LonTuple = []
    LatTuple = []
    for i in range(1, G+1):
        GaugeCoordsTup = GaugeCoords(float(i)/(G+1))
        LonTuple.append(GaugeCoordsTup[0])
        LatTuple.append(GaugeCoordsTup[1])

    return tuple([tuple(LonTuple), tuple(LatTuple)])


def error_variance(A, Q, L, a, b):
    ''' variance of the attenuation measurement error.
    Formula (7) from paper.
    A - attenuation
    Q - quantizatin
    L - length in KM
    a, b - ITU power law parameters
    '''
    if A < 0.001:    # no rain induced attenuation
        A = 0.001

    return ((Q**2)/12.0) * (1/(a*L)) * (1.0/b) * (A**(2.0*(1.0-b)/b))


def grid_weights(R, ROI, method, p_par):
    ''' calculate shepard IDW coefficients/weights (for mapping)
    with radius of influence ROI. Formula (22) from paper. '''

    # shepard coefficients/weights
#    import pdb; pdb.set_trace()
    if method==0:
        if R < ROI:
            R = max(0.001, R)
            w = (ROI/R - 1.0)**p_par
        else:
            w = 0.0
        if w == np.inf:
            return 0.0
    else: # CRESSMAN
        if R < ROI:
            R = max(0.001, R)
            w = (ROI**p_par - R**p_par)/(ROI**p_par + R**p_par) 
        else:
            w = 0.0
        if w == np.inf:
            return 0.0
    return w

calc_grid_weights = np.vectorize(grid_weights, excluded=['ROI','method', 'p_par'])


class IdwIterative():
    def __init__(self, ROI=0.0, max_iterations=1, tolerance=0.0, method=0, p_par=2.0):
        self.ROI = ROI
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.method = method # weighting method (0-shepard, 1-cressman)
        self.p_par = p_par # power parameter of IDW

    def __call__(self, df, xgrid, ygrid, quantization):
        ''' Calculates the rain rate at every virtual gauge of every cml
        and uses the results to create a rain map with IDW interpolation.
        df is a pandas DataFrame that should include the following columns:
        x, y - (x,y) location of virtual gauges
        z - initial z value at each virtual gauge
        xa, ya, xb, yb - (x,y) location of the cml's sites
        L - length of each cml (in KM)
        a, b - ITU power law parameters
        A - attenuation due to rain only
        '''

        # each vector contains data for all the gauges of all the links
        self.df = df
        self.num_cmls = self.df.shape[0]
        self.Q = quantization

        # compute the number of virtual gauges for each cml
        self.df['num_of_gauges'] = self.df['x'].apply(len)
        self.max_num_of_gauges = self.df['num_of_gauges'].max()

        # the grid points and Z values at each point (initialized to 0)
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.Z = np.zeros((self.xgrid.shape[0], self.xgrid.shape[1]))

        # create vectors with x,y,z values of all the virtual gauges
        self.use_gauges = np.zeros((self.num_cmls, self.max_num_of_gauges),
                                   dtype=bool)
        self.gauges_x = np.zeros(self.use_gauges.shape)
        self.gauges_y = np.zeros(self.use_gauges.shape)
        self.gauges_z = np.zeros(self.use_gauges.shape)

        for cml_index, cml in self.df.iterrows():
            gauges = cml['num_of_gauges']
            self.use_gauges[cml_index, :gauges] = True
            self.gauges_x[cml_index, :gauges] = cml['x']
            self.gauges_y[cml_index, :gauges] = cml['y']
            self.gauges_z[cml_index, :gauges] = cml['z']

        self.gauges_z_prev = self.gauges_z.copy()
        import pdb; pdb.set_trace()

        # calculate the measurement error variance for each cml
        # these variance values do not change during each iteration
        self.df['variance'] = df.apply(lambda cml: error_variance(cml['A'],
                                                                  self.Q,
                                                                  cml['L'],
                                                                  cml['a'],
                                                                  cml['b']),
                                       axis=1)
        
        # calculate the radius of influence
        if self.ROI == 0.0:
            self.calc_ROI()
#        print('Radius of influence: {}'.format(self.ROI))

        # run multiple iterations to find the virtual gauge values
        self.dz_vec = []
        dz = np.inf  # change in virtual gauge values

        for i in range(self.max_iterations):
#            print('Running iteration {}'.format(i))

            # perform a single iteration on all the cmls
            self.calc_cmls_from_other_cmls()

            if self.tolerance > 0.0:
                diff_norm = np.linalg.norm(self.gauges_z - self.gauges_z_prev,
                                           axis=None)
                prev_norm = np.linalg.norm(self.gauges_z_prev, axis=None)
                dz = float(diff_norm)/(prev_norm + 1e-10)
                self.dz_vec.append(dz)
#                print('Norm change in Z values: {}'.format(dz))
                if dz <= self.tolerance:
                    break

            # update the previous z values for the next iteration
            self.gauges_z_prev[:, :] = self.gauges_z

        # calculate value at each grid point using the data from the cmls
        self.calc_grid_from_cmls()

        print('Processing finished.\n')
        return self.Z

    def calc_ROI(self):
        ''' Calculate the radius of influence by measuring the largest
        distance between any two cmls. '''

        # calculate the coordinates of each cml's center point
        cml_x_middle = self.df['x'].apply(lambda x: np.mean(x)).values
        cml_y_middle = self.df['y'].apply(lambda y: np.mean(y)).values

        cml_coords_middle = np.concatenate((cml_x_middle.reshape(-1, 1),
                                            cml_y_middle.reshape(-1, 1)),
                                           axis=1)
        dests = cml_coords_middle

        # calculate the distance between every pair of cmls
        subts = cml_coords_middle[:, None, :] - dests
        cml_distances = np.sqrt(np.einsum('ijk,ijk->ij', subts, subts))

        # use the max distance between any two cmls as the ROI
        self.ROI = cml_distances.max(axis=None)

    def calc_cmls_from_other_cmls(self):
        ''' Isolate one cml at a time and compute the influence of other cmls
        in its radius of influence to calculate the rain rate at each of
        the cml's virtual gauges. '''

        # compute the rain rate at each gauge of a cml (with index cml_i)
        # using the rain rate of virtual gauges from all the OTHER cmls
        for cml_i, cml in self.df.iterrows():    # loop over cmls
            cml_gx = cml['x']   # x position of virtual gauges
            cml_gy = cml['y']   # y position of virtual gauges

            # initial new virtual gauge rain vector for current cml
            cml_num_of_gauges = cml['num_of_gauges']
            cml_new_z = np.zeros((cml_num_of_gauges,))

            # loop over current cml's virtual gauges
            for gauge_i in range(cml_num_of_gauges):
                gx = cml_gx[gauge_i]  # x position of current gauge
                gy = cml_gy[gauge_i]  # y position of currnet gauge

                # calculate distance of current gauge from all gauges
                distances = np.sqrt((self.gauges_x - gx)**2.0 +
                                    (self.gauges_y - gy)**2.0)

                # calculate IDW weights
                weights = calc_grid_weights(distances, 
                    self.ROI, 
                    self.method, 
                    self.p_par)
                weights = weights * self.use_gauges
                weights[cml_i, :] = 0.0  # remove weights for current cml
                weights[weights < weights.max()/100.0] = 0.0

                # find the indices of cmls in the current cml's ROI
                cmls_in_ROI = (weights.sum(axis=1) > 0.0)
                gauges_in_ROI = (weights > 0.0)

                # exclude current cml from all further calculations
                cmls_in_ROI[cml_i] = False
                gauges_in_ROI[cml_i, :] = False

                # variances and number of gauges of cmls in ROI
                variances = self.df['variance'][cmls_in_ROI].values
                # num_of_gauges = self.df['num_of_gauges'][cmls_in_ROI].values

                # Initialize a covariance matrix to zero
                M = gauges_in_ROI.sum()  # total number of gauges in ROI
                cov = np.zeros((M, M))
                # import pdb; pdb.set_trace()
                # add measurement quantization error to the cov matrix
                for tt, sigma in enumerate(variances):
                    start = cml_i * np.sum(gauges_in_ROI[tt, :])
                    stop = start + np.sum(gauges_in_ROI[tt, :])
                    cov[start:stop, start:stop] = sigma
                    # if tt==3:
                        # import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()

                # select the indices of cml gauges (only in the ROI)
                select_gauges = gauges_in_ROI * self.use_gauges

                # add IDW weights to covariance matrix
                z = 1.0
                weights_vector = weights[select_gauges].flatten()
                weights_vector /= weights_vector.sum(axis=None)  # Normalize
                W = z*np.diagflat(1.0/weights_vector)  # 1/weights on diagonal
                cov = cov + W

                # compute inverse of covariance matrix
                cov_inv = np.linalg.inv(cov)
                cov_inv_sum = cov_inv.sum(axis=None)  # sum of all elements

                if cov_inv_sum > 0.0:
                    cov_inv_col_sum = cov_inv.sum(axis=0)
                    prev_theta = self.gauges_z_prev[select_gauges]  # a vector
                    nominator = (cov_inv_col_sum * prev_theta).sum()
                    cml_new_z[gauge_i] = nominator / cov_inv_sum
                # if cov_inv_sum == 0.0, cml_new_z[gauge_i] stays zero

            # Apply formula (20) from paper to compute the rain rate vector
            K = cml_num_of_gauges
            R = cml['R']
            b = cml['b']
            theta = cml_new_z**b
            r = (R**b - (1.0/K)*np.sum(theta) + theta)
            r[r < 0.0] = 0.0  # set negative rain rates to 0

            # update the new z values at the cml's virtual gauges
            self.gauges_z[cml_i, :cml_num_of_gauges] = r**(1.0/b)

    def calc_grid_from_cmls(self):
        ''' calculate the z values of each grid point using z values of the
        virtual gauges '''
        Z_flat = self.Z.flatten()
        # use the cmls to caluculate the rain rate at each (x,y) grid point
        i=0
        for i in range(len(self.xgrid.flatten())):
            px = self.xgrid.flatten()[i]
            py = self.ygrid.flatten()[i]
#            import pdb; pdb.set_trace()
            # perform basic IDW
            dist = np.sqrt((self.gauges_x-px)**2 + (self.gauges_y-py)**2)
            weights = calc_grid_weights(dist, self.ROI, self.method, self.p_par)
#            import pdb; pdb.set_trace()
            # use only the exact number of gauges of every cml
            weights[np.bitwise_not(self.use_gauges)] = 0.0

            sum_of_weights = weights.sum(axis=None)
            num_gauges_in_ROI = np.sum(weights > 0.0, axis=None)

            if num_gauges_in_ROI > 1:  # more than 1 gauge in ROI
                Z_flat[i] = (weights * self.gauges_z).sum()\
                                / sum_of_weights
                i += 1
            else:
                Z_flat[i] = np.nan
                i += 1
        self.Z = Z_flat.reshape((self.xgrid.shape[0], self.xgrid.shape[1]))
        
                    
              ## This was written by Daniel and fits a grid in which 
              ## lon doesnt change with lat (and vice versa), i.e like in
              ## the simulatior
#    def calc_grid_from_cmls(self):
#        ''' calculate the z values of each grid point using z values of the
#        virtual gauges '''
#
#        # use the cmls to caluculate the rain rate at each (x,y) grid point
#        for xi in range(self.xgrid.shape[1]):
#            for yi in range(self.ygrid.shape[0]):
#                px = self.xgrid[0, xi]
#                py = self.ygrid[yi, 0]
#
#                # perform basic IDW
#                dist = np.sqrt((self.gauges_x-px)**2 + (self.gauges_y-py)**2)
#                weights = calc_grid_weights(dist, self.ROI, self.method, self.p_par)
##                import pdb; pdb.set_trace()
#                # use only the exact number of gauges of every cml
#                weights[np.bitwise_not(self.use_gauges)] = 0.0
#
#                sum_of_weights = weights.sum(axis=None)
#                num_gauges_in_ROI = np.sum(weights > 0.0, axis=None)
#
#                if num_gauges_in_ROI > 1:  # more than 1 gauge in ROI
#                    self.Z[yi, xi] = (weights * self.gauges_z).sum()\
#                                      / sum_of_weights
#                else:
#                    self.Z[yi, xi] = np.nan