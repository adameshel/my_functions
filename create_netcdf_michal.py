#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:10:49 2020

@author: adameshel
"""

from netCDF4 import Dataset
import numpy as np


def create_netcdf(name='example.nc', description='Example simulation data', axes=('time', 'y', 'x'),
                  dimensions=(3, 3, 4), intervals=(3600, 390, 390), fields_keys=('u', 'v'),
                  fields_values=(np.random.random((3, 3, 4)), np.random.random((3, 3, 4)))):

    root_grp = Dataset(name, 'w', format='NETCDF4')
    root_grp.description = description

    # axes
    for i in range(len(axes)):
        root_grp.createDimension(axes[i], dimensions[i])
        variable = root_grp.createVariable(axes[i], 'f4', (axes[i],))
        variable[:] = np.linspace(0, intervals[i], dimensions[i])

    # fieldss
    for j in range(len(fields_keys)):
        data = root_grp.createVariable(fields_keys[j], 'f8', axes)
        slicer = tuple([slice(None)] * np.ndim(fields_values[j]))
        data[slicer] = fields_values[j]

    root_grp.close()