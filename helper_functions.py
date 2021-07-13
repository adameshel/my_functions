import datetime as dt
import numpy as np


def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac


import os.path
from os import listdir

def split_at(s, c, n):
    '''Function for splitting strings (first arg). The last argument is the position of the
        selected seperator (second arg) in the string.'''
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])

def delete_from_list(my_list, idxs):
    '''Delete items from list (my_list) by supplying a list of indeces (idxs) to delete'''
    for idx in sorted(idxs, reverse=True):
        del my_list[idx]
    return my_list


from IPython.display import display_html

def restartkernel() :
    '''Function to restart kernel. Execute: restartkernel()'''
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)

def nans(shape, dtype=float):
    a = np.empty(np.asarray(shape, dtype=int), dtype)
    a.fill(np.nan)
    return a