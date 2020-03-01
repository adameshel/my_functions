from math import sin, cos, sqrt, atan2, radians, asin
import numpy as np
from pyproj import Proj, transform


def degrees_to_meters(lon, lat):

        proj_degrees = Proj(init='epsg:4326')
        proj_meters = Proj(init='epsg:3395')
        x, y = transform(proj_degrees, proj_meters, lon, lat)
        return x, y
    
    
def meters_to_degrees(x, y):
    proj_degrees = Proj(init='epsg:4326')
    proj_meters = Proj(init='epsg:2039')
#    proj_degrees = Proj(init='epsg:4326')
#    proj_meters = Proj(init='epsg:3395')
    lon, lat = transform(proj_meters, proj_degrees, x, y)
    return lon, lat  
    

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km


def deg2km(x1,x2,y1,y2):
    """Function to calculate distances between lon lat coordinates
    Haversine formula"""
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(y1)
    lon1 = radians(x1)
    lat2 = radians(y2)
    lon2 = radians(x2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c 
    return distance



def geo2utm(lat_vec,lon_vec,lon0=9):
    # function EN = geo2utm(lat,lon,zone)
    # lon0 is the meridian of longitude
    # Results in km
    # simplified formulas taken from Wikipedia
    # https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
    a = 6378.137
    f = 1.0/298.257223563
    #lon0 = zone#32->9?-sweden, 36->33-israel
    #lon0 = zone#33->15?-sweden?
    # lon0 = 33*np.pi/180
    lat = lat_vec.copy()
    lon = lon_vec.copy()
    lat *= np.pi/180
    lon *= np.pi/180
    
    lon0 *= np.pi/180
    
    N0 = 0
    k0 = 0.9996
    E0 = 500
    
    n = f/(2-f)
    A = a/( 1 + n )*(1 + (1/4)*n**2 + (1/64)*n**4 + (1/256)*n**6 + (25/16384)*n**8 + (49/65536)*n**10)
    alpha = [n/2-(n**2)*2/3+(n**3)*5/16,(n**2)*13/48-(n**3)*3/5,(n**3)*61/240]
    # beta = [n/2-(n**2)*2/3+(n**3)*37/96 (n**2)*1/48+(n**3)*1/15 (n**3)*17/480]
    # delta = [n*2-(n**2)*2/3+(n**3)*2 (n**2)*7/3-(n**3)*8/5 (n**3)*56/15]
    
    # lat/lon->(E,N)
    t = np.sinh(np.arctanh(np.sin(lat)) - (2*np.sqrt(n)/(1+n))*np.arctanh((2*np.sqrt(n)/(1+n))*np.sin(lat)))
    xsi = np.arctan(t/np.cos(lon-lon0))
    etta = np.arctanh(np.sin(lon-lon0)/np.sqrt(1 + t**2))
    ind = [0,1,2,3]
    # sigma = 1 + sum(2*ind*alpha*np.cos(2*ind*xsi)*np.cosh(2*ind*etta))
    # tao = sum(2*ind*alpha*np.sin(2*ind*xsi)*np.sinh(2*ind*etta))
    
    # E = E0 + k0*A*(etta + sum(alpha*np.cos(2*ind*xsi)*np.sinh(2*ind*etta)))
    E = E0 + k0*A*(etta + (alpha[0]*np.cos(2*ind[1]*xsi)*np.sinh(2*ind[1]*etta))\
                   + (alpha[1]*np.cos(2*ind[2]*xsi)*np.sinh(2*ind[2]*etta))\
                   + (alpha[2]*np.cos(2*ind[3]*xsi)*np.sinh(2*ind[3]*etta)))
    # N = N0 + k0*A*(xsi + sum(alpha*np.sin(2*ind*xsi)*np.cosh(2*ind*etta)))
    N = N0 + k0*A*(xsi + (alpha[0]*np.sin(2*ind[1]*xsi)*np.cosh(2*ind[1]*etta))\
                   + (alpha[1]*np.sin(2*ind[2]*xsi)*np.cosh(2*ind[2]*etta))\
                   + (alpha[2]*np.sin(2*ind[3]*xsi)*np.cosh(2*ind[3]*etta)))

    return np.array(E), np.array(N)



def utm2geo(E,N,lon0=9):
    # function latlon = utm2geo(E,N,lon0)
    # simplified formulas taken from Wikipedia
    # https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
    #lon0 = zone;%32->9?-sweden, 36->33-israel
    #lon0 = zone;%33->15?-sweden?
    
    # hami = 1;% hemisphere; for southern -1
    lon0 *= np.pi/180
    
    a = 6378.137
    f = 1/298.257223563
    N0 = 0
    k0 = 0.9996
    E0 = 500
    
    n = f/(2-f)
    A = a/( 1 + n )*(1 + (1/4)*n**2 + (1/64)*n**4 + (1/256)*n**6 +\
           (25/16384)*n**8 + (49/65536)*n**10)
    beta = [n/2-(n**2)*2/3+(n**3)*37/96, (n**2)*1/48+(n**3)*1/15, (n**3)*17/480]
    delta = [n*2-(n**2)*2/3+(n**3)*2, (n**2)*7/3-(n**3)*8/5, (n**3)*56/15]
    
    xi = (N-N0)/(k0*A)
    eta = (E-E0)/(k0*A)
    
    xi_tag = xi - beta[0]*np.sin(2*1*xi)*np.cosh(2*1*eta)\
        - beta[1]*np.sin(2*2*xi)*np.cosh(2*2*eta)\
        - beta[2]*np.sin(2*3*xi)*np.cosh(2*3*eta)
    
    eta_tag = eta - beta[0]*np.cos(2*1*xi)*np.sinh(2*1*eta)\
        - beta[1]*np.cos(2*2*xi)*np.sinh(2*2*eta)\
        - beta[2]*np.cos(2*3*xi)*np.sinh(2*3*eta)
    
#    sigma_tag = 1 - 2*beta[0]*np.cos(2*1*xi)*np.cosh(2*1*eta)\
#        - 2*beta[1]*np.cos(2*2*xi)*np.cosh(2*2*eta)\
#        - 2*beta[2]*np.cos(2*3*xi)*np.cosh(2.*3*eta)
    
#    tau_tag = 2*beta[0]*np.sin(2*1*xi)*np.sinh(2*1*eta)\
#        + 2*beta[1]*np.sin(2*2*xi)*np.sinh(2*2*eta)\
#        + 2*beta[2]*sin(2*3*xi)*np.sinh(2*3*eta)
    
    chi = np.arcsin(np.sin(xi_tag)/np.cosh(eta_tag))
    
    lat = chi + delta[0]*np.sin(2*1*chi)\
        + delta[1]*np.sin(2*2*chi)\
        + delta[2]*np.sin(2*3*chi)
    
    lon = lon0 + np.arctan(np.sinh(eta_tag)/np.cos(xi_tag))
    
    lat /= np.pi/180
    lon /= np.pi/180
    
    return np.array(lon), np.array(lat)