def simple_idw(x, y, z, xi, yi):
    # A simple IDW function
    dist = distance_matrix(x,y, xi,yi)
    power = 2 # The power parameter
    # In IDW, weights are 1 / distanceto the poewer of 2
    weights = 1.0 / dist**power
    
    # Make weights sum to one
    weights /= weights.sum(axis=0)
    
    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi
