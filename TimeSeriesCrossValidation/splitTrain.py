"""
Algorithm to separate a given training set into input X and outputs y
"""

import numpy as np

def split_train(sequence, n_steps_input, n_steps_forecast, n_steps_jump):
    """ Returns sets to train a model
        i.e. (X[0], X[1], ..., X[n_steps_input]),  (y[n_steps_input+1], y[n_steps_input+2], ..., y[n_steps_input+n_steps_output]])
        (X[n_steps_jump], ..., X[n_steps_jump+n_steps_input]),  (y[n_steps_jump+n_steps_input+1], ..., y[n_steps_jump+n_steps_input+n_steps_output]])
    
    Parameters:
        sequence (array): Full training dataset
        n_steps_input (int): Number of inputs X used at each training
        n_steps_forecast (int): Number of outputs y used at each training
        n_steps_jump (int): Number of sequence samples to be ignored between (X,y) sets

    Returns:
        X (2D array): Array of n_steps_input arrays.
                      len(X[k]) = n_steps_input
        y (2D array): Array of n_steps_forecast arrays 
                      len(y[k]) = n_steps_forecast

    """
    X, y = list(), list()
    for i in range(len(sequence)):
        i = n_steps_jump*i;
        end_ix = i + n_steps_input

        # Once train data crosses time series length return   
        if end_ix+n_steps_forecast > len(sequence):
            break
        
        seq_x = sequence[i:end_ix] 
        X.append(seq_x)
        seq_y = sequence[end_ix:end_ix+n_steps_forecast]
        y.append(seq_y)
    return np.array(X), np.array(y)
