"""
Algorithm to separate a given training set into input X and outputs y
"""

import numpy as np


def split_train(sequence, numInputs, numOutputs, numJumps):
    """ Returns sets to train a model
        i.e. X[0] = sequence[0], ..., sequence[numInputs]
             y[0] = sequence[numInputs+1], ..., sequence[numInputs+numOutputs]
             ...
             X[k] = sequence[k*numJumps], ..., sequence[k*numJumps+numInputs]
             y[k] = sequence[k*numJumps+numInputs+1], ..., sequence[k*numJumps+numInputs+numOutputs]
    
    Parameters:
        sequence (array)  : Full training dataset
        numInputs (int)   : Number of inputs X used at each training
        numOutputs (int)  : Number of outputs y used at each training
        numJumps (int)    : Number of sequence samples to be ignored between (X,y) sets

    Returns:
        X (2D array): Array of numInputs arrays.
                      len(X[k]) = numInputs
        y (2D array): Array of numOutputs arrays 
                      len(y[k]) = numOutputs
                      
    """
    
    X, y = list(), list()
    
    if (numInputs+numOutputs > len(sequence)):
        print("To have at least one X,y arrays, the sequence size needs to be bigger than numInputs+numOutputs")
        return X, y
    
    for i in range(len(sequence)):
        i = numJumps*i;
        end_ix = i + numInputs

        # Once train data crosses time series length return   
        if end_ix+numOutputs > len(sequence):
            break
        
        seq_x = sequence[i:end_ix] 
        X.append(seq_x)
        seq_y = sequence[end_ix:end_ix+numOutputs]
        y.append(seq_y)
        
    return X, y


def split_train_variableInput(sequence, minSamplesTrain, numOutputs, numJumps):
    """ Returns sets to train a model with variable input length
        i.e. X[0] = sequence[0], ..., sequence[minSamplesTrain]
             y[0] = sequence[0], ..., sequence[minSamplesTrain+numOutputs]
             ...
             X[k] = sequence[0], ..., sequence[k*numJumps+minSamplesTrain]
             y[k] = sequence[0], ..., sequence[k*numJumps+minSamplesTrain+numOutputs]
             
    Parameters:
        sequence (array)       : Full training dataset
        minSamplesTrain (int)  : Minimum number of inputs X used at each training
        numOutputs (int)       : Number of outputs y used at each training
        numJumps (int)         : Number of sequence samples to be jumped between (X,y) sets

    Returns:
        X (2D array): Array of input arrays.
                      len(X[k]) = minSamplesTrain + k*numJumps
        y (2D array): Array of numOutputs arrays 
                      len(y[k]) = minSamplesTrain+numOutputs+k*numJumps

    """
    
    end_ix=0; i=0;
    X, y = list(), list()
    
    if (minSamplesTrain+numOutputs > len(sequence)):
        print("To have at least one X,y arrays, the sequence size needs to be bigger than minSamplesTrain+numOutputs")
        return X, y
        
    # Iterate through all validation splits
    while 1:
        end_ix = minSamplesTrain + numJumps*i;
        
        # Training X
        seq_x = sequence[0:end_ix] 
        X.append(seq_x)
        
        # Training y
        seq_y = sequence[end_ix:end_ix+numOutputs]
        y.append(seq_y)

        i+=1;
          
        # Once val data crosses time series length return   
        if ((minSamplesTrain + numJumps*i + numOutputs) > len(sequence)):
            break
            
    return X, y
