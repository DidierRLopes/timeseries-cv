"""
Several algorithms to split a given training dataset into train, validation and test sets.
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


def split_train_cv_forwardChaining(sequence, n_steps_input, n_steps_forecast, n_steps_jump):
    """ Returns sets to train and cross-validate a model using forward chaining technique
    
    Parameters:
        sequence (array): Full training dataset
        n_steps_input (int): Number of inputs X and Xcv used at each training
        n_steps_forecast (int): Number of outputs y and ycv used at each training
        n_steps_jump (int): Number of sequence samples to be ignored between (X,y) sets

    Returns:
        X (2D array): Array of n_steps_input arrays used for training
                      len(X[k]) = n_steps_input
        y (2D array): Array of n_steps_forecast arrays used for training
                      len(y[k]) = n_steps_forecast
        Xcv (2D array): Array of n_steps_input arrays used for cross-validation
                        len(Xcv[k]) = n_steps_input
        ycv (2D array): Array of n_steps_forecast arrays used for cross-validation
                        len(ycv[k]) = n_steps_forecast
    """
    X, y, Xcv, ycv = dict(), dict(), dict(), dict()
    j=2; # Tracks index of CV set at each train/val split
    
    # Iterate through all train/val splits
    while 1:
        start_ix=0; end_ix=0; startCv_ix=0; endCv_ix=0;
        X_it, y_it, Xcv_it, ycv_it = list(), list(), list(), list()
        i=0; # Index of individual training set at each train/val split
        
        # Iterate until index of individual training set is smaller than index of cv set
        while (i < j):
            ## TRAINING DATA
            start_ix = n_steps_jump*i;
            end_ix = start_ix + n_steps_input;
            
            seq_x = sequence[start_ix:end_ix] 
            X_it.append(seq_x)
            seq_y = sequence[end_ix:end_ix+n_steps_forecast]
            y_it.append(seq_y)
            
            i+=1;
          
        # Once val data crosses time series length return   
        if (((end_ix+n_steps_input)+n_steps_forecast) > len(sequence)):
            break
        
        ## CROSS-VALIDATION DATA
        startCv_ix = end_ix;
        endCv_ix = end_ix + n_steps_input;
        
        seq_xcv = sequence[startCv_ix:endCv_ix] 
        Xcv_it.append(seq_xcv)
        seq_ycv = sequence[endCv_ix:endCv_ix+n_steps_forecast]
        ycv_it.append(seq_ycv) 
            
        ## Add another train/val split 
        X[j-2] = np.array(X_it)
        y[j-2] = np.array(y_it)
        Xcv[j-2] = np.array(Xcv_it)
        ycv[j-2] = np.array(ycv_it)
        
        j+=1;
            
    return X, y, Xcv, ycv


def split_train_cv_kFold(sequence, n_steps_input, n_steps_forecast, n_steps_jump):
    """ Returns sets to train and cross-validate a model using K-Fold technique
    
    Parameters:
        sequence (array): Full training dataset
        n_steps_input (int): Number of inputs X and Xcv used at each training
        n_steps_forecast (int): Number of outputs y and ycv used at each training
        n_steps_jump (int): Number of sequence samples to be ignored between (X,y) sets

    Returns:
        X (2D array): Array of n_steps_input arrays used for training
                      len(X[k]) = n_steps_input
        y (2D array): Array of n_steps_forecast arrays used for training
                      len(y[k]) = n_steps_forecast
        Xcv (2D array): Array of n_steps_input arrays used for cross-validation
                        len(Xcv[k]) = n_steps_input
        ycv (2D array): Array of n_steps_forecast arrays used for cross-validation
                        len(ycv[k]) = n_steps_forecast
    """
    X, y, Xcv, ycv = dict(), dict(), dict(), dict()
    j=2; # Tracks index of CV set at each train/val split
    theEnd = 0; # Flag to terminate function
    
    # Iterate until val set falls outside time series length
    while 1:
        start_ix=0; end_ix=0; startCv_ix=0; endCv_ix=0;
        X_it, y_it, Xcv_it, ycv_it = list(), list(), list(), list()
        i=0; # Index of individual training set at each train/val split
        n=0; # Number of n_steps_jump
        
        # Iterate through all train/val splits
        while 1:
            if (i != j): 
                ## TRAINING DATA
                start_ix = endCv_ix + n_steps_jump*n;
                end_ix = start_ix + n_steps_input;
                n +=1;

                # Leave train/val split loop once training data crosses time series length
                if end_ix+n_steps_forecast > len(sequence):
                    break;

                seq_x = sequence[start_ix:end_ix] 
                X_it.append(seq_x)
                seq_y = sequence[end_ix:end_ix+n_steps_forecast]
                y_it.append(seq_y)
            else:
                ## CROSS-VALIDATION DATA
                startCv_ix = end_ix;
                endCv_ix = end_ix + n_steps_input;
                n = 0;
                
                # Once val data crosses time series length exit tran/val split loop and return
                if endCv_ix+n_steps_forecast > len(sequence):
                    theEnd = 1;
                    break;

                seq_xcv = sequence[startCv_ix:endCv_ix] 
                Xcv_it.append(seq_xcv)
                seq_ycv = sequence[endCv_ix:endCv_ix+n_steps_forecast]
                ycv_it.append(seq_ycv)
            i+=1;
        
        # Only add a train/val split if the time series length has not been crossed
        if (theEnd == 1):
            break
        
        ## Add another train/val split 
        X[j-2] = np.array(X_it)
        y[j-2] = np.array(y_it)
        Xcv[j-2] = np.array(Xcv_it)
        ycv[j-2] = np.array(ycv_it)
        
        j+=1;
            
    return X, y, Xcv, ycv

def split_train_cv_groupKFold(sequence, n_steps_input, n_steps_forecast, n_steps_jump):
    """ Returns sets to train and cross-validate a model using group K-Fold technique
    
    Parameters:
        sequence (array): Full training dataset
        n_steps_input (int): Number of inputs X and Xcv used at each training
        n_steps_forecast (int): Number of outputs y and ycv used at each training
        n_steps_jump (int): Number of sequence samples to be ignored between (X,y) sets

    Returns:
        X (2D array): Array of n_steps_input arrays used for training
                      len(X[k]) = n_steps_input
        y (2D array): Array of n_steps_forecast arrays used for training
                      len(y[k]) = n_steps_forecast
        Xcv (2D array): Array of n_steps_input arrays used for cross-validation
                        len(Xcv[k]) = n_steps_input
        ycv (2D array): Array of n_steps_forecast arrays used for cross-validation
                        len(ycv[k]) = n_steps_forecast
    """
    X, y, Xcv, ycv = dict(), dict(), dict(), dict()
    
    # Iterate through 5 train/val splits
    for j in np.arange(5):
        start_ix=0; end_ix=0; startCv_ix=0; endCv_ix=0;
        X_it, y_it, Xcv_it, ycv_it = list(), list(), list(), list()
        i=0; # Index of individual training set at each train/val split
        n=0; # Number of n_steps_jump
        
        while 1: 
            if ((i+1+j)%(5) != 0):
                # TRAINING DATA
                start_ix = endCv_ix + n_steps_jump*n;
                end_ix = start_ix + n_steps_input;
                n+=1;

                # Leave train/val split loop once training data crosses time series length
                if end_ix+n_steps_forecast > len(sequence)-1:
                    break 

                seq_x = sequence[start_ix:end_ix] 
                X_it.append(seq_x)
                seq_y = sequence[end_ix:end_ix+n_steps_forecast]
                y_it.append(seq_y)
            else:
                # CROSS-VALIDATION DATA
                startCv_ix = end_ix;
                endCv_ix = end_ix + n_steps_input;
                n=0;

                # Once val data crosses time series length return   
                if ((endCv_ix+n_steps_forecast) > len(sequence)):
                    break

                seq_xcv = sequence[startCv_ix:endCv_ix] 
                Xcv_it.append(seq_xcv)
                seq_ycv = sequence[endCv_ix:endCv_ix+n_steps_forecast]
                ycv_it.append(seq_ycv)  
                
            i+=1;
            
        ## Add another train/val split     
        X[j] = np.array(X_it)
        y[j] = np.array(y_it)
        Xcv[j] = np.array(Xcv_it)
        ycv[j] = np.array(ycv_it)
            
    return X, y, Xcv, ycv

def split_train_cv_test_forwardChaining(sequence, n_steps_input, n_steps_forecast, n_steps_jump):
    """ Returns sets to train, cross-validate and test a model using forward chaining technique
    
    Parameters:
        sequence (array): Full training dataset
        n_steps_input (int): Number of inputs X, Xcv and Xtest used at each training
        n_steps_forecast (int): Number of outputs y, ycv, ytest used at each training
        n_steps_jump (int): Number of sequence samples to be ignored between (X,y) sets

    Returns:
        X (2D array): Array of n_steps_input arrays used for training
                      len(X[k]) = n_steps_input
        y (2D array): Array of n_steps_forecast arrays used for training
                      len(y[k]) = n_steps_forecast
        Xcv (2D array): Array of n_steps_input arrays used for cross-validation
                        len(Xcv[k]) = n_steps_input
        ycv (2D array): Array of n_steps_forecast arrays used for cross-validation
                        len(ycv[k]) = n_steps_forecast
        Xtest (2D array): Array of n_steps_input arrays used for testing
                          len(Xtest[k]) = n_steps_input
        ytest (2D array): Array of n_steps_input arrays used for testing
                          len(Xtest[k]) = n_steps_input
    """
    X, y, Xcv, ycv, Xtest, ytest = dict(), dict(), dict(), dict(), dict(), dict()
    j=2; # Tracks index of CV set at each train/val/test split
    
    # Iterate through all train/val/test splits
    while 1:
        start_ix=0; end_ix=0; startCv_ix=0; endCv_ix=0; startTest_ix=0; endTest_ix=0;
        X_it, y_it, Xcv_it, ycv_it, Xtest_it, ytest_it = list(), list(), list(), list(), list(), list()
        i=0; # Index of individual training set at each train/val/test split
        
        # Iterate until index of individual training set is smaller than index of cv set
        while (i < j):
            ## TRAINING DATA
            start_ix = n_steps_jump*i;
            end_ix = start_ix + n_steps_input;
            
            seq_x = sequence[start_ix:end_ix] 
            X_it.append(seq_x)
            seq_y = sequence[end_ix:end_ix+n_steps_forecast]
            y_it.append(seq_y)
            
            i+=1;
          
        # Once test data crosses time series length return   
        if ((((end_ix+n_steps_input)+n_steps_input)+n_steps_forecast) > (len(sequence))):
            break
        
        ## CROSS-VALIDATION DATA
        startCv_ix = end_ix;
        endCv_ix = end_ix + n_steps_input;
        
        seq_xcv = sequence[startCv_ix:endCv_ix] 
        Xcv_it.append(seq_xcv)
        seq_ycv = sequence[endCv_ix:endCv_ix+n_steps_forecast]
        ycv_it.append(seq_ycv) 
        
        ## TEST DATA
        startTest_ix = endCv_ix;
        endTest_ix = endCv_ix + n_steps_input;
        
        seq_xtest = sequence[startTest_ix:endTest_ix] 
        Xtest_it.append(seq_xtest)
        seq_ytest = sequence[endTest_ix:endTest_ix+n_steps_forecast]
        ytest_it.append(seq_ytest) 
            
        ## Add another train/val/test split 
        X[j-2] = np.array(X_it)
        y[j-2] = np.array(y_it)
        Xcv[j-2] = np.array(Xcv_it)
        ycv[j-2] = np.array(ycv_it)
        Xtest[j-2] = np.array(Xtest_it)
        ytest[j-2] = np.array(ytest_it)
        
        j+=1;
            
    return X, y, Xcv, ycv, Xtest, ytest

def split_train_cv_test_kFold(sequence, n_steps_input, n_steps_forecast, n_steps_jump):
    """ Returns sets to train, cross-validate and test a model using K-Fold technique
    
    Parameters:
        sequence (array): Full training dataset
        n_steps_input (int): Number of inputs X, Xcv and Xtest used at each training
        n_steps_forecast (int): Number of outputs y, ycv, ytest used at each training
        n_steps_jump (int): Number of sequence samples to be ignored between (X,y) sets

    Returns:
        X (2D array): Array of n_steps_input arrays used for training
                      len(X[k]) = n_steps_input
        y (2D array): Array of n_steps_forecast arrays used for training
                      len(y[k]) = n_steps_forecast
        Xcv (2D array): Array of n_steps_input arrays used for cross-validation
                        len(Xcv[k]) = n_steps_input
        ycv (2D array): Array of n_steps_forecast arrays used for cross-validation
                        len(ycv[k]) = n_steps_forecast
        Xtest (2D array): Array of n_steps_input arrays used for testing
                          len(Xtest[k]) = n_steps_input
        ytest (2D array): Array of n_steps_input arrays used for testing
                          len(Xtest[k]) = n_steps_input
    """
    X, y, Xcv, ycv, Xtest, ytest = dict(), dict(), dict(), dict(), dict(), dict()
    j=2;  # Tracks index of CV set at each train/val/test split
    theEnd = 0; # Flag to terminate function
    
    # Iterate until test set falls outside time series length
    while 1:
        start_ix=0; end_ix=0; startCv_ix=0; endCv_ix=0; startTest_ix=0; endTest_ix=0;
        X_it, y_it, Xcv_it, ycv_it, Xtest_it, ytest_it = list(), list(), list(), list(), list(), list()
        i=0; # Index of individual training set at each train/val/test split
        n=0; # Number of n_steps_jump
        
        # Iterate through all train/val/test splits
        while 1:
            if (i != j): 
                ## TRAINING DATA
                start_ix = endTest_ix + n_steps_jump*n;
                end_ix = start_ix + n_steps_input;
                n += 1;

                # Leave train/val/test split loop once training data crosses time series length
                if end_ix+n_steps_forecast > len(sequence):
                    break;

                seq_x = sequence[start_ix:end_ix] 
                X_it.append(seq_x)
                seq_y = sequence[end_ix:end_ix+n_steps_forecast]
                y_it.append(seq_y)
            else:
                
                # Once test data crosses time series length return   
                if ((((end_ix+n_steps_input)+n_steps_input)+n_steps_forecast) > (len(sequence))):
                    theEnd = 1;
                    break
                    
                n=0;
                i+=1;
                
                ## CROSS-VALIDATION DATA
                startCv_ix = end_ix;
                endCv_ix = end_ix + n_steps_input;
                
                seq_xcv = sequence[startCv_ix:endCv_ix] 
                Xcv_it.append(seq_xcv)
                seq_ycv = sequence[endCv_ix:endCv_ix+n_steps_forecast]
                ycv_it.append(seq_ycv)
                
                ## TEST DATA
                startTest_ix = endCv_ix;
                endTest_ix = endCv_ix + n_steps_input;

                seq_xtest = sequence[startTest_ix:endTest_ix] 
                Xtest_it.append(seq_xtest)
                seq_ytest = sequence[endTest_ix:endTest_ix+n_steps_forecast]
                ytest_it.append(seq_ytest) 
                
            i+=1;
        
        # Only add a train/val/test split if the time series length has not been crossed
        if (theEnd == 1):
            break
        
        ## Add another train/val/test split 
        X[j-2] = np.array(X_it)
        y[j-2] = np.array(y_it)
        Xcv[j-2] = np.array(Xcv_it)
        ycv[j-2] = np.array(ycv_it)
        Xtest[j-2] = np.array(Xtest_it)
        ytest[j-2] = np.array(ytest_it)
        
        j+=1;
            
    return X, y, Xcv, ycv, Xtest, ytest

def split_train_cv_test_groupKFold(sequence, n_steps_input, n_steps_forecast, n_steps_jump):
    """ Returns sets to train, cross-validate and test a model using group K-Fold technique
    
    Parameters:
        sequence (array): Full training dataset
        n_steps_input (int): Number of inputs X, Xcv and Xtest used at each training
        n_steps_forecast (int): Number of outputs y, ycv, ytest used at each training
        n_steps_jump (int): Number of sequence samples to be ignored between (X,y) sets

    Returns:
        X (2D array): Array of n_steps_input arrays used for training
                      len(X[k]) = n_steps_input
        y (2D array): Array of n_steps_forecast arrays used for training
                      len(y[k]) = n_steps_forecast
        Xcv (2D array): Array of n_steps_input arrays used for cross-validation
                        len(Xcv[k]) = n_steps_input
        ycv (2D array): Array of n_steps_forecast arrays used for cross-validation
                        len(ycv[k]) = n_steps_forecast
        Xtest (2D array): Array of n_steps_input arrays used for testing
                          len(Xtest[k]) = n_steps_input
        ytest (2D array): Array of n_steps_input arrays used for testing
                          len(Xtest[k]) = n_steps_input
    """
    X, y, Xcv, ycv, Xtest, ytest = dict(), dict(), dict(), dict(), dict(), dict()
    
    # Iterate through 5 train/val/test splits
    for j in np.arange(5):
        start_ix=0; end_ix=0; startCv_ix=0; endCv_ix=0; startTest_ix=0; endTest_ix=0;
        X_it, y_it, Xcv_it, ycv_it, Xtest_it, ytest_it = list(), list(), list(), list(), list(), list()
        i=0; # Index of individual training set at each train/val/test split
        n=0; # Number of n_steps_jump
        
        while 1: 
            if ((i+1+j)%(5) != 0):
                # TRAINING DATA
                start_ix = endTest_ix + n_steps_jump*n;
                end_ix = start_ix + n_steps_input;
                n+=1;

                # Leave train/val/test split loop if train data crosses time series length
                if end_ix+n_steps_forecast > len(sequence):
                    break 

                seq_x = sequence[start_ix:end_ix] 
                X_it.append(seq_x)
                seq_y = sequence[end_ix:end_ix+n_steps_forecast]
                y_it.append(seq_y)
            else:
                # CROSS-VALIDATION DATA
                startCv_ix = end_ix;
                endCv_ix = end_ix + n_steps_input;
                
                # Leave train/val/test split loop if val data crosses time series length  
                if ((endCv_ix+n_steps_forecast) > len(sequence)):
                    break

                seq_xcv = sequence[startCv_ix:endCv_ix] 
                Xcv_it.append(seq_xcv)
                seq_ycv = sequence[endCv_ix:endCv_ix+n_steps_forecast]
                ycv_it.append(seq_ycv)
                
                # TEST DATA
                startTest_ix = endCv_ix;
                endTest_ix = endCv_ix + n_steps_input;

                # Leave train/val/test split loop if test data crosses time series length  
                if ((endTest_ix+n_steps_forecast) > len(sequence)):
                    break

                seq_xtest = sequence[startTest_ix:endTest_ix] 
                Xtest_it.append(seq_xtest)
                seq_ytest = sequence[endTest_ix:endTest_ix+n_steps_forecast]
                ytest_it.append(seq_ytest) 
                
                n=0;
                i+=1;
                
            i+=1;
            
        ## Add another train/val split     
        X[j] = np.array(X_it)
        y[j] = np.array(y_it)
        Xcv[j] = np.array(Xcv_it)
        ycv[j] = np.array(ycv_it)
        Xtest[j] = np.array(Xtest_it)
        ytest[j] = np.array(ytest_it)
            
    return X, y, Xcv, ycv, Xtest, ytest
