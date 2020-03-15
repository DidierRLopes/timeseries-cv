"""
Forward Chaining, K-Fold and Group K-Fold algorithms to split a given training dataset into train (X, y), validation (Xcv, ycv) and test (Xtest, ytest) sets
"""

import numpy as np

def split_train_val_test_forwardChaining(sequence, numInputs, numOutputs, numJumps):
    """ Returns sets to train, cross-validate and test a model using forward chaining technique
    
    Parameters:
        sequence (array)  : Full training dataset
        numInputs (int)   : Number of inputs X and Xcv used at each training
        numOutputs (int)  : Number of outputs y and ycv used at each training
        numJumps (int)    : Number of sequence samples to be ignored between (X,y) sets

    Returns:
        X (2D array)      : Array of numInputs arrays used for training
        y (2D array)      : Array of numOutputs arrays used for training
        Xcv (2D array)    : Array of numInputs arrays used for cross-validation
        ycv (2D array)    : Array of numOutputs arrays used for cross-validation
        Xtest (2D array)  : Array of numInputs arrays used for testing
        ytest (2D array)  : Array of numOutputs arrays used for testing

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
            start_ix = numJumps*i;
            end_ix = start_ix + numInputs;
            
            seq_x = sequence[start_ix:end_ix] 
            X_it.append(seq_x)
            seq_y = sequence[end_ix:end_ix+numOutputs]
            y_it.append(seq_y)
            
            i+=1;
          
        # Once test data crosses time series length return   
        if ((((end_ix+numInputs)+numInputs)+numOutputs) > (len(sequence))):
            break
        
        ## CROSS-VALIDATION DATA
        startCv_ix = end_ix;
        endCv_ix = end_ix + numInputs;
        
        seq_xcv = sequence[startCv_ix:endCv_ix] 
        Xcv_it.append(seq_xcv)
        seq_ycv = sequence[endCv_ix:endCv_ix+numOutputs]
        ycv_it.append(seq_ycv) 
        
        ## TEST DATA
        startTest_ix = endCv_ix;
        endTest_ix = endCv_ix + numInputs;
        
        seq_xtest = sequence[startTest_ix:endTest_ix] 
        Xtest_it.append(seq_xtest)
        seq_ytest = sequence[endTest_ix:endTest_ix+numOutputs]
        ytest_it.append(seq_ytest) 
            
        ## Add another train/val/test split 
        X[j-2] = np.array(X_it)
        y[j-2] = np.array(y_it)
        Xcv[j-2] = np.array(Xcv_it)
        ycv[j-2] = np.array(ycv_it)
        Xtest[j-2] = np.array(Xtest_it)
        ytest[j-2] = np.array(ytest_it)
        
        j+=1;
        
    if (len(X)==0 or len(Xcv)==0 or len(Xtest)==0):
        print("The sequence provided does not has size enough to populate the return arrays")
            
    return X, y, Xcv, ycv, Xtest, ytest


def split_train_val_test_kFold(sequence, numInputs, numOutputs, numJumps):
    """ Returns sets to train, cross-validate and test a model using K-Fold technique
    
    Parameters:
        sequence (array)  : Full training dataset
        numInputs (int)   : Number of inputs X and Xcv used at each training
        numOutputs (int)  : Number of outputs y and ycv used at each training
        numJumps (int)    : Number of sequence samples to be ignored between (X,y) sets

    Returns:
        X (2D array)      : Array of numInputs arrays used for training
        y (2D array)      : Array of numOutputs arrays used for training
        Xcv (2D array)    : Array of numInputs arrays used for cross-validation
        ycv (2D array)    : Array of numOutputs arrays used for cross-validation
        Xtest (2D array)  : Array of numInputs arrays used for testing
        ytest (2D array)  : Array of numOutputs arrays used for testing
        
    """
    
    X, y, Xcv, ycv, Xtest, ytest = dict(), dict(), dict(), dict(), dict(), dict()
    j=2;  # Tracks index of CV set at each train/val/test split
    theEnd = 0; # Flag to terminate function
    
    # Iterate until test set falls outside time series length
    while 1:
        start_ix=0; end_ix=0; startCv_ix=0; endCv_ix=0; startTest_ix=0; endTest_ix=0;
        X_it, y_it, Xcv_it, ycv_it, Xtest_it, ytest_it = list(), list(), list(), list(), list(), list()
        i=0; # Index of individual training set at each train/val/test split
        n=0; # Number of numJumps
        
        # Iterate through all train/val/test splits
        while 1:
            if (i != j): 
                ## TRAINING DATA
                start_ix = endTest_ix + numJumps*n;
                end_ix = start_ix + numInputs;
                n += 1;

                # Leave train/val/test split loop once training data crosses time series length
                if end_ix+numOutputs > len(sequence):
                    break;

                seq_x = sequence[start_ix:end_ix] 
                X_it.append(seq_x)
                seq_y = sequence[end_ix:end_ix+numOutputs]
                y_it.append(seq_y)
            else:
                
                # Once test data crosses time series length return   
                if ((((end_ix+numInputs)+numInputs)+numOutputs) > (len(sequence))):
                    theEnd = 1;
                    break
                    
                n=0;
                i+=1;
                
                ## CROSS-VALIDATION DATA
                startCv_ix = end_ix;
                endCv_ix = end_ix + numInputs;
                
                seq_xcv = sequence[startCv_ix:endCv_ix] 
                Xcv_it.append(seq_xcv)
                seq_ycv = sequence[endCv_ix:endCv_ix+numOutputs]
                ycv_it.append(seq_ycv)
                
                ## TEST DATA
                startTest_ix = endCv_ix;
                endTest_ix = endCv_ix + numInputs;

                seq_xtest = sequence[startTest_ix:endTest_ix] 
                Xtest_it.append(seq_xtest)
                seq_ytest = sequence[endTest_ix:endTest_ix+numOutputs]
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
        
    if (len(X)==0 or len(Xcv)==0 or len(Xtest)==0):
        print("The sequence provided does not has size enough to populate the return arrays")
        
    return X, y, Xcv, ycv, Xtest, ytest


def split_train_val_test_groupKFold(sequence, numInputs, numOutputs, numJumps):
    """ Returns sets to train, cross-validate and test a model using group K-Fold technique
    
    Parameters:
        sequence (array)  : Full training dataset
        numInputs (int)   : Number of inputs X and Xcv used at each training
        numOutputs (int)  : Number of outputs y and ycv used at each training
        numJumps (int)    : Number of sequence samples to be ignored between (X,y) sets

    Returns:
        X (2D array)      : Array of numInputs arrays used for training
        y (2D array)      : Array of numOutputs arrays used for training
        Xcv (2D array)    : Array of numInputs arrays used for cross-validation
        ycv (2D array)    : Array of numOutputs arrays used for cross-validation
        Xtest (2D array)  : Array of numInputs arrays used for testing
        ytest (2D array)  : Array of numOutputs arrays used for testing
        
    """
    
    X, y, Xcv, ycv, Xtest, ytest = dict(), dict(), dict(), dict(), dict(), dict()
    
    # Iterate through 5 train/val/test splits
    for j in np.arange(5):
        start_ix=0; end_ix=0; startCv_ix=0; endCv_ix=0; startTest_ix=0; endTest_ix=0;
        X_it, y_it, Xcv_it, ycv_it, Xtest_it, ytest_it = list(), list(), list(), list(), list(), list()
        i=0; # Index of individual training set at each train/val/test split
        n=0; # Number of numJumps
        
        while 1: 
            if ((i+1+j)%(5) != 0):
                # TRAINING DATA
                start_ix = endTest_ix + numJumps*n;
                end_ix = start_ix + numInputs;
                n+=1;

                # Leave train/val/test split loop if train data crosses time series length
                if end_ix+numOutputs > len(sequence):
                    break 

                seq_x = sequence[start_ix:end_ix] 
                X_it.append(seq_x)
                seq_y = sequence[end_ix:end_ix+numOutputs]
                y_it.append(seq_y)
            else:
                # CROSS-VALIDATION DATA
                startCv_ix = end_ix;
                endCv_ix = end_ix + numInputs;
                
                # Leave train/val/test split loop if val data crosses time series length  
                if ((endCv_ix+numOutputs) > len(sequence)):
                    break

                seq_xcv = sequence[startCv_ix:endCv_ix] 
                Xcv_it.append(seq_xcv)
                seq_ycv = sequence[endCv_ix:endCv_ix+numOutputs]
                ycv_it.append(seq_ycv)
                
                # TEST DATA
                startTest_ix = endCv_ix;
                endTest_ix = endCv_ix + numInputs;

                # Leave train/val/test split loop if test data crosses time series length  
                if ((endTest_ix+numOutputs) > len(sequence)):
                    break

                seq_xtest = sequence[startTest_ix:endTest_ix] 
                Xtest_it.append(seq_xtest)
                seq_ytest = sequence[endTest_ix:endTest_ix+numOutputs]
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
        
    if (len(X)==0 or len(Xcv)==0 or len(Xtest)==0):
        print("The sequence provided does not has size enough to populate the return arrays")
            
    return X, y, Xcv, ycv, Xtest, ytest
