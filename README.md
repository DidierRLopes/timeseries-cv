me Series Cross Validation Module
(working on it)

pip install git+https://github.com/DidierRLopes/TimeSeriesCrossValidation --upgrade

Below you can find an example of the usage of each function for the following Time-Series:
timeSeries = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)

------------------------------------------------------------------------------------
from TimeSeriesCrossValidation.splitTrainVal import split_train_val_forwardChaining
X, y, Xcv, ycv = split_train_cv_forwardChaining(timeSeries, n_steps_input=4, n_steps_forecast=3, n_steps_jump=2)
--------- SET 1 ---------
X[1] = [0 1 2 3] [2 3 4 5]
y[1] = [4 5 6] [6 7 8]
Xcv[1] = [6 7 8 9]
ycv[1] = [10 11 12]
--------- SET 2 ---------
X[2] = [0 1 2 3] [2 3 4 5] [4 5 6 7]
y[2] = [4 5 6] [6 7 8] [ 8  9 10]
Xcv[2] = [ 8  9 10 11]
ycv[2] = [12 13 14]
--------- SET 3 ---------
X[3] = [0 1 2 3] [2 3 4 5] [4 5 6 7] [6 7 8 9]
y[3] = [4 5 6] [6 7 8] [ 8  9 10] [10 11 12]
Xcv[3] = [10 11 12 13]
ycv[3] = [14 15 16]
--------- SET 4 ---------
X[4] = [0 1 2 3] [2 3 4 5] [4 5 6 7] [6 7 8 9] [ 8  9 10 11]
y[4] = [4 5 6] [6 7 8] [ 8  9 10] [10 11 12] [12 13 14]
Xcv[4] = [12 13 14 15]
ycv[4] = [16 17 18]
--------- SET 5 ---------
X[5] = [0 1 2 3] [2 3 4 5] [4 5 6 7] [6 7 8 9] [ 8  9 10 11] [10 11 12 13]
y[5] = [4 5 6] [6 7 8] [ 8  9 10] [10 11 12] [12 13 14] [14 15 16]
Xcv[5] = [14 15 16 17]
ycv[5] = [18 19 20]
--------- SET 6 ---------
X[6] = [0 1 2 3] [2 3 4 5] [4 5 6 7] [6 7 8 9] [ 8  9 10 11] [10 11 12 13] [12 13 14 15]
y[6] = [4 5 6] [6 7 8] [ 8  9 10] [10 11 12] [12 13 14] [14 15 16] [16 17 18]
Xcv[6] = [16 17 18 19]
ycv[6] = [20 21 22]
--------- SET 7 ---------
X[7] = [0 1 2 3] [2 3 4 5] [4 5 6 7] [6 7 8 9] [ 8  9 10 11] [10 11 12 13] [12 13 14 15] [14 15 16 17]
y[7] = [4 5 6] [6 7 8] [ 8  9 10] [10 11 12] [12 13 14] [14 15 16] [16 17 18] [18 19 20]
Xcv[7] = [18 19 20 21]
ycv[7] = [22 23 24]
--------- SET 8 ---------
X[8] = [0 1 2 3] [2 3 4 5] [4 5 6 7] [6 7 8 9] [ 8  9 10 11] [10 11 12 13] [12 13 14 15] [14 15 16 17] [16 17 18 19]
y[8] = [4 5 6] [6 7 8] [ 8  9 10] [10 11 12] [12 13 14] [14 15 16] [16 17 18] [18 19 20] [20 21 22]
Xcv[8] = [20 21 22 23]
ycv[8] = [24 25 26]

<img width="744" alt="trainValTest - forwardChaining" src="https://user-images.githubusercontent.com/25267873/74094566-6fab1e00-4adb-11ea-810d-e085518c3cb5.png">

------------------------------------------------------------------------------------

<img width="744" alt="trainValTest - groupKFold" src="https://user-images.githubusercontent.com/25267873/74094567-70dc4b00-4adb-11ea-994b-c3f1727f4b83.png">

<img width="742" alt="trainVal - forwardChaining" src="https://user-images.githubusercontent.com/25267873/74094568-720d7800-4adb-11ea-8d69-7c1cbd6774c7.png">

<img width="744" alt="trainVal - groupKFold" src="https://user-images.githubusercontent.com/25267873/74094569-72a60e80-4adb-11ea-8345-1233b0a47e2e.png">

<img width="745" alt="trainValTest - kFold" src="https://user-images.githubusercontent.com/25267873/74094570-73d73b80-4adb-11ea-94cd-5ab4d02c8cbf.png">

<img width="743" alt="trainVal - kFold" src="https://user-images.githubusercontent.com/25267873/74094572-746fd200-4adb-11ea-91fd-93935d51982f.png">
