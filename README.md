me Series Cross Validation Module
(working on it)

pip install git+https://github.com/DidierRLopes/TimeSeriesCrossValidation --upgrade

-------------------------------------------------------------------------------
Below you can find an example of the usage of each function for the following Time-Series:

timeSeries = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)

-------------------------------------------------------------------------------
from TimeSeriesCrossValidation.splitTrainVal import split_train_val_forwardChaining

X, y, Xcv, ycv = split_train_val_forwardChaining(timeSeries, n_steps_input=4, n_steps_forecast=3, n_steps_jump=2)

<img width="744" alt="trainValTest - forwardChaining" src="https://user-images.githubusercontent.com/25267873/74094566-6fab1e00-4adb-11ea-810d-e085518c3cb5.png">

-------------------------------------------------------------------------------
from TimeSeriesCrossValidation.splitTrainVal import split_train_val_kFold

X, y, Xcv, ycv = split_train_val_kFold(timeSeries, n_steps_input=4, n_steps_forecast=3, n_steps_jump=2)

<img width="744" alt="trainValTest - groupKFold" src="https://user-images.githubusercontent.com/25267873/74094567-70dc4b00-4adb-11ea-994b-c3f1727f4b83.png">

-------------------------------------------------------------------------------
from TimeSeriesCrossValidation.splitTrainVal import split_train_val_groupKFold

X, y, Xcv, ycv = split_train_val_groupKFold(timeSeries, n_steps_input=4, n_steps_forecast=3, n_steps_jump=2)

<img width="742" alt="trainVal - forwardChaining" src="https://user-images.githubusercontent.com/25267873/74094568-720d7800-4adb-11ea-8d69-7c1cbd6774c7.png">

-------------------------------------------------------------------------------
from TimeSeriesCrossValidation.splitTrainValTest import split_train_val_test_forwardChaining

X, y, Xcv, ycv, Xtest, ytest = split_train_val_test_forwardChaining(timeSeries, n_steps_input=4, n_steps_forecast=3, n_steps_jump=2)

<img width="744" alt="trainVal - groupKFold" src="https://user-images.githubusercontent.com/25267873/74094569-72a60e80-4adb-11ea-8345-1233b0a47e2e.png">

-------------------------------------------------------------------------------
from TimeSeriesCrossValidation.splitTrainValTest import split_train_val_test_kFold

X, y, Xcv, ycv, Xtest, ytest = split_train_val_test_kFold(timeSeries, n_steps_input=4, n_steps_forecast=3, n_steps_jump=2)

<img width="745" alt="trainValTest - kFold" src="https://user-images.githubusercontent.com/25267873/74094570-73d73b80-4adb-11ea-94cd-5ab4d02c8cbf.png">

-------------------------------------------------------------------------------
from TimeSeriesCrossValidation.splitTrainValTest import split_train_val_test_groupKFold

X, y, Xcv, ycv, Xtest, ytest = split_train_val_test_groupKFold(timeSeries, n_steps_input=4, n_steps_forecast=3, n_steps_jump=2)

<img width="743" alt="trainVal - kFold" src="https://user-images.githubusercontent.com/25267873/74094572-746fd200-4adb-11ea-91fd-93935d51982f.png">
