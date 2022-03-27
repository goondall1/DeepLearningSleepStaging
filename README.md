# DeepLearningSleepStaging
structure:
SleepStageFromPPG.ipynb [1]
src ->
	models ->
		sleepstagingcnn.py [2]
utils ->
	consts.py [3]
	utils.py [4]
	paths.py [5]
	sleep_extract.py [6]
	
[1] the main code engine importing all neccessery packages,
	loading all the data and preprocess it,
	performing train and evaluation tasks.
	
[2] contains all the different model architecures beeing experimented.

[3] as the name suggests defining some env veaiables and hyper-parameters.

[4] all the functions we are using to load and prepeocess the data.

[5] sets specific paths to data locations.

[6] parsing xmls to sleep stages.

The data can be obtained from https://sleepdata.org/datasets/mesa. 
Once obtained, the ECG and PPG signals needs to be extracted from 
the EDF files and stored in a folder. Signal quality is then calculated using signal_quality.py (not included here). 

training and evalution is then run from the main jupyter notebook 
after the correct paths for the data is been set.


