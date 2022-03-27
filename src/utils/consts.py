# origonal sample rates
fs_PPG = 256 
fs_sleep = 1/30

#desired sample rates to downsample to
new_fs_PPG = 16

dT = 30
windows = 5
hops = 1


SLEEP_STAGES = 4
sleep_encoding = {}
sleep_encoding[3]= {0: 0,
                    1: 1, 2: 1, 3: 1,
                    4: 2}

sleep_encoding[4]= {0: 0,
                    1: 1, 
                    2: 1, 
                    3: 2,
                    4: 3}

sleep_encoding[5]= {0: 0,
                    1: 1,
                    2: 2,
                    3: 3,
                    4: 4}

sleep_encoding = sleep_encoding[SLEEP_STAGES]


#Optimization Parameters
from skopt.space import Real, Categorical, Integer

dim_single_fold = Categorical(categories=['true'],name='dim_single_fold')
dim_epochs = Integer(low=5, high=15, name='dim_epochs')
dim_dropout = Real(low=1e-4, high=5e-1, name='dim_dropout')
dim_learning_rate = Real(low=1e-4, high=1e-1,name='dim_learning_rate')
dim_batch_size = Integer(low=7, high=9, name='dim_batch_size')                              # power of 2
dim_kernel_size = Integer(low = 3, high=13, name='dim_kernel_size')
dim_channels = Integer(low=4, high=7,name='dim_channels')

dimensions = [dim_single_fold,
              dim_epochs, 
              dim_dropout,
              dim_learning_rate,
              dim_batch_size,
              dim_kernel_size, 
              dim_channels,
             ]
dimensions_names = ['single_fold',
              'epochs', 
              'dropout',
              'learning_rate',
              'batch_size(power)',
              'kernel_size', 
              'channels(power)',
               ]
default_parameters = ['true', 5, 0.25, 1e-3, 7, 8, 5]



