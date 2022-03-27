import os
from pathlib import Path
import getpass

# Windows directories
if os.name == 'nt':
    data_dir = Path('Z:/mesa/polysomnography/')
    saved_dataset_dir = Path('C:/Users/kkotzen/temp/datasets')
    model_checkpoints = Path('C:/Users/kkotzen/temp/model_checkpoints')

# Ubuntu Directories
else:
    # data_dir = Path('/MLdata/AIMLab/databases/mesa/polysomnography/')

    if getpass.getuser() == 'kevin':
        data_dir = Path('/home/kevin/databases/mesa/polysomnography/')
        saved_dataset_dir = Path('/home/kevin/RespirationProject/datasets')
        model_checkpoints = Path('/home/kevin/RespirationProject/model_checkpoints')
        figures_dir = Path('/home/kevin/RespirationProject/figures')
        saved_learning_data = Path('/MLdata/AIMLab/Yuval/097248/learning_data')
        sleep_saved_learning_data = Path('/home/kevin/RespirationProject/datasets')
        
    elif getpass.getuser() == 'yuvalb':
        saved_dataset_dir = Path('/MLdata/AIMLab/Yuval/097248/datasets')
        model_checkpoints = Path('/MLdata/AIMLab/Yuval/097248/model_checkpoints')
        figures_dir = Path('/MLdata/AIMLab/Yuval/097248/plots')
        saved_learning_data = Path('/MLdata/AIMLab/Yuval/097248/learning_data')

    elif getpass.getuser() == 'daniel':
        data_dir = Path('/home/daniel/databases/mesa/polysomnography/')
        saved_dataset_dir = Path('/home/daniel/Documents/project/SleepStagingProject/datasets')
        model_checkpoints = Path('/home/daniel/Documents/project/SleepStagingProject/model_checkpoints')
        figures_dir = Path('/home/daniel/Documents/project/SleepStagingProject/figures')
