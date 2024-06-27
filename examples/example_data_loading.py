import foss4fus.data_loading as dl
import foss4fus.quality_control as qc
from time import time
from datetime import timedelta


### HYPERPARAMETERS
### Check the doc of the DataLoaderAndPreproc object in the data_loading module to see what each parameter is doing!

root_folder = 'E:/20-foss4fus'
experiment_ID = 'example_dataset_gratings'

mode = 'folder_tree'
excel_source = None

expgroup_ID = None
subject_ID = ['mouseA', 'mouseB', 'mouseC']
session_ID = None
stim_ID = ['backward', 'forward']

reduction = 'median'
level_avg = 'subject'

register = True
baseline = range(4,20)

make_reliability_maps = True
remove_unreliable = 0.6


### TRIALS PREPROCESSING OBJECT
### Check the quality_control module doc for more information!

frame_removal = qc.OutlierFrameRemoval()


### DATA LOADING

tmstp_full = time()

loader = dl.DataLoaderAndPreproc(
root_folder,
experiment_ID,
expgroup_ID=expgroup_ID,
subject_ID=subject_ID,
session_ID=session_ID,
stim_ID=stim_ID,
mode=mode,
excel_source=excel_source,
reduction=reduction,
level_avg=level_avg,
register=register,
baseline=baseline,
make_reliability_maps=make_reliability_maps,
remove_unreliable=remove_unreliable,
trial_preprocessing=frame_removal
)

loader.process_data()
elapsed_time = time()-tmstp_full

print(F"Total elapsed time {timedelta(seconds=elapsed_time)}")
