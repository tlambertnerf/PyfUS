import pyfus.utils as u
import pyfus.correlation_analysis as correl
import pyfus.region_averaging_analysis as region_avg
import pyfus.clustering as cl
import pyfus.quantification as q
import matplotlib.pyplot as plt
import numpy as np


################################################################################
#                           DATASET SELECTION
################################################################################
### Here put the path that resulted from the data loading example
src = 'E:/20-pyfus/loaded_data/example_dataset_gratings_median_subject_reg_07b6f81d'

### The list of stims, expgroup, subjects to include in the analysis. Set to None if you want everything to be included.
stims = ['backward']
expgroup_ID = None
subject_ID = None
session_ID = None

### Sampling rate of the data, in Hz. Used for the quantifications.
sampling_rate = 1.6

### Whether the data you will analyse has been registered or not
registered = True

### List of regions to exclude from the analysis. Default is [] (no regions excluded)
regions_to_exclude = []

### No modification is expected from the user (lines 28-32)
atlas_resolution = 100 # The resolution of the atlas. Default resolution is 100 and extension is '_nolayersnoparts'
ext = '_nolayersnoparts' # The extension to select the atlas. Default resolution is 100 and extension is '_nolayersnoparts'
atlas_path, _, regions_info_path = u.get_atlas_and_info_paths(atlas_resolution, ext)
data_paths = u.post_data_loading_iterator(src, expgroup_ID=expgroup_ID, subject_ID=subject_ID, session_ID=session_ID, stim_ID=stims)

################################################################################
#                              REGION AVERAGING
################################################################################

### Remove/add '#' if you want to skip/open this section
#"""
assert registered == True, "You cannot use region averaging analysis if you did not register your data during data loading."
raa = region_avg.RegionAveragingAnalysis(data_paths, atlas_path, regions_info_path, regions_to_exclude=regions_to_exclude)

### Method for plotting the barcode
raa.plot_barcode(names=None, separate_plots=True, scale=[-0.1, 0.1])
### Method for plotting the temporal traces of individual regions
raa.plot_region([('SCs', 'LR')], scale=[-0.04, 0.12])
### To display all the generated plots
plt.show()

### Method for extracting traces from individual regions for further processing
res = raa.get_regions_traces([('LGd', 'L'), ('SCs', 'L')])

### Example of temporal quantification
tq_ravg = q.TemporalQuantification(res, sampling_rate)
tq_ravg.plot_metric('Peak amplitude')
#"""

################################################################################
#                                  CORRELATION
################################################################################

### Remove/add '#' if you want to skip/open this section
#"""

### Definition of the parameters for the correl. Check the doc of the CorrelationAnalysis object in the correlation module!
correlation_pattern = [(20,60)]
n_samples = 20
significance_threshold = 0.01

ca = correl.CorrelationAnalysis(data_paths, correlation_pattern, n_samples=n_samples, significance_threshold=significance_threshold)
res = ca.process(registered=registered, atlas_path=atlas_path, regions_info_path=regions_info_path)

### Example of spatial quantification. Only works if your data has been registered!
sq_corr = q.SpatialQuantification(res, atlas_path, regions_info_path)
sq_corr.print_quantification_region('SCs', 'L')

#"""

################################################################################
#                                 CLUSTERING
################################################################################

### Remove/add '#' if you want to skip/open this section
#"""

### Check the doc of the SingleVoxelClusteringWrapper object from the clustering module for details on the parameters
method = 'hemisphere'
n_clusters = 4
fe_method = 'pca'
fe_params = {'n_components': 10}
noise_th = 1.5

svc = cl.SingleVoxelClusteringWrapper(method, n_clusters, data_paths, atlas_path, regions_info_path, fe_method=fe_method, fe_params=fe_params, noise_th=noise_th, registered=registered)

### To uncomment if method = 'brainwide'
#svc.process()
### To uncomment if method = 'hemisphere'
svc.process('LR')
### To uncomment if method = 'structure' and you want to use an anatomical group
#svc.process([('Isocortex', 'LR')])
### To uncomment if method = 'multi_region' and you want to use a list of regions
#svc.process([('SCs', 'L'), ('SCm', 'L'), ('SCig', 'L'), ('VISp', 'L'), ('VISa', 'L'), ('VISal', 'L'),('VISam', 'L'),('VISl', 'L'),('VISli', 'L'),('VISpl', 'L'),('VISpm', 'L'),('VISrl', 'L'),('VISpor', 'L'),('LGv', 'L'),('LGd', 'L'), ('VPM', 'L'), ('MD', 'L'), ('VAL', 'L'), ('VM', 'L'), ('VPL', 'L'), ('SSp-bfd', 'L')])

### Example definition of a custom colormap (RGBA)
custom_cmap = [[0.4, 0.8, 0.2, 1.], [1., 0.6, 0., 1.], [1., 0., 0.8, 1.], [0., 0.6, 1., 1.], [0.1, 0.1, 0.1, 1.], [0.5, 0.5, 0.5, 1.]]
svc.change_cmap(custom_cmap)

### Methods for displaying the results of the clustering process
svc.plot_signals(display='mean_std', scale=[-0.1,0.2])
svc.display_cluster_locations()
plt.show()

### Method for getting cluster maps for further processing
res = svc.get_cluster_maps()

### Example of spatial quantification. Only works if your data has been registered!
sq_cl = q.SpatialQuantification(res, atlas_path, regions_info_path)
sq_cl.print_quantification_region('VISli', 'L')

### Method for getting cluster signals for further processing
res = svc.get_signals()

### Example of spatial quantification
tq_cl = q.TemporalQuantification(res, sampling_rate)
tq_cl.plot_metric('Peak amplitude', colors=custom_cmap)

#"""
