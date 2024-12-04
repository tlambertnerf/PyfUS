import numpy as np
import os, typing, pickle
import matplotlib.pyplot as plt
import pyfus.utils as u
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD
from matplotlib import cm
from time import time
from pyfus.features_extraction import FeatureExtractor
from typing import Union, List, Tuple
from copy import deepcopy
from scipy.ndimage import median_filter



def volume_loading(filelist: List[str]) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    """
    Utility function for loading volumes along with the necessary info: coordinates, file indices and shape of the volume.
    Note that 'nan' values are excluded during the process.

    Parameters
    ----------
    filelist : list of str
        List of paths to the files containing the volumes to be loaded.

    Returns
    -------
    dataset : ndarray
        Contains the volumes data (time traces).
    coords : ndarray
        Contains the coordinates in the volume of each voxel.
    file_indices : array
        Array filled with int values corresponding to the volume the data was extracted from. For example, 0 will correspond to data from the first loaded file.
    data_volume_shape : array
        Shape of the volumes (excluding the time dimension) for reconstructing the volume later
    """

    dataset, coords, file_indices = [], [], []

    for i, file in enumerate(filelist):

        data = np.load(file)

        data_volume_shape = data.shape[:-1]
        data_shape = (data.shape[0]*data.shape[1]*data.shape[2], data.shape[-1]) #one single voxel per line

        data = data.reshape(data_shape)

        coords_grid = np.argwhere(np.ones(data_volume_shape) != 0)

        idx_nan = ~np.isnan(data).any(axis=1)
        dataset.append(data[idx_nan, :])
        coords.append(coords_grid[idx_nan, :])

        N = np.sum(idx_nan)
        file_indices += [i]*N

    data_len = [d.shape[-1] for d in dataset]
    assert data_len.count(data_len[0]) == len(data_len), "All data must have the same number of frames. Check the different stimuli and/or experimental groups. Exiting..."
    dataset, coords = np.vstack(dataset), np.vstack(coords)

    return(dataset, coords, np.array(file_indices), data_volume_shape)



def region_loading(
    filelist: List[str],
    acr_list: List[str],
    atlas: np.ndarray,
    regions_nb: np.ndarray,
    regions_acr: np.ndarray,
    regions_centered: bool = True
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]):

    """
    Utility function for loading a single volume along with the necessary info: coordinates, file indices and shape of the volume.
    Note that 'nan' values are excluded during the process.

    Parameters
    ----------
    filelist : list of str
        The list of paths to the files that will be included in the clustering process.
    acr_list : list of str
        The list of the acronyms to be used during the clustering.
    atlas : ndarray
        3D volume of the atlas, where each voxel's value corresponds to its region number.
    regions_nb : ndarray
        Array containing the regions' numbers.
    regions_acr : ndarray
        Array containing the regions' acronyms. Same order as regions_acr.
    regions_centered : bool
        If True, the output will be cropped so that the boundaries fit the selected areas.

    Returns
    -------
    data : ndarray
        Contains the volume data (time traces).
    coords : ndarray
        Contains the coordinates in the volume of each voxel.
    file_indices : array
        Array filled with int values corresponding to the volume the data was extracted from. For example, 0 will correspond to data from the first loaded file.
    data_volume_shape : list of array
        list of shapes of the volumes (excluding the time dimension) for reconstructing the volumes later
    """

    # format acr_list: [(region_acr, hemisphere)] e.g. [('SCs', 'L')], 'L' = left, 'R' = right
    dataset, coords, file_indices = [], [], []

    mask = u.get_atlas_mask_from_regions(atlas, regions_nb, regions_acr, acr_list)

    non_zero_coords = np.argwhere(mask==1)
    if regions_centered:
        zmin, zmax = np.min(non_zero_coords[:,0]), np.max(non_zero_coords[:,0])
        ymin, ymax = np.min(non_zero_coords[:,1]), np.max(non_zero_coords[:,1])
        xmin, xmax = np.min(non_zero_coords[:,2]), np.max(non_zero_coords[:,2])
        volume_boundaries = [(zmin, zmax), (ymin, ymax), (xmin, xmax)]
    else:
        volume_boundaries = None

    data_volume_shape = atlas.shape

    coords_grid = np.argwhere(mask == 1)

    for i, file in enumerate(filelist):

        data = np.load(file)
        data = data[mask]

        idx_nan = ~np.isnan(data).any(axis=1)
        dataset.append(data[idx_nan, :])

        coords_tmp = coords_grid[idx_nan, :]
        coords.append(coords_tmp)

        N = np.sum(idx_nan)
        file_indices += [i]*N

    data_len = [d.shape[-1] for d in dataset]
    assert data_len.count(data_len[0]) == len(data_len), "All data must have the same number of frames. Check the different stimuli and/or experimental groups. Exiting..."
    dataset, coords = np.vstack(dataset), np.vstack(coords)

    return(dataset, coords, np.array(file_indices), data_volume_shape, volume_boundaries)




class SingleVoxelClustering:

    """
    Main class handling the single voxel clustering process.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    data : ndarray
        Data volume (3D in time) on which to perform the clustering.
    coords : ndarray
        Array containing the volume coordinates of each voxel in 'data'.
    file_indices : array
        Array containing for each voxel the index identifying the corresponding file.
    names : array
        Array containing for each voxel the index identifying the corresponding file.
    data_volume_shape : array
        The shape of the 'data' array.
    fe_method : str
        Selection of the method for extracting the features, check the user guide for more info.
    fe_params : dict
        Dictionary of parameters for the feature extraction method.
    noise_th : float
        Threshold on the standard deviation of the baseline to remove 'noisy' voxels.
    """

    def __init__(self,
        n_clusters: int,
        data: np.ndarray,
        coords: np.ndarray,
        file_indices, ### TO DO
        names: List[str],
        data_volume_shape: np.array,
        fe_method: str = 'pca',
        fe_params: Union[dict, None] = {},
        noise_th: Union[float, None] = None
        ):

        self.n_clusters = n_clusters
        self.data = data[..., 4:]
        self.coords = coords
        self.file_indices = file_indices
        self.names = names
        self.data_volume_shape = data_volume_shape

        if noise_th is not None:
            noise_mask = np.std(self.data, 1) > noise_th
            self.data[np.isinf(self.data)] = 0
            self.data[noise_mask, :] = 0

        if fe_method is not None:
            self.features = self.extract_features(fe_method, fe_params)
        else:
            self.features = None

        cm = plt.get_cmap('tab10') if n_clusters <= 10 else plt.get_cmap('tab20')
        self.cmap = [cm(i) for i in range(n_clusters)]


    def print_file_info(self):

        """
        To print in terminal the names of files included in the process and their associated indices.
        """

        for f, n in zip(self.file_indices, self.names):
            print(int(f), n)


    def extract_features(self,
        fe_method: str, # OR INSTANCE, CHECK HOW TO DECLARE IT
        fe_params: dict
        ):

        """
        Method for applying the feature extraction process to the data.

        Parameters
        ----------
        fe_method : str | object
            Refer to the feature extraction package for more details. Accepted values: pca, ica, nnmf, or an object with 'fit' or 'fit_predict' method
        fe_params : dict
            Dictionary containing the parameters to provide to the feature extractor.

        Returns
        -------
        features : ndarray
            Features extracted from the data.
        """

        fe = FeatureExtractor(fe_method, fe_params)
        features = fe.process(self.data)
        print("Dimensionality reduction done!")

        return(features)


    def cluster_dataset(self):

        #### OPTIONS FOR OTHER CHOICES THAN KMEANS?

        """
        Method for attributing each time trace with a cluster. Default model is K-Means.
        """

        self.model = MiniBatchKMeans(n_clusters=self.n_clusters, n_init=50, batch_size=8192)
        data = self.data if self.features is None else self.features
        self.model.fit(np.unique(data, axis=0))
        self.labels = self.model.predict(data)

        print("Clustering done. Labels have been predicted.")


    def merge_clusters(self, pairs_to_merge, adjust_cmap=True):

        """
        Utility function for merging clusters together.

        Parameters
        ----------
        pairs_to_merge : list of tuple of int
            List of pair of clusters to be merged. Higher cluster number will be merged to the lower cluster number, ie if cluster 5 and 3 are merged,
            in the cluster maps 5s will become 3s. Example input: [(5,3), (1,6)].
        adjust_cmap : bool
            Whether or not to adjust the colormap to the new number of clusters.
        """

        label_adjustment = [] # to adjust labels later, eg. cluster 5 becomes cluster 4 if clusters 1 and 2 were merged

        for pair in pairs_to_merge:
            # cl_src becomes cl_dst
            if pair[0] > pair[1]:
                cl_src, cl_dst = pair[0], pair[1]
            else:
                cl_src, cl_dst = pair[1], pair[0]
            cl_src, cl_dst = cl_src-1, cl_dst-1 # in displays clusters are presented from 1 to n_clusters, but encoded in the object in the range (0, n_clusters-1)
            assert cl_src < self.n_clusters , "You cannot merge a cluster id higher than the total number of clusters."

            self.labels[self.labels == cl_src] = cl_dst
            label_adjustment.append(cl_src)

        label_adjustment.sort(reverse=True)
        for cl_src in label_adjustment:

            self.labels[self.labels >= cl_src] -= 1

            if adjust_cmap:
                self.cmap.pop(cl_src)

        self.n_clusters -= len(label_adjustment)


    def generate_background_img(self):

        """
        Method for generating the microDoppler background image
        """

        pass


    def get_signals_or_coords(self,
        output: str = 'signals',
        file_idx: Union[int, None] = None
        ):

        """
        Generates an iterator yielding for each sample in data the cluster id and either the data or the coords

        Parameters
        ----------
        output : str
            If 'signals', yield the time trace of each sample in data. If 'coords', the coordinates instead.
        file_idx : int | None
            If int, yield the signals associated with the file index. Else, returns everything.
        """

        if file_idx is None:
            rows_file = np.ones(len(self.file_indices)).astype('bool')
        else:
            rows_file = (self.file_indices == file_idx)

        res_file = self.data[rows_file, :] if output == 'signals' else self.coords[rows_file, :]
        labels_file = self.labels[rows_file]

        for cl_id in range(self.n_clusters):

            rows_cl = (labels_file == cl_id)
            res_cl = res_file[rows_cl, :]

            yield(cl_id, res_cl)


    def reset_cmap(self):

        """
        Function for reseting the colormap to default values, including when the number of cluster has changed.
        """

        cm = plt.get_cmap('tab10') if self.n_clusters <= 10 else plt.get_cmap('tab20')
        self.cmap = [cm(i) for i in range(self.n_clusters)]


    def change_cmap(self,
        new_cmap: Union[list, str],
        continuous=True
        ):

        """
        Method for changing the colormap.

        Parameters
        ----------
        new_cmap : list | str
            Either a list of RGBA values or the name of a colormap available in pyplot.
        continuous : bool
            Set to True if the colormap is continuous and False if sequential.
        """

        if type(new_cmap) == list:

            assert len(new_cmap) >= self.n_clusters, "Not enough colors provided for the number of clusters."
            self.cmap = new_cmap

        else:

            cm = plt.get_cmap(new_cmap)
            if continuous:
                self.cmap = [cm(i / self.n_clusters) for i in range(self.n_clusters)]
            else:
                self.cmap = [cm(i) for i in range(self.n_clusters)]


    def switch_colors_in_cmap(self, color_swaps):

        """
        Utility method for swapping colors in the colormap to adjust the display of the cluster traces.

        Parameters
        ----------
        color_swaps : list of tuple of int
            List containing pairs of colors to be switched.
        """

        for (c1, c2) in color_swaps:
            tmp = self.cmap[c1-1]
            self.cmap[c1-1], self.cmap[c2-1] = self.cmap[c2-1], tmp


    def plot_signals(self,
        file_idx: Union[int, None] = None,
        display: str = 'all',
        ncols: int = 4,
        scale: Union[None, Tuple[float, float]] = None,
        stimulation_pattern = None
        ):

        """
        Method for plotting the signals in each cluster.

        Parameters
        ----------
        file_idx : int | None
            If int, only the traces of the file with index 'file_idx' will be plotted.
        display : str
            Defines the way traces will be displayed
            'all': display all traces
            'mean_std': only display the mean trace and the standard deviation.
        ncols : int
            Number of columns for the display.
        scale : tuple of float | None
            Min and max scales for the display (arguments vmin / vmax from plt.imshow). If None, auto scale will be used.
        stimulation_pattern : list of tuple of int | None
            List containing beginnings and ends of stimulation windows.
        """

        nrows = int(self.n_clusters / ncols - 1e-5) + 1
        signals_iterator = iter(self.get_signals_or_coords(output='signals', file_idx=file_idx))
        f = plt.figure()

        for cl_id, signals in signals_iterator:

            ax = plt.subplot(nrows, ncols, 1+cl_id)
            m, sd = np.median(signals, 0), np.std(signals, 0)
            x = [i+4 for i in range(len(m))]

            plt.xlim(0, len(m)+4)
            if scale:
                plt.ylim(scale[0], scale[1])
            else:
                scale = plt.gca().get_ylim()

            if stimulation_pattern:
                for pattern in stimulation_pattern:
                    plt.fill_between(range(pattern[0], pattern[1]), scale[0], scale[1], color=[0.8,0.8,0.8,.5])

            if display == 'mean_std':
                plt.fill_between(np.arange(4, len(m)+4), m-sd, m+sd, color=self.cmap[cl_id])

            else:
                for i, s in enumerate(signals):
                    if i % 10 == 0:
                        plt.plot(x, s, linewidth=0.2, color=self.cmap[cl_id])

            plt.plot(x, m, color='black')
            #plt.ylim(-1, 3)

        #plt.show()


    def flatten_volume(self,
        volume: np.ndarray,
        volume_boundaries: np.array = None,
        ncols: int = 8,
        apply_colormap: bool = True,
        ) -> np.ndarray:

        """
        Utility function for creating flattened 3D volumes towards displaying them as a mosaique of 2D images, where each value is the cluster attribution of the associated voxel.

        Parameters
        ----------
        volume : ndarray
            The 3D volume to be flattened.
        volume_boundaries : array | None
            For setting custom boundaries, if None the dimensionses of the whole volume will be taken.
        ncols : int
            Number of columns to be used for the flattening.
        apply_colormap : bool
            If True, the colormap is applied to replace cluster IDs with the corresponding color.

        Returns
        -------
        res : ndarray
            A 2D array with values being either int (cluster ID) or tuple (color associated with cluster ID) depeding on the 'apply_colormap' parameter.
        """

        res, nx, nz, nrows = u.reshape_volume_for_display(volume, volume_boundaries, ncols, return_for_clustering=True)

        if apply_colormap:

            res_color = np.zeros((nz*nrows, nx*ncols, 4))
            for i in range(self.n_clusters):
                res_color[res==i+1, :] = self.cmap[i]
            res_color[res == -1] = [0,0,0,1]

            return(res_color)

        else:
            return(res)


    def get_cluster_locations(self,
        file_idx: int,
        volume_boundaries: np.array = None,
        plot: bool = True,
        registered: bool = False, #### THIS OPTION HAS NOT BEEN PROPERPLY TESTED
        atlas_path: Union[None, str] = None,
        cluster_ids = None,
        extrema = None,
        hires_dst = None
        ):

        """
        Method for displaying the map in which each voxel is color-coded depending on its cluster attribution.

        Parameters
        ----------
        file_idx : int
            Index of the file to be displayed.
        volume_boundaries : array | None
            For setting custom boundaries, if None the dimensionses of the whole volume will be taken.
        plot : bool
            If True, a map in which each voxel is color-coded depeding on its cluster attribution will be displayed.
        registered : bool
            If the data was registered during the data loading, set to True to have the atlas info available through the cursor.
        atlas_path : str
            Path to the atlas that the data has been registered to.
        cluster_ids : list of int | None
            The list of cluster to be displayed.
        extrema : tuple
            Dictionary containing for each cluster the extrema for normalization.
        hires_dst : str | None
            If string, path where the hi-res files will be saved. If None, hi-res files won't be generated nor saved.

        Returns
        -------
        volume : ndarray
            A volume (same size as the original data) in which each value represents the cluster attribution.
        """

        ### WATCH OUT: IF file_idx IS NONE, OUTPUT WILL BE SHITTY. TO FIX AT SOME POINT

        if atlas_path is None:
            atlas_path, _, _ = u.get_atlas_and_info_paths(100, "_nolayersnoparts")

        coords_iterator = iter(self.get_signals_or_coords(output='coords', file_idx=file_idx))
        volume = np.zeros(self.data_volume_shape)

        for cl_id, coords in coords_iterator:
            if cluster_ids and cl_id not in cluster_ids:
                continue
            volume[coords[:,0], coords[:,1], coords[:,2]] = cl_id+1

        volume_flat = self.flatten_volume(volume, volume_boundaries)

        if extrema:
            amplitude_map = self.compute_amplitude_map(file_idx, extrema)
            amplitude_map[volume == 0] = 0. #to handle when cluster_ids is not None
            amplitude_map = self.flatten_volume(amplitude_map, volume_boundaries, apply_colormap=False)
            amplitude_map[amplitude_map < 0] = 0. #to handle when cluster_ids is not None
            volume_flat[:, :, 3] = amplitude_map

        if hires_dst:
            f, ax = plt.subplots(figsize=(30,60))
            ax.set_title(self.names[file_idx])
            ax.imshow(volume_flat)
            f.savefig(os.path.join(hires_dst, F"cluster_map_{self.names[file_idx]}.svg"))
            plt.close(f)

        if plot:
            f, ax = plt.subplots()
            ax.set_title(self.names[file_idx])
            ax.imshow(volume_flat)
            if registered ==  True:
                atlas = np.load(atlas_path)
                atlas_flat = self.flatten_volume(atlas, volume_boundaries, apply_colormap=False)
                ax.format_coord = u.Formatter(atlas_flat, self.regions_acr, self.regions_nb)

        return(volume)


    def compute_amplitude_map(self,
        file_idx,
        extrema
        ):

        """
        Method for computing the transparency map based on the normalized amplitude.

        Parameters
        ----------
        file_idx : int
            Index of the file to be displayed.
        extrema : dict
            A dictionary containing the extrema for each cluster for the amplitude normalization.

        Returns
        -------
        amplitude_map : ndarray
            The map containing the transparency values.
        """

        amplitude_map = np.zeros(self.data_volume_shape)

        signals_iterator = iter(self.get_signals_or_coords(output='signals', file_idx=file_idx))
        coords_iterator = iter(self.get_signals_or_coords(output='coords', file_idx=file_idx))

        for (cl_id, sig), (_, coords) in zip(signals_iterator, coords_iterator):

            normalized_amp = (np.max(np.abs(sig), 1) - extrema[str(cl_id)][0]) / (extrema[str(cl_id)][1] - extrema[str(cl_id)][0])
            normalized_amp[normalized_amp > 1] = 1
            normalized_amp[normalized_amp < 0.1] = 0.1
            amplitude_map[coords[:,0], coords[:,1], coords[:,2]] = normalized_amp

        return(amplitude_map)




class SingleVoxelClusteringWrapper:

    """
    Wrapper to perform the single voxel clustering in various configurations.

    Parameters
    ----------
    method : str
        Choice of the method, between:
            - volume: all voxels in the image
            - hemisphere: all voxels in a given or both hemispheres
            - structure: all voxels belonging to a list of structures
            - multiregion: all voxels belonging to a list of regions
    n_clusters : int
        Number of clusters to search for.
    filelist : list of string
        List of paths to the volumes whose voxels will be clustered.
    atlas_path : string
        Path to the file containing the atlas volume.
    regions_info_path : string
        Text file listing all the regions included in the atlas.
    fe_method : string
        Selection of the feature extraction method, to choose between 'pca', 'ica', 'nnmf' or custom.
    fe_params : dict
        Selection of the feature extraction parameters.
    noise_th : float.
        Threshold on the standard deviation of the baseline to remove 'noisy' voxels.
    """

    def __init__(self,
        method: str,
        n_clusters: int,
        filelist: List[str],
        atlas_path: str,
        regions_info_path: str,
        fe_method: Union[str, None] = None,
        fe_params: dict = {},
        noise_th: float = None,
        registered: bool = True,
        normalization: Union[str, None] = None
        ):

        assert method in ['volume', 'structure', 'hemisphere', 'multiregion'], "Unrecognized data selection method: choose between brainwide, structure, hemisphere, multi_region."
        self.method = method

        self.n_clusters = n_clusters
        self.filelist = filelist
        self.names = [u.name_from_path(file) for file in filelist]
        self.fe_method = fe_method
        self.fe_params = fe_params
        self.noise_th = noise_th

        self.atlas = u.load_atlas(atlas_path)
        self.regions_nb, self.regions_acr, self.groups_acr = u.extract_info_from_region_file(regions_info_path)

        # necessary to adjust displays and select data
        self.volume_boundaries = None
        self.registered = registered

        self.normalization = normalization


    def normalize(self, data):

        if self.normalization == "0-1":
            data = (data[:,:] - np.min(data, 1)[:, None]) / (np.max(data, 1) - np.min(data, 1))[:, None]

        return(data)


    def get_regions_from_groups(self,
        group: str,
        hemisphere: str
        ):

        """
        Utility function for converting an anatomical group into a list of regions included in that group in the format expected by the clustering code.

        Parameters
        ----------
        group : str
            Acronym of the anatomical group
        hemisphere : str
            'L' or 'R' respectively for left and right hemispheres, or 'LR' for both.

        Returns
        -------
        res : list
            List of tuples in the format (region acronym, hemisphere 'L' or 'R')
        """

        res = []
        regions = self.regions_acr[self.groups_acr == group]

        if hemisphere == 'LR':
            for r in regions:
                res.append((r, 'L'))
                res.append((r, 'R'))
        else:
            for r in regions:
                res.append((r, hemisphere))

        return(res)


    def process(self,
        acr_list: Union[str, List[Tuple[str, str]], None] = None
        ):

        """
        Method function to call for clustering the data.

        Parameters
        ----------
        acr_list : list of tuple | None
            If None, all regions are considered. Otherwise, only the acronyms present in the list will be included in the clustering process.
            The format of the tuple is (<region acronym>, <hemisphere>).
        """

        if self.method == 'volume':

            data, coords, file_indices, data_volume_shape = volume_loading(self.filelist)

        elif self.method == 'hemisphere':

            assert acr_list in ['L', 'R', 'LR'], "Incorrect input for method 'hemisphere': should be either 'L' (left), 'R' (right) or 'LR' (both)"
            acr_list = u.get_regions_from_hemisphere(acr_list, self.regions_acr, self.groups_acr)
            data, coords, file_indices, data_volume_shape, self.volume_boundaries = region_loading(self.filelist, acr_list, self.atlas, self.regions_nb, self.regions_acr)

        elif self.method == 'structure':

            assert acr_list is not None, "No list of acronyms provided"
            res = []
            for acr in acr_list:
                res += u.get_regions_from_groups(acr[0], acr[1], self.regions_acr, self.groups_acr)#self.get_regions_from_groups(acr[0], acr[1])
            data, coords, file_indices, data_volume_shape, self.volume_boundaries = region_loading(self.filelist, res, self.atlas, self.regions_nb, self.regions_acr)

        else: # self.method == 'multiregion'

            assert acr_list is not None, "No list of acronyms provided"
            acr_list_full = []
            for elt in acr_list:
                assert elt[1] in ['L', 'R', 'LR'], "Hemispheres values should be either 'L' (left), 'R' (right) or 'LR' (both)"
                if elt[1] == 'LR':
                    acr_list_full.append((elt[0], 'L'))
                    acr_list_full.append((elt[0], 'R'))
                else:
                    acr_list_full.append(elt)

            data, coords, file_indices, data_volume_shape, self.volume_boundaries = region_loading(self.filelist, acr_list_full, self.atlas, self.regions_nb, self.regions_acr)

        if self.normalization:
            data = self.normalize(data)

        self.svc = SingleVoxelClustering(self.n_clusters, data, coords, file_indices, self.names, data_volume_shape, fe_method=self.fe_method, fe_params=self.fe_params, noise_th=self.noise_th)
        self.svc.cluster_dataset()

        if self.registered:
            self.svc.regions_acr = self.regions_acr
            self.svc.regions_nb = self.regions_nb


    def get_names(self):

        """
        Method for printing the names of the data elements in the clustering object.
        """

        print(F"Elements available for display: {self.names}")


    def reset_cmap(self):

        """
        Method for reseting the color map.
        """

        self.svc.reset_cmap()


    def change_cmap(self,
        new_cmap: Union[None, list, str] = None,
        continuous: bool = True
        ):

        """
        Method for changing the colormap.

        Parameters
        ----------
        new_cmap : None | list | str
            Either a list of RGBA values or the name of a colormap available in pyplot.
        continuous : bool
            Set to True if the colormap is continuous and False if sequential.
        """

        if new_cmap is not None:
            self.svc.change_cmap(new_cmap=new_cmap, continuous=continuous)


    def switch_colors_in_cmap(self, color_swaps):

        """
        Utility method for swapping colors in the colormap to adjust the display of the cluster traces.

        Parameters
        ----------
        color_swaps : list of tuple of int
            List containing pairs of colors to be switched.
        """

        self.svc.switch_colors_in_cmap(color_swaps)


    def plot_signals(self,
        display: str = 'all',
        scale: Union[None, Tuple[float, float]] = None,
        stimulation_pattern = None
        ):

        """
        Utility function for calling the display function for temporal traces depending on the method that was selected.

        Parameters
        ----------
        display : str
            Either 'all' to display all signals or 'mean_std' for mean/std of the signals.
        scale : tuple of float | None
            Min and max scales for the display (arguments vmin / vmax from plt.imshow). If None, auto scale will be used.
        stimulation_pattern : list of tuple of int | None
            List containing beginnings and ends of stimulation windows.
        """

        assert display in ['all', 'mean_std'], "Please set display param to 'all' or 'mean_std'"
        self.svc.plot_signals(display=display, scale=scale, stimulation_pattern=stimulation_pattern)


    def compute_amplitude_extrema(self,
        quantiles
        ):

        """
        Utility function for computing the extrema of each cluster for computing the transparency value associated with parameter 'amplitude_transparency'
        from method 'display_cluster_locations'.
        Note: since the extrema are noisy, quantiles are used instead.

        Parameters
        ----------
        quantiles : list of float
            List of size 2 containing the quantiles for the minimum and maximum estimation respectively.

        Returns
        ----------
        extrema : dict of tuple
            Dictionary containing for each cluster the extrema for normalization.
        """

        assert len(quantiles) == 2, "Parameter 'quantiles' should be of length exactly 2."
        signals = self.get_signals(reduction=None)
        extrema = {cl_id: [] for cl_id in signals[next(iter(signals))]}

        for name in self.names:
            for cl_id in signals[name]:
                extrema[cl_id].append(signals[name][cl_id])

        for cl_id in extrema:
            extrema[cl_id] = np.vstack(extrema[cl_id])
            extrema[cl_id] = np.quantile(np.min(np.abs(extrema[cl_id]), 1), quantiles[0]), np.quantile(np.max(np.abs(extrema[cl_id]), 1), quantiles[1])

        return(extrema)


    def display_cluster_locations(self,
        names: Union[list, None] = None,
        amplitude_transparency = False,
        quantiles = [0.05, 0.95],
        cluster_ids = None,
        hires_dst = None
        ):

        """
        Utility function for calling the display function for spatial locations depending on the method that was selected.

        Parameters
        ----------
        names : list
            Identifiers of the elements to be displayed.
        amplitude_transparency : bool
            Whether or not to modulate the amplitude of the cluster based on the amplitude of the signals. Modulation is cluster specific.
        quantiles : list of float
            List of size 2 containing the quantiles for the minimum and maximum estimation respectively. Only used if amplitude transparency is True.
        cluster_ids : None | list
            List of the clusters to be displayed. If None, all clusters will be displayed.
        hires_dst : str | None
            If string, path where the hi-res files will be saved. If None, hi-res files won't be generated nor saved.
        """

        names = self.names if names is None else names

        if amplitude_transparency:
            extrema = self.compute_amplitude_extrema(quantiles)
        else:
            extrema = None

        if cluster_ids: # cluster range in 0 -> n_clusters-1
            cluster_ids = [cl_id - 1 for cl_id in cluster_ids]

        for name in names:
            i = self.names.index(name)
            print(name, i)
            self.svc.get_cluster_locations(i, self.volume_boundaries, registered=self.registered, plot=True, extrema=extrema, cluster_ids=cluster_ids, hires_dst=hires_dst)


    def get_signals(self,
        reduction='median'
        ) -> dict:

        """
        Utility function for accessing the temporal signals associated with each cluster.

        Parameters
        ----------
        reduction : str | None
            If 'median', returns the median of all signals in each cluster. If None, returns all signals.

        Returns
        -------
        res : dict
            A dictionary with keys being the cluster IDs and values the associated temporal trace(s).
        """

        res = {}

        for i, name in enumerate(self.names):

            res[name] = {}
            signals_iterator = iter(self.svc.get_signals_or_coords(output='signals', file_idx=i))

            for cl_id, signals in signals_iterator:

                if reduction == 'median':
                    res[name][str(cl_id)] = np.median(signals, 0)
                else:
                    res[name][str(cl_id)] = signals

        return(res)


    def get_cluster_maps(self) -> dict:

        """
        Utility function for getting the arrays containing the cluster maps where each voxel's value represents its cluster attribution.

        Parameters
        ----------
        use_boundaries : bool
            Whether or not to adjust the boundaries of the maps to the selected regions or structure. If False, the full map are output.

        Returns
        -------
        res : dict
            A dictionary with keys being the element names (eg all sessions IDs) and values the associated cluster maps.
        """

        res = {}
        for i, name in enumerate(self.names):
            map = self.svc.get_cluster_locations(i, None, registered=False, plot=False)
            res[name] = map

        return(res)


    def merge_clusters(self,
        pairs_to_merge,
        adjust_cmap=True
        ):

        """
        Utility function for merging clusters together.

        Parameters
        ----------
        pairs_to_merge : list of tuple of int
            List of pair of clusters to be merged. Higher cluster number will be merged to the lower cluster number, ie if cluster 5 and 3 are merged,
            in the cluster maps 5s will become 3s. Example input: [(5,3), (1,6)].
        adjust_cmap : bool
            Whether or not to adjust the colormap to the new number of clusters.
        """

        self.svc.merge_clusters(pairs_to_merge, adjust_cmap=adjust_cmap)
        self.n_clusters = self.svc.n_clusters


    def display_atlas_mask_selected_regions(self,
        regions_list
        ):

        """
        Utility function to display on the atlas the list of selected regions.

        Parameters
        ----------
        regions_list : list of str
            List of regions acronyms to be displayed on the atlas.
        """

        cm = plt.get_cmap('jet')
        cm.set_under(color='white')
        atlas_sub = np.zeros_like(self.atlas)-1

        for i, (r, h) in enumerate(regions_list):

            print(r)
            r_idx = self.regions_nb[self.regions_acr == r][0]
            s = -1 if h == 'L' else 1
            atlas_sub[self.atlas == s*r_idx] = i#/len(regions_list)

        atlas_sub = u.reshape_volume_for_display(atlas_sub, volume_boundaries=self.volume_boundaries)
        plt.figure(figsize=(30,20))
        plt.imshow(atlas_sub, cmap='jet')
        plt.show()
