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

    mask = np.zeros(atlas.shape)
    for acr, h in acr_list:
        assert len(regions_nb[regions_acr == acr]) != 0, F"Acronym \'{acr}\' is not correct. Please check the atlas."
        nb = regions_nb[regions_acr == acr][0]
        h_sign = -1 if h == 'L' else 1
        mask[atlas == nb*h_sign] = 1
    mask = mask.astype('bool')

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
        #self.model = AgglomerativeClustering(n_clusters=self.n_clusters, metric='euclidean', linkage='ward')
        data = self.data if self.features is None else self.features
        self.model.fit(np.unique(data, axis=0))
        self.labels = self.model.predict(data)
        #self.labels = self.model.fit_predict(data)
        print("Clustering done. Labels have been predicted.")


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


    def plot_signals(self,
        file_idx: Union[int, None] = None,
        display: str = 'all',
        ncols: int = 4,
        scale: Union[None, Tuple[float, float]] = None
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
        """

        nrows = int(self.n_clusters / ncols - 1e-5) + 1
        signals_iterator = iter(self.get_signals_or_coords(output='signals', file_idx=file_idx))
        f = plt.figure()

        for cl_id, signals in signals_iterator:

            ax = plt.subplot(nrows, ncols, 1+cl_id)
            m, sd = np.median(signals, 0), np.std(signals, 0)
            x = [i+4 for i in range(len(m))]

            if display == 'mean_std':
                plt.fill_between(np.arange(4, len(m)+4), m-sd, m+sd, color=self.cmap[cl_id])
                print(cl_id, np.mean(np.mean(signals, 1)), np.mean(np.std(signals, 1)))

            else:
                for i, s in enumerate(signals):
                    if i % 10 == 0:
                        plt.plot(x, s, linewidth=0.2, color=self.cmap[cl_id])

            plt.plot(x, m, color='black')
            plt.xlim(0, len(m)+4)
            if scale:
                plt.ylim(scale[0], scale[1])
            #plt.ylim(-1, 3)

        #plt.show()


    def flatten_volume(self,
        volume: np.ndarray,
        volume_boundaries: np.array = None,
        ncols: int = 8,
        apply_colormap: bool = True
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
        atlas_path: Union[None, str] = None
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
            volume[coords[:,0], coords[:,1], coords[:,2]] = cl_id+1

        volume_flat = self.flatten_volume(volume, volume_boundaries)

        if plot:
            f, ax = plt.subplots()
            ax.set_title(self.names[file_idx])
            ax.imshow(volume_flat)#, vmin=0, vmax=self.n_clusters+1)
            if registered ==  True:
                atlas = np.load(atlas_path) #### TRANSFORM INTO A PARAMETER, NOT HARD PATH
                atlas_flat = self.flatten_volume(atlas, volume_boundaries, apply_colormap=False)
                ax.format_coord = u.Formatter(atlas_flat, self.regions_acr, self.regions_nb)

        return(volume)




class SingleVoxelClusteringWrapper:

    """
    Wrapper to perform the single voxel clustering in various configurations.
    DEV NOTE: the wrapper was necessary because of h5py data loading.

    Parameters
    ----------
    method : str
        Choice of the method, between:
            - multi_BW: multiple volumes, brainwide clustering
            - single_BW: single volume, brainwide clustering
            - multi_region: multiple volumes, clustering on subset of regions
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
        registered: bool = True
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

            assert acr_list is not None, "No list of acronyms provided"
            hemi = acr_list
            assert hemi in ['L', 'R', 'LR'], "Hemispheres values should be either 'L' (left), 'R' (right) or 'LR' (both)"
            acr_list = []
            for g in ['P', 'MY', 'HY', 'TH', 'MB', 'CB', 'CTXsp', 'HPF', 'Isocortex', 'OLF', 'STR', 'PAL']:
                acr_list += self.get_regions_from_groups(g, hemi)
            data, coords, file_indices, data_volume_shape, self.volume_boundaries = region_loading(self.filelist, acr_list, self.atlas, self.regions_nb, self.regions_acr)

        elif self.method == 'structure':

            assert acr_list is not None, "No list of acronyms provided"
            res = []
            for acr in acr_list:
                assert acr[0] in ['P', 'MY', 'HY', 'TH', 'MB', 'CB', 'CTXsp', 'HPF', 'Isocortex', 'OLF', 'STR', 'PAL'], "The structure you selected does not exist..."
                assert acr[1] in ['L', 'R', 'LR'], "Hemispheres values should be either 'L' (left), 'R' (right) or 'LR' (both)"
                res += self.get_regions_from_groups(acr[0], acr[1])
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

        self.svc = SingleVoxelClustering(self.n_clusters, data, coords, file_indices, self.names, data_volume_shape, fe_method=self.fe_method, fe_params=self.fe_params, noise_th=self.noise_th)
        self.svc.cluster_dataset()

        if self.registered:
            self.svc.regions_acr = self.regions_acr
            self.svc.regions_nb = self.regions_nb


    def get_names(self):

        print(F"Elements available for display: {self.names}")


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


    def plot_signals(self,
        display: str = 'all',
        scale: Union[None, Tuple[float, float]] = None
        ):

        """
        Utility function for calling the display function for temporal traces depending on the method that was selected.

        Parameters
        ----------
        display : str
            Either 'all' to display all signals or 'mean_std' for mean/std of the signals.
        """

        assert display in ['all', 'mean_std'], "Please set display param to 'all' or 'mean_std'"
        self.svc.plot_signals(display=display, scale=scale)


    def display_cluster_locations(self,
        names: Union[list, None] = None
        ):

        """
        Utility function for calling the display function for spatial locations depending on the method that was selected.

        Parameters
        ----------
        names : list
            Identifiers of the elements to be displayed.
        """

        names = self.names if names is None else names

        #self.svc.display_cluster_locations(file_idx, self.volume_boundaries)
        for name in names:
            i = self.names.index(name)
            print(name, i)
            self.svc.get_cluster_locations(i, self.volume_boundaries, registered=self.registered, plot=True)
        #plt.show()


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


    def display_atlas_mask_selected_regions(self, regions_list):

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
