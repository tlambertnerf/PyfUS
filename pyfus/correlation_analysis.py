import numpy as np
import matplotlib.pyplot as plt
import typing
import pyfus.utils as u
from typing import List, Union, Dict
from time import time


class CorrelationAnalysis:

    """
    Object handling the generation and display of correlation maps and Z-maps.

    Parameters
    ----------
    data_paths : str | list of str
        Paths to the data to be analyzed.
    correlation_pattern : list of tuple | array | dict of array
        - if list of tuple, first and second elements of each tuple are respectively beginning and end of a square pulse.
        - if np.array, this array should be the same size as the temporal dimension of data. This array will be correlated with each data. Options 1 and 2 assume that all data have same number of frames.
        - if dict containing np.arrays, keys should match the data names (check the info file, key are before 'n_samples') and arrays should have the same length as their associated data.
    n_samples: None | int | dict of int
        Number of samples used for computing the data reduction, if applicable (if not, set to None).
            - if None, no Z-map will be computed.
            - if int, assumes that all data have the same number of samples
            - if dict, keys should match the data names
    significance_threshold: float
        The statistical significance threshold used for the computation of the Z-maps. Either 0.05, 0.01 or 0.001.
    """

    def __init__(self,
        data_paths: Union[str, List[str]],
        correlation_pattern: Union[List[tuple], np.array, Dict[str, np.array]],
        n_samples: Union[None, int, Dict[str, int]] = None,
        significance_threshold: float = 0.05
        ):

        if type(data_paths) == str:
            data_path = [data_path]
        self.data_paths = data_paths

        assert (type(n_samples) in [None, int]) or (len(n_samples) == len(data_paths)), "Incorrect number of values for n_samples. Either None, one value or same number of values as data_paths."
        self.n_samples = n_samples

        if type(correlation_pattern) == list:
            self.mode = 'square'
        elif type(correlation_pattern) in [np.array, np.ndarray]:
            self.mode = 'custom'
        elif type(correlation_pattern) == dict:
            self.mode = 'custom_per_element'
        else:
            raise(TypeError, "Wrong type for variable correlation_pattern. Should be list of tuple, np.array or dict[np.array]")
        self.init_pattern(correlation_pattern)

        assert significance_threshold in [0.05, 0.01, 0.001], 'Please select a significance threshold in [0.05, 0.01, 0.001]'
        self.significance_threshold = str(significance_threshold)
        self.z_thresholds = {'0.05': 1.64, '0.01': 2.33, '0.001': 3.08}


    def init_pattern(self,
        correlation_pattern: Union[List[tuple], np.array, Dict[str, np.array]]
        ):

        """
        Function for creating the correlation pattern based on the selected mode and parameters. Automatically called during initialization of the object.

        Parameters
        ----------
        correlation_pattern : list of tuple | array | dict of array
            See description in the init function.
        """

        if self.mode == 'square':

            T = np.load(self.data_paths[0]).shape[-1]
            pattern = np.zeros(T)
            for (a, b) in correlation_pattern:
                pattern[a:b] = 1
            pattern = {'pattern': pattern}
            keys = ['pattern']*len(self.data_paths)

        elif self.mode == 'custom':

            pattern = {'pattern': correlation_pattern}
            keys = ['pattern']*len(self.data_paths)

        else: # mode == custom_per_element

            pattern = correlation_pattern
            keys = [u.name_from_path(data_path) for data_path in self.data_paths]
            test_keys = [(key in keys) for key in correlation_pattern]
            assert np.sum(test_keys) == len(keys), F"Missing keys in correlation_pattern. Check that you have: {keys}"

        self.correlation_patterns, self.keys = pattern, keys


    def compute_corr_map(self,
        data_path: str,
        corr_pattern: np.array,
        n_samples: int,
        plot: bool = True,
        registered: bool = False,
        atlas_path: Union[str, None] = None,
        regions_info_path: Union[str, None] = None
        ):

        """
        Function for computing the correlation map for a given data. The loading is handled in the function, as well as the display of the maps if requested.

        Parameters
        ----------
        data_path : str
            Path to the data to be loaded.
        corr_pattern : array
            Correlation pattern to be correlated with the data.
        n_samples : int | None
            Number of samples used for the reduction of data (e.g., number of trials x number of sessions). Used to compute the Z-map. Note: if None, no Z-map will be computed.
        plot : bool
            If True, correlation and Z-maps (if applicable) will be displayed on a window.
        registered : bool
            Whether or not the data was registered. Only required to display the atlas overlay.
        atlas_path: str
            Path to the atlas the data has been registered to. Only necessary if registered is True.
        regions_info_path : str
            Path to the region info path. Only necessary if registered is True.

        Returns
        -------
        corr_map : ndarray
            Ndarray containing the correlation map.
        z_map : ndarray
            Ndarray containing the Z-map computed using the number of samples and the significance threshold.
        """

        data = np.load(data_path)
        assert data.shape[-1] == len(corr_pattern), F'Correlation pattern and data respective lengths don\'t match: {data.shape[-1]} VS {len(corr_pattern)}'

        corr_map = np.zeros(data.shape[:-1])

        data_shape = data.shape
        tmp_data = data.reshape((-1, data.shape[-1]))
        data_centered = tmp_data - tmp_data.mean(1)[:, None]
        pattern_centered = corr_pattern - corr_pattern.mean()
        ss_data, ss_pattern = (data_centered**2).sum(1), (pattern_centered**2).sum()
        corr_map = np.dot(data_centered, pattern_centered) / np.sqrt(np.dot(ss_data[:, None], ss_pattern[None]))
        corr_map = corr_map.reshape(data.shape[:-1])

        # Fisher transform
        if n_samples is not None:
            z_map = (np.sqrt(n_samples - 3)*np.arctanh(corr_map)) > self.z_thresholds[self.significance_threshold]
        else:
            z_map = np.zeros(data.shape[:-1])

        if plot:
            corr_map_2D, z_map_2D = u.reshape_volume_for_display(corr_map), u.reshape_volume_for_display(z_map)
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            fig.suptitle(u.name_from_path(data_path))
            im = ax1.imshow(corr_map_2D, cmap='jet', vmin=-1.0, vmax=1.0)
            plt.colorbar(mappable=im, ax=ax1)
            ax2.imshow(z_map_2D.astype('int'), cmap='binary', vmin=0, vmax=1)
            if registered == True:
                atlas = np.load(atlas_path)
                atlas_flat = u.reshape_volume_for_display(atlas, volume_boundaries=None)
                regions_nb, regions_acr, _ = u.extract_info_from_region_file(regions_info_path)
                ax1.format_coord = u.Formatter(atlas_flat, regions_acr, regions_nb)
                ax2.format_coord = u.Formatter(atlas_flat, regions_acr, regions_nb)

        return(corr_map, z_map)


    def process(self,
        plot: bool = True,
        return_map: str = 'zmap',
        registered: bool = False,
        atlas_path: Union[str, None] = None,
        regions_info_path:  Union[str, None] = None
        ):

        """
        Main function computing the correlation maps for each of the selected data.

        Parameters
        ----------
        plot : bool
            If True, all the computed correlation maps and Z-maps (if computed) will be displayed in separate windows.
        return_map : str
            Either 'both', 'zmap' or 'cmap'. Determine what will be output by the function.
        MISSING DOC

        Returns
        -------
        res : dict
            A dictionary with keys being the names of the selected data, and values a tuple whose 1st element is the correlation map and the 2nd the Z-map.

        """

        if (self.n_samples is None) or (type(self.n_samples) == int):
            n_samples_dict = {key: self.n_samples for key in self.keys}

        if atlas_path is None or regions_info_path is None:
            atlas_path, _, regions_info_path = u.get_atlas_and_info_paths(100, '_nolayersnoparts')

        res = {}

        for data_path, key in zip(self.data_paths, self.keys):

            corr_pattern = self.correlation_patterns[key]
            corr_map, z_map = self.compute_corr_map(data_path, corr_pattern, n_samples_dict[key], plot=plot, registered=registered, atlas_path=atlas_path, regions_info_path=regions_info_path)

            res[u.name_from_path(data_path)] = (corr_map, z_map)

        if plot:
            plt.show()

        if return_map == 'zmap':
            for key in res:
                res[key] = res[key][1].astype('int')
        elif return_map == 'cmap':
            for key in res:
                res[key] = res[key][0].astype('float')
        else:
            pass

        return(res)
