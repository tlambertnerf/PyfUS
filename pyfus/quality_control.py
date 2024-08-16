import numpy as np
import pandas as pd
import pyfus.utils as u
import typing, os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import deepcopy
from typing import List


class OutlierFrameRemoval:

    """
    Object for removing noisy frames from trials. Adapted from Brunner et al., Nature Protocols 2021, code data_stability.m and image_rejection.m
    /!\ designed to work with volumetric data in the order DV.AP.LR

    Parameters
    ----------
    interp : str
        Interpolation method. Check the argument 'kind' of scipy.interpolate interp1d function for further detail.
    verbose : bool
        If True, info about the amount of frames rejected for each element will be displayed in the terminal.
    """

    def __init__(self,
        interp: str = 'linear',
        verbose: bool = False
        ):

        self.interp = interp
        self.verbose = verbose


    def get_outliers(self,
        data: np.ndarray
        ) -> (np.ndarray, float):

        """
        Method for identifying the outlier frames. A frame is considered outlier when the average value of the frame is above the threshold of 3 times the standard deviation of the frames from the recording.

        Parameters
        ----------
        data : ndarray
            Volumetric data in the order DV.AP.LR + time

        Returns
        -------
        outliers : ndarray of bool
            Ndarray of the same size as data without the time dimension, with True as value if the average frame value is above 3 x std of the distribution.
        N_rejected : float
            Proportion of rejected frames.
        """

        data_red = np.nanmean(data, axis=(0,2))
        for y in range(data_red.shape[0]):
            data_red[y, :] = data_red[y, :] / np.nanmedian(data_red[y,:])

        # computes the std on the negative part of the distribution
        threshold = 1 + np.std(data_red[data_red < 1])*3

        outliers = data_red > threshold
        N_rejected = np.nanmean(outliers)

        return(outliers, N_rejected)


    def replace_outliers(self,
        data: np.ndarray,
        outliers: np.ndarray
        ) -> np.ndarray:

        """
        Method for replacing the frames identified as outliers with an interpolation of neighbouring frames.

        Parameters
        ----------
        data : ndarray
            Input data, 3D volume in time.
        outliers : ndarray
            Array of outliers as output by the 'get_outliers' method.

        Returns
        -------
        data : ndarray
            The new data where outlier frames were replace by interpolation.
        """

        accepted = ~outliers
        data_rej = data
        y_index_bool = np.array([i for i in range(data.shape[1])])

        for y in range(data.shape[1]):

            accepted_y = accepted[y, :]
            frames_ok = np.argwhere(accepted_y).flatten()

            data_ok = data[:, y, :, :]
            data_ok = data_ok[:, :, accepted_y]

            if np.sum(accepted_y) < len(accepted_y):

                interp = interp1d(frames_ok, data_ok, kind=self.interp, fill_value='extrapolate')
                data[:, y:y+1, :, ~accepted_y] = np.moveaxis(interp(np.argwhere(~accepted_y)), 3, 1)

        return(data)


    def __call__(self,
        data_list: List[np.ndarray],
        name: str
        ) -> List[np.ndarray]:

        """
        Main method for performing the outlier removal procedure.

        Parameters
        ----------
        data_list : list of ndarray
            List of data to be processed. Each data is a 4D volume (3D space in time).
        name : str
            Name of the batch of data being processed. Used for displaying the amount of replaced frames.

        Returns
        -------
        data_list : list of ndarray
            Same dimensions as the input data_list, but outliers frames were removed and replaced as detailed in the other methods.
        """

        N_rejected_all = []

        for i in range(len(data_list)):

            outliers, N_rejected = self.get_outliers(data_list[i])
            N_rejected_all.append(N_rejected)
            data_list[i] = self.replace_outliers(data_list[i], outliers)

        if self.verbose:
            print(F"{np.mean(N_rejected_all)}% of frames rejected for data {name}.")

        return(data_list)




class ReliabilityMaps:

    """ Computes reliability maps before trial averaging

    The reliability of a voxel is defined as the proportion of samples averaged for obtaining its value.
    Example: if there are 10 samples to be averaged, a voxel whose value result from the 10 samples will have a reliability of 100%, while only 7 samples (because the rest is NaN) will have 70% reliability.
    /!\\ Assumes that the input are registered data, otherwise the averaging will not make sense!

    Parameters
    ----------
    atlas : ndarray
        Atlas to be used (volume).
    regions_info_file : str
        Path to the regions info file.
    atlas_contours : ndarray
        Contours of the atlas (volume) to be used for the visualization. Purely optional.
    reliability_threshold : float
        Threshold to determine if a voxel is reliable or not.
    verbose : bool
        If True, results of the reliability check will be displayed in the terminal.
    """

    def __init__(self,
        atlas: np.ndarray,
        regions_info_file: str,
        atlas_contours: np.ndarray = None,
        reliability_threshold: float = 0.6,
        verbose: bool = False
        ):

        self.maps = []
        self.atlas = atlas
        self.atlas_contours = atlas_contours
        self.regions_nb, self.regions_acr, self.groups_acr = u.extract_info_from_region_file(regions_info_file)
        self.reliability_threshold = reliability_threshold
        self.verbose =  verbose


    def compute_reliability_map(self,
        data: np.ndarray
        ):

        """
        Computes the reliability map from an individual session and stores it in self.maps.

        Parameters
        ----------
        data : ndarray
            4D data (registered) from a session.
        """

        map = ~np.any(np.isnan(data), 3)
        self.maps.append(map.astype('float'))


    def process(self,
        output_folder: str,
        name: str,
        plot: bool = True
        ) -> np.ndarray:

        """
        Main method to be called to compute the maps, extract the regions below the reliability threshold and get the mask to remove unreliable voxels.

        Parameters
        ----------
        output_folder : str
            Folder in which the reliability maps will be saved.
        name : str
            Name of the element from which the reliability map is computed.
        plot : bool
            If True, the reliability maps will be displayed on the screen. /!\ The display will stop the data loading while the display window is open.

        Returns
        -------
        filter: ndarray of bool
            An array with value True where voxels are below the reliability threshold.
        """

        n_samples = len(self.maps)
        reliability_map = np.mean(self.maps, 0)

        summary = {'regions_acr':[], '% reliability':[]}

        for r in self.regions_nb:

            mask = (self.atlas == r)
            voxels_in_r, reliable_voxels_in_r = np.sum(mask), np.sum(reliability_map[mask])
            summary['regions_acr'].append(self.regions_acr[self.regions_nb == r][0])
            summary['% reliability'].append(reliable_voxels_in_r / voxels_in_r)

        summary = pd.DataFrame(summary)
        self.list_excluded = summary.loc[summary['% reliability'] < self.reliability_threshold]['regions_acr'].tolist()

        if self.verbose:
            print("List of regions below the threshold:\n")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(summary.loc[summary['% reliability'] < self.reliability_threshold])
            print("\nIf you want to skip unreliable regions during region averaging analysis, use the following list for the param \'regions_to_skip\':")
            print(self.list_excluded)

        self.filter = reliability_map < self.reliability_threshold

        flattened_map = u.reshape_volume_for_display(reliability_map)
        plt.figure(figsize=(12,16))
        plt.imshow(flattened_map/2 + 0.5, cmap='bwr', vmin=0, vmax=1)
        plt.savefig(os.path.join(output_folder, F"{name}_reliability_map.svg"))

        filter_display = deepcopy(self.filter).astype('int')
        if self.atlas_contours is not None:
            filter_display[self.atlas_contours > 0] = -1
        flattened_binary_map = u.reshape_volume_for_display(filter_display)
        plt.figure(figsize=(12,16))
        plt.imshow(-1*flattened_binary_map, cmap='binary')
        plt.savefig(os.path.join(output_folder, F"{name}_binary_reliability_map.svg"))

        if plot:
            plt.show()

        return(self.filter)


    def remove_unreliable(self,
        data: np.ndarray
        ) -> np.ndarray:

        """
        Convenience method for setting to nan voxels whose reliability is below the threshold. Must be called after process.

        Parameters
        ----------
        data : ndarray
            Data to be processed (4D).

        Returns
        -------
        data : ndarray
            Same as input data but with nan values where reliability is below the threshold.

        """

        assert hasattr(self, "filter"), "Please call the \'process\' method before calling the \'filter_data\' function."

        data[self.filter] = np.nan
        return(data)
