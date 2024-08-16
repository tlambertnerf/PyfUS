import numpy as np
import pyfus.utils as u
import pandas as pd
import matplotlib.pyplot as plt
import typing
import seaborn as sns
from scipy.integrate import simps
from scipy.signal import medfilt
from typing import Dict, Union, Tuple
from collections.abc import Iterable


class SpatialQuantification:

    """
    Object for performing basic spatial quantification from the output of the correlation and single voxel clustering analyses based on the atlas. Assumes that the data are registered.
    Example of input include the Z-maps and the cluster maps. Each unique value in the input will be considered as the identifier of a set.

    Parameters
    ----------
    data_dict : dict of ndarray
        Dictionary containing 3D volumes with int-like values.
    atlas_path : str
        Path to the atlas volume.
    regions_info_path : str
        Path to the regions info file.
    """

    def __init__(self,
        data_dict: Dict[str, np.ndarray],
        atlas_path: str,
        regions_info_path: str
        ):

        self.atlas = u.load_atlas(atlas_path)
        self.data_dict = data_dict
        self.regions_nb, self.regions_acr, self.groups_acr = u.extract_info_from_region_file(regions_info_path)

        unique_vals = []
        for name in self.data_dict:
            unique_vals += list(np.unique(self.data_dict[name]))
        self.set_ids = list(set(unique_vals))

        self.names = [name for name in self.data_dict]
        print(F"names available for quantification: {self.names}\n")

        self.quantification_per_region()


    def quantification_per_region(self):

        """
        Fill and store in the object two dataframes containing for each region all the available info and quantification for each set. Called during initialization.
        """

        r_nb, r_acr, g_acr, d_name, hemi, v_id, vox_in, vox_tot = [], [], [], [], [], [], [], []

        for h, s in zip(['L', 'R'], [-1, 1]):

            for r in self.regions_nb:

                atlas_mask = (self.atlas == r*s)
                vox_tot_r = np.sum(atlas_mask)
                acr = self.regions_acr[self.regions_nb == r][0]
                g = self.groups_acr[self.regions_nb == r][0]

                for name in self.data_dict:

                    data = self.data_dict[name][atlas_mask]

                    for val in self.set_ids:

                        r_nb.append(r)
                        r_acr.append(acr)
                        g_acr.append(g)
                        hemi.append(h)
                        d_name.append(name)
                        v_id.append(val)
                        vox_in.append(np.sum(data == val))
                        vox_tot.append(vox_tot_r)

        self.quantif_df = pd.DataFrame({'Region Nb': r_nb, 'Region Acr': r_acr, 'Group Acr': g_acr, 'Data Name': d_name, 'Hemisphere': hemi, 'Set ID': v_id, 'Voxels In': vox_in, 'Voxels Total': vox_tot, 'Proportion': np.array(vox_in)/np.array(vox_tot)}, index=[i for i in range(len(r_acr))])


    def print_quantification_region(self,
        region_acr: str,
        hemisphere: str,
        data_name: Union[str, None] = None,
        set_id: Union[int, None] = None
        ):

        """
        Method for printing the quantification associated with a specific region.

        Parameters
        ----------
        region_acr : str
            Acronym of the region to be printed.
        hemisphere : str
            Either 'L' or 'R', for left and right hemispheres respectively.
        data_name : str | None
            Either the name of the specific data to be displayed. If None, all available data will be printed.
        set_id : int | None
            Either the value defining a specific set, e.g. 1 for cluster 1. If None, all the sets will be printed.
        """

        assert hasattr(self, "quantif_df"), "Please call the \'quantification_per_region\' method before calling the \'print_quantification\' function."
        assert hemisphere in ['L', 'R'], "Hemisphere must be L or R."

        quantif = self.quantif_df.loc[(self.quantif_df['Region Acr'] == region_acr) & (self.quantif_df['Hemisphere'] == hemisphere)]

        if data_name is not None:
            assert data_name in self.names, "Requested data name not found..."
            quantif = quantif.loc[quantif['Data Name'] == data_name]

        if set_id is not None:
            quantif = quantif.loc[quantif['Set ID'] == set_id]

        print(quantif)


    def print_quantification_structure(self,
        acr: str,
        hemisphere: str,
        data_name: Union[str, None] = None,
        ignore_non_selected: bool = False
        ):

        """
        Method for printing the quantification associated with a specific anatomical structure.

        Parameters
        ----------
        acr : str
            Acronym of the group to be printed.
        hemisphere : str
            Either 'L' or 'R', for left and right hemispheres respectively.
        data_name : str | None
            Either the name of the specific data to be displayed or None. If None, all available data will be printed.
        ignore_non_selected : bool
            To set to True if used with clustering, to remove voxels that were not included in the clustering process if a subset of regions was selected.
        """

        assert hasattr(self, "quantif_df"), "Please call the \'quantification_per_region\' method before calling the \'print_quantification\' function."
        assert hemisphere in ['L', 'R'], "Hemisphere must be L or R."

        quantif = self.quantif_df.loc[(self.quantif_df['Group Acr'] == acr) & (self.quantif_df['Hemisphere'] == hemisphere)]

        if data_name is not None:
            assert data_name in self.names, "Requested data name not found..."
            quantif = quantif.loc[quantif['Data Name'] == data_name]

        if ignore_non_selected:
            # to get the number of voxels non selected but belonging to the group to subtract them
            tot_0 = quantif.loc[quantif['Set ID'] == 0].groupby(['Group Acr']).sum()['Voxels In'].to_numpy()[0]

        res = {'Set ID':[], 'Voxels In':[], 'Voxels Total':[], 'Proportion':[]}

        for i in self.set_ids:

            q = quantif.loc[quantif['Set ID'] == i].groupby(['Group Acr']).sum()
            in_, tot_ = q['Voxels In'].to_numpy()[0], q['Voxels Total'].to_numpy()[0]

            if ignore_non_selected and i != 0:
                tot_ -= tot_0
            elif ignore_non_selected and i == 0:
                continue
            else:
                pass

            res['Set ID'].append(i)
            res['Voxels In'].append(in_)
            res['Voxels Total'].append(tot_)
            res['Proportion'].append(in_/tot_)

            #print(F"SET {i} :\nVoxels In: {in_}\nVoxels Total: {tot_}\nProportion: {in_/tot_}\n")

        print(pd.DataFrame(res))



    def print_quantification_per_set(self,
        data_name: Union[str, None] = None,
        set_id: Union[int, None] = None,
        normalization: str = 'brainwide'
        ):

        """
        Method for printing the proportion of voxels per set in each anatomical structure.

        Parameters
        ----------
        data_name : str | None
            Either the name of the specific data to be displayed or None. If None, all available data will be printed.
        set_id : int | None
            Either the id of the set to be displayed or None. If None, quantification for all sets will be printed.
        normalization: str
            Either 'hemisphere' or 'brainwide'
        """

        assert hasattr(self, "quantif_df"), "Please call the \'quantification_per_region\' method before calling the \'print_quantification\' function."

        if data_name is not None:
            assert data_name in self.names, "Requested data name not found..."
            quantif = self.quantif_df.loc[self.quantif_df['Data Name'] == data_name]
        else:
            quantif = self.quantif_df

        set_id = self.set_ids if set_id is None else set_id
        all_groups_acr = np.unique(self.groups_acr)

        for i in set_id:

            tot_in_i = quantif.loc[(quantif['Set ID'] == i)]['Voxels In'].sum()

            for acr in all_groups_acr:

                for hemisphere in ['L', 'R']:

                    q_acr_h = quantif.loc[(quantif['Set ID'] == i) & (quantif['Group Acr'] == acr) & (quantif['Hemisphere'] == hemisphere)]

                    if normalization == 'brainwide':
                        print(i, acr, hemisphere, q_acr_h['Voxels In'].sum() / tot_in_i)
                    else: # multitply by 2 because both hemispheres have the same size
                        tot_in_i_h = quantif.loc[(quantif['Set ID'] == i) & (quantif['Hemisphere'] == hemisphere)]['Voxels In'].sum()
                        print(i, acr, hemisphere, q_acr_h['Voxels In'].sum() / tot_in_i_h)



    def print_quantification_per_hemisphere(self, data_name: Union[str, None] = None):

        """
        Method for printing the proportion of voxels per set in each hemisphere.

        Parameters
        ----------
        data_name : str | None
            Either the name of the specific data to be displayed or None. If None, all available data will be printed.
        """

        assert hasattr(self, "quantif_df"), "Please call the \'quantification_per_region\' method before calling the \'print_quantification\' function."

        if data_name is not None:
            assert data_name in self.names, "Requested data name not found..."
            quantif = self.quantif_df.loc[self.quantif_df['Data Name'] == data_name]
        else:
            quantif = self.quantif_df

        for hemisphere in ['L', 'R']:

            print(F"HEMISPHERE: {hemisphere}")

            for i in self.set_ids:

                quantif_h = quantif.loc[(quantif['Set ID'] == i) & (quantif['Hemisphere'] == hemisphere)]
                prop = quantif_h['Voxels In'].sum() / quantif_h['Voxels Total'].sum()
                print(F"{i} {prop}")
                #quantif_h = quantif.loc[(quantif['Hemisphere'] == hemisphere)].groupby(['Set ID'])['Voxels Total'].sum()

            print()





def compute_amplitude(
    signal: np.array,
    time_stim : Union[None, Iterable]
    ):

    """
    Utility function for computing the amplitude of an input signal.

    Parameters
    ----------
    signal : array
        1D array, temporal signal.
    time_stim : None | iterable

    Returns
    -------
    amp : float
        Computed amplitude of the input signal.
    """

    if time_stim is None:
        time_stim = range(len(signal))

    amp = np.max(signal)

    return(amp)


def compute_auc(
    signal: np.array,
    sr: float = 5
    ):

    """
    Utility function for computing the area under the curve (AUC) of an input signal.

    Parameters
    ----------
    signal : array
        1D array, temporal signal.
    sr : float
        Temporal sampling rate of the signal.

    Returns
    -------
    auc : float
        Computed AUC of the input signal.
    """

    auc = simps(signal, dx=sr, axis=0)

    return(auc)


def compute_time_to_peak(
    signal: np.array,
    sr: float = 5
    ):

    """
    Utility function for computing the time to peak (TTP) of an input signal.

    Parameters
    ----------
    signal : array
        1D array, temporal signal.
    sr : float
        Temporal sampling rate of the signal.

    Returns
    -------
    ttp : float
        Computed TTP of the input signal.
    """


    data_t0, data_t1 = np.zeros(len(signal)+1), np.zeros(len(signal)+1)
    data_t1[:-1], data_t0[1:] = signal, signal
    derivative = (data_t1 - data_t0)
    pos, neg = derivative > 0, derivative < 0
    inflexion_points = np.argwhere(np.logical_or(pos[:-1]*neg[1:], derivative[:-1]==0)).ravel()
    arr_inds = signal[inflexion_points].argsort()
    peak = inflexion_points[arr_inds[::-1]][0]

    return(peak/sr)


def compute_full_width_half_max(
    signal: np.array,
    sr: float = 5
    ):

    """
    Utility function for computing the full width at half maximum (FWHM) and time to half maximum (THM) of an input signal.

    Parameters
    ----------
    signal : array
        1D array, temporal signal.
    sr : float
        Temporal sampling rate of the signal.

    Returns
    -------
    fwhm : float
        Computed FWHM of the input signal.
    thm : float
        Computed THM of the input signal.
    """

    signal = medfilt(signal)
    half_amp = np.max(signal) / 2
    above_half_amp = signal > half_amp
    m = np.r_[False, above_half_amp, False]
    idx = np.flatnonzero(m[:-1] != m[1:])
    fwhm = (idx[1::2]-idx[::2]).max()
    thm = np.where(above_half_amp == 1)

    return(fwhm/sr, thm[0][0]/sr)



class TemporalQuantification:

    """
    Object for performing basic temporal quantification from the output of the region averaging and single voxel clustering analyses.
    Example of input include the temporal traces of individual regions of the atlas or clusters.

    Parameters
    ----------
    data_dict : dict of array
        Dictionary of 1D temporal signals.
    sampling_rate : float
        Temporal sampling rate of the signals.
    """

    def __init__(self,
        data_dict: Dict[str, np.array],
        sampling_rate: float
        ):

        self.data_dict = data_dict
        self.sampling_rate = sampling_rate

        self.compute_metrics()


    def compute_metrics(self):

        """
        Method for computing the metrics from each signal and storing them in a dataframe. Called during initialization.
        Element refers to a mouse, session, etc. Key refers to a region, a cluster ID, etc.
        """

        res = {'Element': [], 'Key': [],  'Metric':[], 'Value':[]}

        for element in self.data_dict:

            for key in self.data_dict[element]:

                signal = self.data_dict[element][key].astype('float32')

                amp = compute_amplitude(signal, None)
                auc = compute_auc(signal, sr=self.sampling_rate)
                ttp = compute_time_to_peak(signal, sr=self.sampling_rate)
                fwhm, time_hm =  compute_full_width_half_max(signal, sr=self.sampling_rate)

                res['Element'] += [element]*5
                res['Key'] += [key]*5
                res['Metric'] += ['Peak amplitude', 'AUC', 'Time to peak (s)', 'FWHM (s)', 'Time to half max (s)']
                res['Value'] += [amp, auc, ttp, fwhm, time_hm]

        self.df_metrics = pd.DataFrame(res)


    def plot_metric(self,
        metric: str,
        scale: Union[Tuple[float, float], None] = None,
        colors=None
        ):

        """
        Convenience method for plotting a given metric.

        Parameters
        ----------
        metric : str
            Name of the metric to be plotted. Available metrics are 'Peak amplitude', 'AUC', 'Time to peak (s)', 'FWHM (s)', 'Time to half max (s)'
        scale : tuple of float | None
            Min and max scales for the display (arguments vmin / vmax from plt.imshow). If None, auto scale will be used.
        """

        plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0

        df_metric = self.df_metrics.loc[self.df_metrics['Metric'] == metric]

        g = sns.catplot(kind='violin', data=df_metric, x='Key', y='Value', inner='quartile', palette=colors, cut=1.5)
        sns.swarmplot(data=df_metric, x='Key', y='Value', c='black', ax=g.ax)
        if scale:
            plt.ylim(scale[0], scale[1])
        else:
            plt.ylim(np.round(df_metric['Value'].min()*0.8, decimals=2), np.round(df_metric['Value'].max()*1.2, decimals=2))
        plt.title(metric)
        plt.show()


    def print_metric(self, metric):

        """
        Convenience method for printing a given metric.

        Parameters
        ----------
        metric : str
            Name of the metric to be printed.
        """

        df_metric = self.df_metrics.loc[self.df_metrics['Metric'] == metric]
        print(df_metric)
