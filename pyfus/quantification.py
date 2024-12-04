import numpy as np
import pyfus.utils as u
import pandas as pd
import matplotlib.pyplot as plt
import typing
import seaborn as sns
import plotly.graph_objects as go
import pyfus.global_variables as gv
from scipy.integrate import simps
from scipy.signal import medfilt
from typing import Dict, Union, Tuple
from collections.abc import Iterable
from matplotlib.colors import LinearSegmentedColormap, to_hex


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




class SpatialTransitionAnalysis:

    """
    Object for spatial transition analysis, ie investigating how voxels behave in terms of
    clusters changes between conditions, at different spatial scales. Also works with the Z-maps.

    Parameters
    ----------
    maps : dict of array
        Dictionary containing 3D volumes with int-like values.
    mode : str
        Either 'multiregion', 'structure', 'hemisphere'. Determines which voxels are included in the process:
            - hemisphere: all voxels in a given or both hemispheres
            - structure: all voxels belonging to a list of structures
            - multiregion: all voxels belonging to a list of regions
    mode_params : tuple | list | str
        Parameters associated with the selected mode:
            - if 'hemisphere': 'L' or 'R' respectively for left and right hemispheres, or 'LR' for both.
            - if 'structure': a tuple or list of tuples with format (<structure>, <hemisphere>).
            - if 'multiregion': a tuple or list of tuples with format (<region acronym>, <hemisphere>).
    atlas_path : str
        Path to the file containing the atlas volume.
    regions_info_path : str
        Text file listing all the regions included in the atlas.
    n_clusters : int
        Number of clusters present in the maps.
    cmap : None | str | list
        Either a list of RGBA values or the name of a colormap available in pyplot.
    """

    def __init__(self,
        maps,
        mode,
        mode_params,
        atlas_path,
        regions_info_path,
        n_clusters,
        cmap = None
        ):

        if mode == 'multiregion' or mode == 'structure':
            assert type(mode_params) == tuple or type(mode_params) == list, "For mode 'multiregion', please input either the region acronym or a list of region acronyms."
        elif mode == 'hemisphere':
            assert type(mode_params) == str, "For mode 'hemisphere', please input either the hemisphere to be processed ('L', 'R', 'LR')."
        else:
            raise NameError("Incorrect choice of mode. Please choose between region and cluster.")

        self.maps, self.mode, self.mode_params = maps, mode, mode_params
        self.atlas = u.load_atlas(atlas_path)
        self.regions_nb, self.regions_acr, self.groups_acr = u.extract_info_from_region_file(regions_info_path)

        self.n_clusters = n_clusters
        self.cmap = cmap


    def get_cluster_transitions(self,
        cluster_maps,#: List[Tuple[str, str]],
        cluster_switches,#: List[Tuple[int, int]],
        mask_map = True,
        show = True
        ):

        """
        Method for highlighting the voxels following a specific transition pattern. For example, voxels switching from
        cluster 1 in condition A to cluster 2 in condition B and from cluster 1 in condition C to cluster 3 in condition D.

        Parameters
        ----------
        cluster_maps : list of tuples
            List of tuples containing the pairs of cluster maps, for example (condition A, condition B).
        cluster_switches : list of tuples
            List of tuples containing the associated cluster transitions, for example (1, 2). Must be the same length as cluster_maps.
        mask_map : bool
            Whether or not to mask voxels outside of the area defined by self.mode.
        show : bool
            Whether or not to display the resulting transitions map.

        Returns
        -------
        map_diff : ndarray
            The resulting transition map, with 1s indicating the voxels following the switching pattern.
        """

        assert len(cluster_maps) == len(cluster_switches), "'cluster_maps' and 'cluster_switches' variables must have the same length."

        title = ""
        map_diff = np.ones_like(self.maps[next(iter(self.maps))])

        for (cluster_map_from, cluster_map_to), (from_clusters, to_clusters) in zip(cluster_maps, cluster_switches):

            title += F"FROM {cluster_map_from}, {from_clusters}    TO    {cluster_map_to}, {to_clusters}\n"

            cluster_map_from = self.maps[cluster_map_from]
            cluster_map_to = self.maps[cluster_map_to]

            if type(from_clusters) == int:
                from_clusters = [from_clusters]
            if type(to_clusters) == int:
                to_clusters = [to_clusters]

            map_from = np.zeros(cluster_map_from.shape)
            for fc in from_clusters:
                map_from = np.logical_or(map_from, (cluster_map_from == fc))

            map_to = np.zeros(cluster_map_to.shape)
            for tc in to_clusters:
                map_to = np.logical_or(map_to, (cluster_map_to == tc))

            map_diff = map_diff * map_from * map_to

        if mask_map:
            mask = self._get_mask()
            map_diff *= mask

        map_bckg = (self.maps[next(iter(self.maps))] != 0)
        map_diff_flat = u.reshape_volume_for_display(map_bckg.astype('int') + map_diff.astype('int'))
        map_diff_flat[map_diff_flat == -1] = 1

        if show == True:
            fig, ax = plt.subplots()
            cmap = LinearSegmentedColormap.from_list('cmap_transition', [[1.]*3, [0.]*3, [1.0,0.,0.]])
            plt.title(title)
            plt.imshow(map_diff_flat, cmap=cmap)#, cmap=np.array([[1.,1.,1.], [.5,.5,.5], [0.,0.,0.]]))
            atlas_flat = u.reshape_volume_for_display(self.atlas)
            ax.format_coord = u.Formatter(atlas_flat, self.regions_acr, self.regions_nb)
            plt.show()

        return(map_diff)


    def merge_transition_maps(self,
        maps,
        legends,
        cmap=None
        ):

        """
        Function for merging multiple transition maps and displaying the outcome. Each map will have a separate color.

        Parameters
        ----------
        maps : list of array
            A list of maps, where each map is an output of the function 'get_cluster_transitions'.
        legends: list of string
            The label associated with each map.
        cmap: None | list
            Either None or a list of RGB values. If None, a default colormap will be used.
        """

        if cmap == None:
            cmap = plt.get_cmap('tab10')
            colors = [[1.]*3, [0.]*3] + [cmap(i) for i in range(len(maps))]
        else:
            colors = [[1.]*3, [0.]*3] + cmap

        res = maps[0]
        for i in range(1, len(maps)):
            res[maps[i] > 1] = 2+i
        res = u.reshape_volume_for_display(res)

        fig, ax = plt.subplots()
        cmap = LinearSegmentedColormap.from_list('cmap_transition', colors)
        plt.title("Merged maps")
        plt.legend()
        plt.imshow(res, cmap=cmap)#, cmap=np.array([[1.,1.,1.], [.5,.5,.5], [0.,0.,0.]]))
        atlas_flat = u.reshape_volume_for_display(self.atlas)
        ax.format_coord = u.Formatter(atlas_flat, self.regions_acr, self.regions_nb)
        plt.show()


    def get_quantif_regions_from_transitions(self,
        transition_map,
        threshold,
        plot=True,
        title="",
        prop='region'
        ):

        """
        Method for getting proportions of voxels in anatomical regions out of a transition map.

        Parameters
        ----------
        transition_map : ndarray
            Transition map as output by the method 'get_cluster_transitions' .
        threshold: float
            Threshold on the proportion (either with normalization 'region' or 'total', see 'prop') to determine which regions are included. Should be between 0 and 1.
        plot: bool
            Whether or not to plot the quantification as a bar plot. If False, quantifications will be displayed in the terminal. First and second values
            corresponds to the 'region' and 'total' normalizations, respectively. See argument 'prop' for further detail.
        title: string
            The title of the plot.
        prop: str
            Either 'region' or 'total'. If 'region', the number of voxels in the region will be normalized with respect to the size of the region.
            If 'total', the total number of voxels with value 1 in the transition map will be used for normalization.
        """

        keys, vals, structs = [], [], []
        n_map = np.sum(transition_map)
        assert prop in ['region', 'total'], "'prop' variable should be either 'region' or 'total'"

        for r_acr, g_acr in zip(self.regions_acr, self.groups_acr):

            for hemi in ['L', 'R']:

                mask = u.get_atlas_mask_from_regions(self.atlas, self.regions_nb, self.regions_acr, [(r_acr, hemi)])
                n_in, n_tot = np.sum(transition_map[mask]), np.sum(mask)

                if ((n_in / n_tot) > threshold) or ((n_in / n_map) > threshold):

                    if not plot:
                        print(F"{r_acr} {hemi}:  {round(n_in / n_tot *100, 2)}  {round(n_in / n_map *100, 2)}")

                    else:
                        keys.append(F"{r_acr}_{hemi}")
                        structs.append(g_acr)
                        if prop == 'region':
                            vals.append(round(n_in / n_tot *100, 2))
                        else:
                            vals.append(round(n_in / n_map *100, 2))

        if plot:
            data = pd.DataFrame({"Regions": keys, "Proportion (%)": vals, 'Structure': structs})
            g = sns.catplot(data=data, x='Regions', y='Proportion (%)', kind='bar', hue='Structure', dodge=False)
            g.set_xticklabels(rotation=45)
            plt.title(F"{title} {prop}")
            plt.show()


    def _get_mask(self):

        """
        Utility method for getting the mask associated with the selected mode.

        Returns
        ----------
        mask : ndarray
        """

        if self.mode == 'multiregion':
            mask = u.get_atlas_mask_from_regions(self.atlas, self.regions_nb, self.regions_acr, self.mode_params)

        elif self.mode == 'structure':
            acr_list = []
            for elt in self.mode_params:
                acr_list += u.get_regions_from_groups(elt[0], elt[1], self.regions_acr, self.groups_acr)
            mask = u.get_atlas_mask_from_regions(self.atlas, self.regions_nb, self.regions_acr, acr_list)

        elif self.mode == 'hemisphere':
            acr_list = u.get_regions_from_hemisphere(self.mode_params, self.regions_acr, self.groups_acr)
            mask = u.get_atlas_mask_from_regions(self.atlas, self.regions_nb, self.regions_acr, acr_list)

        else:
            raise NotImplementedError

        return(mask)


    def _get_mask_from_smallest_map(self):

        """
        Utility method for getting the mask providing the common set of voxels between maps.

        Returns
        ----------
        mask : ndarray
        """

        count = np.inf
        map_mask = np.ones_like(self.maps[next(iter(self.maps))])

        for i, name in enumerate(self.maps):
            map_mask *= (self.maps[name] != 0)

        return(map_mask.astype('bool'))


    def _get_sources_and_targets(self):

        """
        Method for getting the sources and targets for the Sankey plot. Each combination (cluster number, map) has a unique identifier.

        Returns
        ----------
        sources : ndarray
            An array of indices giving the starting points of the flow in the Sankey plot.
        targets : ndarray
            An array of indices giving the end points of the flow in the Sankey plot.
        """

        sources, targets = [], []
        mask = self._get_mask()
        map_mask = self._get_mask_from_smallest_map()

        names = [name for name in self.maps]
        indices_i, indices_j = [i for i in range(len(names)-1)], [j for j in range(1, len(names))]

        for i, j in zip(indices_i, indices_j):
            src = self.maps[names[i]][mask*map_mask]
            tgt = self.maps[names[j]][mask*map_mask]
            sources.append(src + i*self.n_clusters-1)
            targets.append(tgt + j*self.n_clusters-1)

        return(np.hstack(sources), np.hstack(targets))


    def _reduce_sources_targets_values(self,
        sources,
        targets
        ):

        """
        Method for grouping the same pairs (source, target) as a single flow in the Sankey plot.

        Parameters
        ----------
        sources : ndarray
            Source array as output by method _get_sources_and_targets.
        targets : ndarray
            Target array as output by method _get_sources_and_targets.

        Returns
        ----------
        reduced_sources : ndarray
            An array of indices giving the starting points of the reduced flow in the Sankey plot.
        reduced_targets : ndarray
            An array of indices giving the end points of the reduced flow in the Sankey plot.
        reduced_values : ndarray
            The values associated with the reduced flow.
        """

        reduced_sources, reduced_targets, reduced_values = [], [], []
        names = [name for name in self.maps]

        for src in np.unique(sources):
            for tgt in np.unique(targets):
                reduced_sources.append(src)
                reduced_targets.append(tgt)
                reduced_values.append(np.sum((sources == src) * (targets == tgt)))

        return(reduced_sources, reduced_targets, reduced_values)


    def _get_labels_and_colors(self):

        """
        Method for getting the labels <map name>_<cluster ID> and associated colors for the Sankey plot.

        Returns
        ----------
        labels : list of str
            The labels for the Sankey plot.
        colors : list
            List of colors. Can be strings or hexadecimal values.
        """

        labels = [F"{name}_{cl+1}" for name in self.maps for cl in range(self.n_clusters)]

        if self.cmap is not None:
            colors = [to_hex(self.cmap[i]) for _ in range(len(self.maps)) for i in range(self.n_clusters)]
        else:
            colors = 'gray'

        return(labels, colors)


    def sankey_from_maps(self,
        reduce=True
        ):

        """
        Method for generating a Sankey plot allowing for the visualization of cluster transitions across cluster maps.

        Parameters
        ----------
        reduce : bool
            If True, reduces the flows in the Sankey plot, ie voxels switching from same source to same target will be grouped as a single flow.
        """

        sources, targets = self._get_sources_and_targets()
        values = [1] * len(sources)
        if reduce:
            sources, targets, values = self._reduce_sources_targets_values(sources, targets)

        labels, colors = self._get_labels_and_colors()

        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = labels,
              color = colors
            ),
            link = dict(
              source = sources,
              target = targets,
              value = values
          ))])

        fig.update_layout(title_text=F"Cluster transitions in {self.mode_params} (mode {self.mode})", font_size=10)
        fig.show()
