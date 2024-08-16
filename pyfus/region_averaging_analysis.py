import numpy as np
import os, time, pickle, typing, warnings
import matplotlib.pyplot as plt
import pandas as pd
import pyfus.utils as u
import seaborn as sns
from sklearn.decomposition import KernelPCA, PCA
from itertools import compress
from matplotlib.patches import Patch
from typing import Union, List, Dict, Tuple


# TO COMMENT WHEN IN DEVELOPMENT
warnings.filterwarnings('ignore')


class RegionAveragingAnalysis:

    """
    Object handling the region averaging and associated displays.

    Parameters
    ----------
    data_paths : list of str
        List of paths to the folders in which are stored the data after preprocessing. See doc for more info on folder structure and how to automatically select data.
    atlas_path : str
        Path to the atlas volume file.
    regions_info_path : str
        Text file listing all the regions included in the atlas.
    remove_empty : bool
        If true, removes all the anatomical parts labelled as 'empty', eg fiber tracts.
    to_zscore : bool
        If true, traces will be displayed as Z-scores. Requires to set the baseline argument as well.
    baseline : iterable
        Baseline of the recording. Used for computing the Z-score.
    """

    def __init__(self,
        data_paths: List[str],
        atlas_path: str,
        regions_info_path: str,
        remove_empty: bool = True,
        regions_to_exclude: list = [],
        to_zscore = None,
        baseline = None
        ) -> None:

        atlas = u.load_atlas(atlas_path)

        self.data, self.acr, self.groups, self.hemispheres = {}, {}, {}, {}
        self.regions_nb, self.regions_acr, self.groups_acr = u.extract_info_from_region_file(regions_info_path)
        self.structures = sorted(set(list(self.groups_acr)))

        if remove_empty:
            self.structures.remove('empty')

        self.names = []

        for data_path in data_paths:

            name = u.name_from_path(data_path)
            data = np.load(data_path)
            # /!\ /!\ /!\ /!\ dirty PATCH TO CHANGE IN DL baseline -1
            data[data == -1] = 0
            d, a, g, h = u.projection(data, atlas, self.regions_nb, self.regions_acr, self.groups_acr, regions_to_exclude=regions_to_exclude, start_at=4)

            if to_zscore:
                d = u.convert_to_zscore(d, baseline)

            if remove_empty:
                filt = (g != 'empty')
                d, a, g, h = d[filt,:], a[filt], g[filt], h[filt]

            self.data[name], self.acr[name], self.groups[name], self.hemispheres[name] = d, a, g, h
            self.names.append(name)

            #print(F"Elapsed time: {time.time() - tmstp}")

        print("Projections done!\nElements available for display:\n{}".format('\n'.join(str(name) for name in self.names)))
        self.data_df, self.data_info = u.convert_projections_to_df(self.data, self.acr, self.hemispheres)


    def plot_barcode(self,
        names: Union[None, List[str]] = None,
        separate_plots: bool = False,
        scale: Tuple[float, float] = (-0.4, 0.4),
        show: bool = False
        ) -> None:

        """
        Method for plotting the barcode view from the averaged data.

        Parameters
        ----------
        names : None | list of str
            If None, all the data will be displayed. If list of strings, only the names specified in it will be displayed.
        separate_plots : bool
            Whether or not to make separate windows for each element to be plotted, eg sessions.
        scale : tuple of float
            Min and max scales for the display (arguments vmin / vmax from plt.imshow).
        show : bool
            If True, the plot will be displayed at the end of the function's execution. Otherwise, a call to plt.show() must be used to display the figures.
        """

        cm = plt.get_cmap('tab20')
        self.cmap = [cm(i/len(self.structures)) for i in range(len(self.structures))]

        names = self.names if names is None else set(names)

        ncols = 2 if separate_plots else len(names)*2
        if not separate_plots:
            fig, axes = plt.subplots(nrows=1, ncols=ncols, squeeze=False)
        acr_all, nb_all = np.hstack((self.regions_acr, self.regions_acr)), np.hstack((self.regions_nb, self.regions_nb))

        for i, name in enumerate(names):

            # separate hemispheres based on the value in self.hemispheres. Positive for right and negative for left
            left_idx, right_idx = self.hemispheres[name] < 0, self.hemispheres[name] > 0
            data_L, data_R = self.data[name][left_idx, :], self.data[name][right_idx, :]
            acr_L, acr_R = self.acr[name][left_idx], self.acr[name][right_idx]
            group_L, group_R = self.groups[name][left_idx], self.groups[name][right_idx]

            # remove NANs
            idx_nan = ~np.isnan(data_L).any(axis=1)
            data_L, acr_L, group_L = data_L[idx_nan,:], acr_L[idx_nan], group_L[idx_nan]
            idx_nan = ~np.isnan(data_R).any(axis=1)
            data_R, acr_R, group_R = data_R[idx_nan,:], acr_R[idx_nan], group_R[idx_nan]
            #print(data_L.shape, data_R.shape)
            #print("\n".join(["{} {}".format(a, g) for a, g, in zip(acr_L, group_L)]))
            #print()
            #print("\n".join(["{} {}".format(a, g) for a, g, in zip(acr_R, group_R)]))

            if separate_plots:
                fig, axes = plt.subplots(nrows=1, ncols=ncols, squeeze=False)

            # custom legend to associated colors with major anatomical groups
            legend_elements = [Patch(facecolor=color, edgecolor=color, label=group) for group, color in zip(self.structures, self.cmap)]

            for j, (side, data, acr, group) in enumerate(zip(['left', 'right'], [data_L, data_R], [acr_L, acr_R], [group_L, group_R])):

                k = 2*i+j if not separate_plots else j
                #im = axes[0,k].imshow(data, vmin=-np.max(data_L), vmax=np.max(data_L), cmap='jet')
                im = axes[0,k].imshow(data, vmin=scale[0], vmax=scale[1], cmap='jet')
                axes[0,k].set_aspect('equal', adjustable='datalim')
                axes[0,k].set_yticklabels(acr, fontdict={'fontsize':8})
                axes[0,k].set_yticks([i for i in range(len(acr))])
                if i == 0 and j == 0:
                    axes[0,k].legend(handles=legend_elements)
                axes[0,k].format_coord = FormatterBarcode(acr)

                # to color each label depending on its anatomical group
                for struct, color in zip(self.structures, self.cmap):
                    plt.setp(compress(axes[0,k].get_yticklabels(), group==struct), backgroundcolor=color)

                axes[0,k].set_xticks([i for i in range(6, data.shape[1]+1, 10)], [i+4 for i in range(6, data.shape[1]+1, 10)])
                axes[0,k].set_title(F"{name} {side}")

        fig.colorbar(im)

        if show:
            plt.show()


    def plot_region(self,
        region_acr: List[Tuple[str, str]],
        names: Union[None, List[str]] = None,
        separate_plots: bool = True,
        display_individual_elements: bool = True,
        scale: Union[None, Tuple[float, float]] = None,
        show: bool = False
        ) -> None:

        """
        Method for plotting the barcode view from the averaged data.

        Parameters
        ----------
        region_acr : str | list of str
            Acronym(s) of the region to be plotted.
        hemisphere : str | None
            If None, both hemispheres are plotted. Otherwise, 'L' or 'R' will only plot the selected hemisphere.
        names : None | list of str
            If None, all the data will be displayed. If list of strings, only the names specified in it will be displayed.
        separate_plots : bool
            If True, the plots will be displayed in separate windows. If not, all regions will be plotted together.
        display_individual_elements : bool
            Whether or not to display individual elements defined in 'self.names' (eg animals, sessions) for the region. If not, 95% CI will be displayed instead.
            /!\ For the sake of readability, this option is not available if separate_plots is set to False.
        scale : tuple of float | None
            Min and max scales for the display (arguments vmin / vmax from plt.imshow). If None, auto scale will be used.
        show : bool
            If True, the plot will be displayed at the end of the function's execution. Otherwise, a call to plt.show() must be used to display the figures.
        """

        if not separate_plots:
            df = pd.DataFrame(columns=self.data_df.columns)
        else:
            if len(region_acr) < 4:
                nrows, ncols = 1, len(region_acr)
            else:
                nrows, ncols = int(len(region_acr) / 4 - 1e-5) + 1, 4
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
            axes = axes.reshape(-1)

        names = self.names if names is None else set(names)
        df_source = self.data_df.loc[self.data_df['Name'].isin(names)]

        for i, (acr, hemisphere) in enumerate(region_acr):

            assert (acr in list(self.regions_acr)), 'The acronym you provided does not exist...'
            #df_region = self.data_df.loc[self.data_df['RegionAcr']==acr]
            df_region = df_source.loc[df_source['RegionAcr']==acr]

            if hemisphere != 'LR':
                assert hemisphere in ['L', 'R'], "Please choose an hemisphere between left 'L' or right 'R', or set to 'LR' to display both"
                df_region = df_region.loc[df_region['Hemisphere'] == hemisphere]

            if separate_plots:
                #plt.figure()
                if display_individual_elements:
                    sns.lineplot(x='Frame', y='Amplitude', style='Hemisphere', hue='Name', data=df_region, ax=axes[i])
                else:
                    sns.lineplot(x='Frame', y='Amplitude', hue='Hemisphere', data=df_region, ax=axes[i])

                if scale is not None:
                    plt.ylim(scale[0], scale[1])
                axes[i].set_title(acr)
                axes[i].set_xlim(0, self.data_info['maxFrame'])
                if i != 0:
                    axes[i].legend([],[], frameon=False)

            else:
                df = df.append(df_region)

        if not separate_plots:
            plt.figure()
            sns.lineplot(x='Frame', y='Amplitude', style='Hemisphere', hue='RegionAcr', data=df)
            plt.title(region_acr)
            if scale is not None:
                plt.ylim(scale[0], scale[1])
            plt.xlim(0, self.data_info['maxFrame'])

        if show:
            plt.show()


    def get_regions_traces(self,
        acronyms: Union[Tuple[str, str], List[Tuple[str, str]]],
        names: Union[str, List[str], None] = None
        ) -> Dict[str, Dict[str, np.array]]:

        """
        Method for getting traces of individual regions from the object.

        Parameters
        ----------
        acronyms : tuple | list of tuple
            The format of the tuple is (<region acronym>, <hemisphere>). Single tuple and list of tuples following this format are accepted.
        names : str | list of str | None
            Names of the elements to get the data from. If None, all elements will be considered. Check on your terminal the elements available for display or the variable self.name.

        Returns
        -------
        res : dict of dict of arrays
            Dictionary whose keys are the names of the elements, and values dictionaries with regions acronyms as keys and temporal traces as values.
        """

        if names == None:
            names = self.names
        elif type(names) == str:
            names = [names]
        else:
            pass

        if type(acronyms) == tuple:
            acronyms = [acronyms]

        res = {}

        for name in names:

            data_n = self.data[name]
            res[name] = {}

            for acr in acronyms:

                assert (acr[0] in list(self.regions_acr)), 'The acronym you provided does not exist...'
                h = 1 if acr[1] == 'R' else -1
                res[name][F"{acr[0]}_{acr[1]}"] = data_n[(self.acr[name] == acr[0]) * (self.hemispheres[name] == h)][0,:]

        return(res)



class FormatterBarcode(object):

    """
    Object for dynamically displaying the regions of the atlas under the cursor in the plots.

    Parameters
    ----------
    atlas : ndarray
        The 3D atlas volume.
    regions_acr : array
        Array containing the acronyms of the regions included in the atlas.
    regions_nb : array
        Array containing the regions' numbers / IDs included in the atlas.
    """

    def __init__(self, regions_acr):
        self.regions_acr = regions_acr

    def __call__(self, x, y):
        g = self.regions_acr[int(y)]
        return(F"{g}")
