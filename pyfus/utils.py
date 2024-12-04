"""
@author: Théo Lambert

This module regroups all the utility functions of generic use.
"""


import numpy as np
import pandas as pd
import sys, time, os, typing, pickle
import matplotlib.pyplot as plt
import pkg_resources
import pyfus.global_variables as gv
from sklearn.decomposition import FastICA
from typing import Union, List
from importlib.resources import files



def extract_info_from_region_file(
    regions_info_file: str
    ) -> (np.array, np.array, np.array):

    """
    Utility function for extracting region numbrs, acronyms, and anatomical groups from the region info file.

    Parameters
    ----------
    regions_info_file : str
        Path to the regions_info_file.

    Returns
    -------
    All output are ordered in the same way.
    regions_nb : array
        1D array containing the regions' numbers
    regions_acr : array
        Same but for the acronyms.
    groups_acr : array
        Same but for the anatomical group.
    """

    regions_info = pd.read_csv(regions_info_file, sep=" ", header=None, on_bad_lines='skip').to_numpy()
    regions_nb, regions_acr, groups_acr = regions_info[:,0], regions_info[:,1], regions_info[:,2]

    return(regions_nb, regions_acr, groups_acr)


def projection(
    data: np.ndarray,
    atlas: np.ndarray,
    regions_nb: np.array,
    regions_acr: np.array,
    groups_acr: np.array,
    regions_to_exclude: list = [],
    reduction: str = 'median',
    use_allen: bool = True,
    start_at: int = 4
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    """
    Convert a 4D input data (volume in time) into a 2D matrix in which each line corresponds to the reduced time trace associated with a given region of the atlas.

    Parameters
    ----------
    data : ndarray
        4D (volume in time) matrices containing the fus signal variations per voxel.
    atlas : ndarray
        3D volume of the same size as data (without temporal) where each voxel value correspond to its number in the Allen CCF v3 ontology.
    regions_nb : ndarray
        Array containing all the region numbers in the atlas used.
    regions_acr : ndarray
        Array containing all the region acronyms in the atlas used (same order as in regions_nb).
    groups_acr : ndarray
        Array containing all the anatomical groups acronyms associated with each region acronym (same order as in regions_nb).
    regions_to_exclude : list
        List of regions to be excluded during the projection. Default correspond to voxels only belonging to a major anatomical groups, eg boundaries between cortical areas.
    reduction : str
        Selection of the reduction method to average the data in each region, either 'mean' or 'median'.
    use_allen : bool
        Set to True if you use the Allen brain atlas, otherwise False. If True, some specific regions will be automatically removed.
    use_allen : bool
        Set to True if you use the Allen brain atlas, otherwise False. If True, some specific regions will be automatically removed.
    start_at : int
        To remove the first frames of the recording if required. Usually 4 first frames are removed since the acquisition processing pipelines is fully operational after these 4 frames.

    Returns
    -------
    All output are ordered the same way.
    proj
        Temporal traces resulting from the projection.
    acr
        Array containing the regions acronyms associated with the traces.
    groups
        Array containing the groups acronyms associated with the traces.
    hemi
        Array containing the hemisphere (-1 for left, 1 for right) associated with the traces.
    """

    assert reduction in ['median', 'mean'], "reduction param should be either median or mean"

    if use_allen == True:
        regions_to_exclude += [771, 354, 1097, 549, 313, 512, 703, 1089, 698] # big regions such as CB etc that only encomposses joints between smaller regions

    regions_indices = np.unique(atlas)
    proj, acr, groups, hemi = [], [], [], []

    for i, region in enumerate(regions_indices):

        if region != 0 and (abs(region) not in regions_to_exclude): # 0 is background

            if reduction == 'median':
                region_data = np.nanmedian(data[atlas == region, start_at:], 0)
            else:
                region_data = np.nanmean(data[atlas == region, start_at:], 0)

            region_acr = regions_acr[abs(region) == regions_nb][0]
            group_acr = groups_acr[abs(region) == regions_nb][0]
            h = -1 if region <= 0 else 1

            proj.append(region_data)
            acr.append(region_acr)
            groups.append(group_acr)
            hemi.append(h)

    # sorting per anatomical group and then by alphabetical order
    sorting = sorted(sorted(range(len(groups)), key=acr.__getitem__), key=groups.__getitem__)

    return(np.array(proj)[sorting], np.array(acr)[sorting], np.array(groups)[sorting], np.array(hemi)[sorting])


def single_voxels_in_region(
    data: np.array,
    atlas: np.array,
    regions_nb: np.array,
    regions_acr: np.array,
    region: int,
    show: bool = False
    ):

    """
    Utility function for diplaying or returning the temporal traces of all single voxels within a region.

    Parameters
    ----------
    data : ndarray
        4D volume (space and time) containing the time traces of each voxel.
    atlas : array
        3D volume of the atlas, where each value is a number indicating to which region a voxel belongs.
    regions_nb : array
        Array providing all the regions number.
    regions_acr : array
        Array providing all the regions acronyms. Same order as regions_nb.
    region : int
        The region to be displayed.
    show : bool
        If True, plot the single voxel temporal traces, else return them.

    Returns
    -------
    single_voxels : ndarray
        2D array whose first dim contains single voxels and second dim the temporal traces
    """

    region_idx = regions_nb[regions_acr == region]
    single_voxels = data[atlas == region, :]

    if show:
        for i in single_voxels.shape[0]:
            plt.plot(single_voxels[i,:])
        plt.show()

    return(single_voxels)


def convert_projections_to_df(
    projections: np.ndarray,
    regions_acr: np.array,
    hemispheres: np.array,
    start_at: int = 4
    ) -> (pd.DataFrame, dict):

    """
    Utility function for converting the projections from the 'projection' function into a pandas dataframe. It allows convenient and fancy display.

    Parameters
    ----------
    projections : ndarray
        First output of the 'projection' function, ie a 2D matrix with rows as regions and columns as timepoints.
    regions_acr : array
        Array providing all the regions acronyms.
    hemispheres : array
        Array providing the hemispheres (-1 for left, 1 for right)
    start_at : int
        If some early frames need to be removed because of noise. Standard value is 4.

    Returns
    -------
    df : dataframe
        A dataframe structure containing all the data from projections.
    info : dict
        A dictionary with useful information for setting the scaling displays.
    """

    df = pd.DataFrame(columns=['Frame', 'Amplitude', 'RegionAcr', 'Hemisphere', 'Name'])
    frames_all, amp_all, acr_all, hemi_all, names_all = [], [], [], [], []

    for name in projections:

        proj = projections[name]
        T = proj.shape[1]

        for i in range(proj.shape[0]):

            frames_all += [i+start_at for i in range(T)]
            amp_all += proj[i,:].tolist()
            acr_all += [regions_acr[name][i]]*T
            hemi = 'L' if hemispheres[name][i] <= 0 else 'R'
            hemi_all += [hemi]*T
            names_all += [name]*T

    df = pd.DataFrame({'Frame':frames_all, 'Amplitude':amp_all, 'RegionAcr':acr_all, 'Hemisphere':hemi_all, 'Name':names_all})
    info = {'minFrame':0, 'maxFrame':T, 'minAmp':np.min(amp_all), 'maxAmp':np.max(amp_all)}

    return(df, info)



def convert_to_zscore(d, baseline):

    if baseline is None:
        baseline = range(d.shape[1])

    m, s = np.mean(d[:, baseline], 1), np.std(d[:, baseline], 1)

    return((d - m[:,None]) / s[:,None])



def post_data_loading_iterator(
    source_dir: str,
    expgroup_ID: Union[List[str], str, None] = None,
    subject_ID: Union[List[str], str, None] = None,
    session_ID: Union[List[str], str, None] = None,
    stim_ID: Union[List[str], str, None] = None
    ) -> List[str]:

    """
    Function for selecting data once the data loading has been done.

    Parameters
    ----------
    source_dir : str
        Path where the data structure was created during the data loading. Usually param <root_folder>/loaded_data/<experiment_ID>. Check parameters from data_loading.py to see what these correspond to.
    expgroup_ID : list of str | str | None
        - if string, the specified experimental group will be selected.
        - if list of string, the experimental groups in the list will be selected.
        - if None: all experimental groups available will be selected.
    subjects_ID : list of str | string | None
        Same as expgroup_ID but for subjects IDs.
    session_ID : list of str | string | None
        Same as expgroup_ID but for sessions IDs.
    stim_ID : list of str | string | None
        Same as expgroup_ID but for stimuli IDs.

    Returns
    -------
    filelist : list of str
        Lists containing the paths to the selected data.
    """

    filelist = []

    for dir in os.walk(source_dir):

        if len(dir[-1]) != 0:

            for filename in dir[-1]:

                if filename[-4:] == '.npy':

                    filename_split = filename.split('_')
                    stim = filename_split[-2]
                    session = None if len(filename_split) < 5 else filename_split[2]
                    subject = None if len(filename_split) < 4 else filename_split[1]
                    expgroup = None if len(filename_split) < 3 else filename_split[0]

                    bool_expgroup = (expgroup is None) or (expgroup_ID is None) or (expgroup in expgroup_ID)
                    bool_subject = (subject is None) or (subject_ID is None) or (subject in subject_ID)
                    bool_session = (session is None) or (session_ID is None) or (session in session_ID)
                    bool_stim = (stim_ID is None) or (stim in stim_ID)

                    if bool_expgroup and bool_subject and bool_session and bool_stim:
                        filelist.append(os.path.join(dir[0], filename))

    assert len(filelist) != 0, "No file found... Check the path or the IDs you requested."

    return(filelist)



def load_atlas(atlas_path: str) -> np.ndarray:

    """
    Utility function for loading the atlas from the given path. Left hemisphere will be identified by negative values.

    Parameters
    ----------
    atlas_path : str
        Path to the atlas file.

    Returns
    -------
    atlas : ndarray
        The atlas volume (3D).
    """

    atlas = np.load(atlas_path).astype('int64')
    coronal_dim_half = int(atlas.shape[2] / 2)
    atlas[:,:, :coronal_dim_half+1] *= -1

    return(atlas)



def reshape_volume_for_display(
    volume: np.ndarray,
    volume_boundaries: np.array = None,
    ncols: int = 8,
    return_for_clustering=False
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
    return_for_clustering : bool
        If True, returns extra values useful for the colormap display in the clustering object.

    Returns
    -------
    res : ndarray
        A 2D array with values being either int (cluster ID) or tuple (color associated with cluster ID) depeding on the 'apply_colormap' parameter.
    """

    if volume_boundaries is None:
        nz, ny, nx = volume.shape[0], volume.shape[1], volume.shape[2]
    else:
        zmin, zmax = volume_boundaries[0]
        ymin, ymax = volume_boundaries[1]
        xmin, xmax = volume_boundaries[2]
        nz, ny, nx = zmax-zmin+1, ymax-ymin+1, xmax-xmin+1
    nrows = int(ny / ncols) + 1

    res = np.zeros((nz*nrows, nx*ncols))
    y_idx = 0

    for i in range(nrows):

        for j in range(ncols):

            if y_idx < ny:

                if volume_boundaries is None:
                    res[nz*i:nz*(i+1), nx*j:nx*(j+1)] = volume[:, y_idx, :]
                else:
                    res[nz*i:nz*(i+1), nx*j:nx*(j+1)] = volume[zmin:zmax+1, ymin+y_idx, xmin:xmax+1]

                res[nz*(i+1)-1, :] = -1
                res[:, nx*(j+1)-1] = -1

                y_idx += 1

    if return_for_clustering:
        return(res, nx, nz, nrows)
    else:
        return(res)



def name_from_path(path):

    """
    Simple utility function for getting the name from the path of loaded_data.

    Parameters
    ----------
    path : str
        Path from which the name is to be extracted.

    Returns
    -------
    str
        The extracted name.
    """

    path = os.path.normpath(path)

    if len(path.split('/')) > 1:
        return(path.split('/')[-1][:-4])
    else:
        return(path.split('\\')[-1][:-4])



def average_files(filelist: List[str]) -> np.ndarray:

    res = []

    for file in filelist:

        data = np.load(file)
        res.append(data)

    return(np.mean(res, 0))



def post_data_loading_average(
    source_dir: str,
    level_avg: str
    ):

    """
    A FINIR CORRECTEMNT, LES FICHIERS GENERES SONT PRIS EN COMPTE LORS DE OS.WALK
    """

    expgroups, subjects, sessions, stims = [], [], [], []

    for root, dirs, files in os.walk(source_dir):

        for file in files:

            if file[-4:] != '.npy':
                continue

            split = file.split('_')
            stims.append(split[-2])
            expgroups.append(split[0])

            if len(split) >= 4:
                subjects.append(split[1])
            if len(split) == 5:
                sessions.append(split[2])

    expgroups, subjects, sessions, stims = list(set(expgroups)), list(set(subjects)), list(set(sessions)), list(set(stims))

    for stim in stims:

        if level_avg == 'expgroup':
            filelist = post_data_loading_iterator(source_dir, stim_ID=stim)
            print(filelist)
            res = average_files(filelist)
            np.save(os.path.join('E:/17-brainpain', F"{stim}_avg.npy"), res)
            continue

        for expgroup in expgroups:

            if level_avg == 'subjects':
                filelist = post_data_loading_iterator(source_dir, expgroup_ID=expgroup, stim_ID=stim)
                print(filelist)
                res = average_files(filelist)
                np.save(os.path.join('E:/17-brainpain', F"{expgroup}_{stim}_avg.npy"), res)

    print("Average completed.")



def generate_color_coded_atlas_flat(
    atlas_resolution: int,
    ext: str,
    cmap = None
    ):

    """
    Function for displaying an atlas color-coded depending on the major anatomical structures.

    Parameters
    ----------
    atlas_resolution : int
        The resolution of the atlas to be loaded.
    ext : str
        The extension of the atlas to be loaded.
    cmap : None | list of colors
        List of colors for the anatomical structures.
    """

    groups = np.array(gv.MAJOR_STRUCTURES + ['empty'])

    if cmap == None:
        cmap = plt.get_cmap('tab20')
        cmap = [cmap(i/len(groups)) for i in range(len(groups))]
    atlas = np.load(F"../atlases/atlases_npy/atlas_ccf_v3_{atlas_resolution}{ext}.npy")
    atlas = reshape_volume_for_display(atlas)
    regions_nb, regions_acr, groups_acr = extract_info_from_region_file(F"../atlases/atlases_lists/regions_ccf_v3_{atlas_resolution}{ext}.txt")
    atlas_colored = np.zeros(np.append(atlas.shape, 4))

    for r in regions_nb:

        try:
            g = np.where(groups == groups_acr[regions_nb == r][0])[0][0]
        except IndexError:
            pass
        if g != 12:
            atlas_colored[atlas == r] = cmap[g]

    plt.imshow(atlas_colored)
    plt.show()



class Formatter(object):

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

    def __init__(self, atlas, regions_acr, regions_nb):
        self.atlas = np.abs(atlas)
        self.regions_acr = regions_acr
        self.regions_nb = regions_nb

    def __call__(self, x, y):
        g = self.regions_acr[self.regions_nb == int(self.atlas[int(y), int(x)])]
        return(F"{g}")



def get_atlas_and_info_paths(atlas_resolution, ext):

    """
    Utility function to get the path to the atlases and info files from the library.

    Parameters
    ----------
    atlas_resolution : int
        Resolution of the atlas to use, in µm.
    ext : str
        Extension of the atlas to be used.
    """

    path_atlas = files('pyfus.atlases.atlases_npy').joinpath('atlas_ccf_v3_100_nolayersnoparts.npy')
    path_atlas_contours = files('pyfus.atlases.atlases_npy').joinpath('atlas_ccf_v3_100_contours.npy')
    path_regions_info = files('pyfus.atlases.atlases_lists').joinpath('regions_ccf_v3_100_nolayersnoparts.txt')
    return(path_atlas, path_atlas_contours, path_regions_info)



def get_atlas_mask_from_regions(atlas, regions_nb, regions_acr, acr_list):

    """
    Utility function to get a volume matching the atlas size with ones where voxels belong to the selected regions and zeros elesewhere.

    Parameters
    ----------
    atlas : array
        Resolution of the atlas to use, in µm.
    regions_acr : array
        Array containing the acronyms of the regions included in the atlas.
    regions_nb : array
        Array containing the regions' numbers / IDs included in the atlas.
    acr_list : list of tuples
        List containing the acronyms and associated hemispheres of the selected regions. Ex: [('SCs', 'L'), ('VISp', 'R')].
    """

    assert type(acr_list) == list

    mask = np.zeros(atlas.shape)

    for acr, h in acr_list:
        assert len(regions_nb[regions_acr == acr]) != 0, F"Acronym \'{acr}\' is not correct. Please check the atlas."
        nb = regions_nb[regions_acr == acr][0]
        h_sign = -1 if h == 'L' else 1
        mask[atlas == nb*h_sign] = 1

    return(mask.astype('bool'))



def get_regions_from_groups(
    group: str,
    hemisphere: str,
    regions_acr: np.array,
    groups_acr: np.array
    ):

    """
    Utility function for converting an anatomical group into a list of regions included in that group in the format expected by the clustering code.

    Parameters
    ----------
    group : str
        Acronym of the anatomical group
    hemisphere : str
        'L' or 'R' respectively for left and right hemispheres, or 'LR' for both.
    regions_acr : array
        1D array containing the regions' acronyms.
    groups_acr : array
        1D array containing the anatomical groups' acronyms.

    Returns
    -------
    res : list
        List of tuples in the format (region acronym, hemisphere 'L' or 'R')
    """

    assert group in gv.MAJOR_STRUCTURES, "The structure you selected does not exist..."
    assert hemisphere in ['L', 'R', 'LR'], "Hemispheres values should be either 'L' (left), 'R' (right) or 'LR' (both)"

    res = []
    regions = regions_acr[groups_acr == group]

    if hemisphere == 'LR':
        for r in regions:
            res.append((r, 'L'))
            res.append((r, 'R'))
    else:
        for r in regions:
            res.append((r, hemisphere))

    return(res)



def get_regions_from_hemisphere(
    hemisphere: str,
    regions_acr: np.array,
    groups_acr: np.array
    ):

    """
    Utility function for converting a hemisphere group into a list of regions included in that group in the format expected by the clustering code.

    Parameters
    ----------
    group : str
        Acronym of the anatomical group
    hemisphere : str
        'L' or 'R' respectively for left and right hemispheres, or 'LR' for both.
    regions_acr : array
        1D array containing the regions' acronyms.
    groups_acr : array
        1D array containing the anatomical groups' acronyms.

    Returns
    -------
    res : list
        List of tuples in the format (region acronym, hemisphere 'L' or 'R')
    """

    assert hemisphere in ['L', 'R', 'LR'], "Hemispheres values should be either 'L' (left), 'R' (right) or 'LR' (both)"

    res = []
    for g in gv.MAJOR_STRUCTURES:
        res += get_regions_from_groups(g, hemisphere, regions_acr, groups_acr)

    return(res)



def save_object(path, obj):

    """
    Utility function to save an object, for example a clustering object. Rely on pickle.

    Parameters
    ----------
    path : string
        The path where the object should be saved. The extension of the file must be '.p'
    obj : object
        The object to be saved.
    """

    pickle.dump(obj, open(path, 'wb'))



def load_object(path):

    """
    Utility function to load an object, for example a clustering object. Rely on pickle.

    Parameters
    ----------
    path : string
        The path where the object is saved.

    Returns
    -------
    obj
        The loaded object.
    """

    return(pickle.load(open(path, 'rb')))
