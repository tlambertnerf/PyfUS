"""
@author: Théo Lambert, adapted from original code by Gabriel Montaldo

This module regroups all the functions related to data registration.
Throughout the documentation, the following acronyms are used:
    - DV: dorso-ventral axis
    - AP: antero-posterior axis
    - LR: left-right axis
"""


import numpy as np
import matplotlib.pyplot as plt
import time, pickle, typing
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform, map_coordinates
from scipy.io import loadmat
from PIL import Image
from skimage.transform import warp
import nibabel as nib
import mat73
from typing import Union


def visualize_volume(
    volume: np.ndarray,
    atlas: np.ndarray = None,
    orientation: str = 'coronal',
    overlay: bool = True
    ) -> None:

    """
    Open a window and display consecutive sections of a volume one by one.

    Parameters
    ----------
    volume : ndarray
        Volume to be displayed, array-like with dimensions order (DV, AP, LR, time (optional)).
    atlas : ndarray
        Atlas to be displayed on a second plot, array-like with dimensions order (DV, AP, LR).
    orientation : str
        Either 'axial', 'coronal' or 'sagittal'. Determine which dimensions are displayed.
    overlay : bool
        Whether to overlay the atlas onto the volume. If atlas is None, nothing happens.

    Returns
    -------
    None
    """

    if orientation == 'axial':
        dim, axes_swap = volume.shape[0], [0,2]
    elif orientation == 'coronal':
        dim, axes_swap = volume.shape[1], [1,2]
    elif orientation == 'sagittal':
        dim, axes_swap = volume.shape[2], [0,0]
    else:
        print('Unrecognized orientation (coronal, sagittal, axial). No display.')
        return()

    data = np.moveaxis(volume, *axes_swap)
    if len(data.shape) == 4: # if the volume is 4d (with time dim), average over time to get a single volume
        data = np.mean(data, 3)

    if atlas is not None:
        data_atlas = np.moveaxis(atlas, *axes_swap)
        fig, (ax1, ax2) = plt.subplots(nrows=2)
    else:
        fig, ax1 = plt.subplots(nrows=1)
    plt.ion()
    plt.show()

    for i in range(dim):

        img = np.power(np.abs(data[:,:,i]), .25)
        img = np.nan_to_num(img)
        img = img / np.max(img)*255

        if atlas is not None:
            img_atlas = data_atlas[:,:,i]
            img_atlas = img_atlas / np.max(img_atlas)*255
            ax2.imshow(img_atlas)
            if overlay:
                img[data_atlas[:,:,i] > 0] = 0

        ax1.imshow(img, cmap='gray')

        plt.draw()
        plt.pause(0.05)

    plt.close()


def load_data_and_tf(
    data_path: str,
    tf_path: str,
    atlas_resolution: Union[int, None] = None,
    visualize_data: Union[str, None] = None,
    reshape: bool = False
    ) -> (dict, dict):

    """
    Function for loading raw fUS data from .mat files and associated transformation matrix. Output format is (DV-VD, AP-PA, LR-RL, time)

    Parameters
    ----------
    data_path : str
        Path to the fUS data (ex: C:/tmp/fusi_data.mat).
    tf_path : str
        Path to the affine transformation matrix used to register the data. (ex: C:/tmp/transf.mat)
    atlas_resolution : int
        Resolution of the atlas to which the data will be registered in µm (ex: 100). The original estimate is made with the 50µm CCF v3 atlas,
        so the translations need to be scaled. Set to None if you don't want any normalization.
    visualize_data : None | str
        Fed to the visualize_volume function for parameter 'orientation', please refer to the associated doc.

    Returns
    -------
    data : ndarray
        The fUS data dict, with fields 'Data' (the fUS volume), 'Size' (the data ndarray shape), 'VoxelSize' (voxel size in mm), 'Direction' (axes DV-VD, AP-PA, LR-RL)
    tf: dict
        The transformation matrix dict, with fields 'M' (the transform matrix), the rest is useless
    """


    data_mat = loadmat(data_path, squeeze_me=True)
    voxel_size = data_mat['md']['voxelSize'].tolist()
    direction = data_mat['md']['Direction']

    ### data reshaping to get data in the order DV.AP.LR
    if reshape:
        size_ = data_mat['md']['size'].tolist()
        size_ = [size_[0], size_[2], size_[1]]
        data_shape = np.append(size_, data_mat['I'].shape[2])
        data_fus = data_mat['I'].reshape(data_shape)
    else:
        size_ = data_mat['md']['imageSize'].tolist()
        size_ = [size_[i] for i in [0,2,1]]
        data_fus = np.swapaxes(data_mat['I'], 1, 2)

    data = {'Data':data_fus, 'Size':size_, 'VoxelSize':voxel_size, 'Direction':str(direction)}

    if visualize_data: #visualization in coronal
        visualize_volume(data['Data'], orientation=visualize_data)

    try:
        tf_mat = loadmat(tf_path, squeeze_me=True)['Transf'].tolist()
        M = tf_mat[0]
    except NotImplementedError: # for fucking compatibility
        tf_mat = mat73.loadmat(tf_path)['Transf']
        M = tf_mat['M']
    # original estimate is made with 50µm CCF v3 atlas, so translation (last line) must adjusted if the atlas resolution changes
    if atlas_resolution is not None:
        M[3] /= (atlas_resolution/50)
        M[3,3] = 1

    return(data, M)


def interpolate3D(
    data: dict,
    atlas_resolution: int,
    tf_matrix: dict,
    visualize_interp: bool = False
    ) -> dict:

    """
    Function for interpolating raw fUS data to the atlas resolution.

    Parameters
    ----------
    data : dict
        The fUS data dict, with fields 'Data' (the fUS volume), 'Size' (the data ndarray shape), 'VoxelSize' (voxel size in mm), 'Direction' (axes DV-VD, AP-PA, LR-RL)
    atlas_resolution : int
        Resolution of the atlas to which the data will be registered in µm (ex: 100).
    tf_matrix : dict
        The transformation matrix dict, with fields 'M' (the transform matrix)
    visualize_data : None | str
        Fed to the visualize_volume function for parameter 'orientation', please refer to the associated doc.

    Returns
    -------
    data_interp : dict
        The fUS interpolated data dict, with fields 'Data' (the interpolated fUS volume), 'VoxelSize' (voxel size in mm), 'Direction' (axes DV-VD, AP-PA, LR-RL)
    """

    # Z=DV, Y=AP, X=LR
    # visualize_interp: if None, no display, otherwise string specifying the orientation (axial, coronal, sagittal)
    dzint, dyint, dxint = np.array([atlas_resolution/1000]*3)
    dz, dy, dx = data['VoxelSize']
    nz, ny, nx = data['Size']

    n1z = int((nz-1)*dz / dzint) + 1
    n1y = int((ny-1)*dy / dyint) + 1
    n1x = int((nx-1)*dx / dxint) + 1

    Z, Y, X = np.linspace(0, nz, nz), np.linspace(0, ny, ny), np.linspace(0, nx, nx)
    interp = RegularGridInterpolator((Z, Y, X), data['Data'], method='nearest')

    Zq, Yq, Xq = np.meshgrid(range(n1z-1)*dzint/dz+1, range(n1y-1)*dyint/dy+1, range(n1x-1)*dxint/dx+1, indexing='ij')
    Zq, Yq, Xq = np.round(Zq, decimals=8), np.round(Yq, decimals=8), np.round(Xq, decimals=8)
    ai = interp((Zq, Yq, Xq))

    data_interp = {'Data': ai, 'VoxelSize': np.array([atlas_resolution/1000]*3), 'Direction': data['Direction']}

    if visualize_interp: # visualization in coronal
        visualize_volume(data_interp['Data'], orientation=visualize_interp)

    return(data_interp)


def flip_dims(
    data: dict
    ) -> np.ndarray:

    """
    Utility function for flipping dimensions in the fUS data when required from the Direction metadata.

    Parameters
    ----------
    data : dict
        A fUS data dict, with fields 'Data' (the fUS volume), 'Size' (the data ndarray shape), 'VoxelSize' (voxel size in mm), 'Direction' (axes DV-VD, AP-PA, LR-RL)

    Returns
    -------
    volume : ndarray
        The volume with correctly flipped dimensions (corresponding to data['Data']), /!\ the function does not return the full dict!
    """

    directions = data['Direction'].split('.')

    if directions[0] == 'VD':
        data['Data'] = np.flip(data['Data'], 0)

    if directions[1] == 'RL': # directions[1] --> flip(.., 2) because order of dimensions have changed
        data['Data'] = np.flip(data['Data'], 2)

    if directions[1] == 'PA': # directions[2] --> flip(.., 1) because order of dimensions have changed
        data['Data'] = np.flip(data['Data'], 1)

    return(data['Data'])


def register_data(
    atlas: np.ndarray,
    data: dict,
    tf_matrix: dict,
    atlas_resolution: int,
    visualize_reg: bool = False,
    overlay: bool = True
    ) -> np.ndarray:

    """
    Main function for registering  raw fUS data to the selected atlas.

    Parameters
    ----------
    atlas : ndarray
        3D volume of the atlas, where each value is a number indicating to which region a voxel belongs.
    data : dict
        The fUS data dict, with fields 'Data' (the fUS volume), 'Size' (the data ndarray shape), 'VoxelSize' (voxel size in mm), 'Direction' (axes DV-VD, AP-PA, LR-RL).
    tf_matrix : np.ndarray
        The 4x4 transform matrix.
    visualize_reg : None | str
        Fed to the visualize_volume function for parameter 'orientation', please refer to the associated doc.
    overlay : bool
        To specify whether the atlas should be overlaid on the registered data. /!\ if True, you should use the atlas contours as input for variable 'atlas'

    Returns
    -------
    data_interp : dict
        The fUS interpolated data dict, with fields 'Data' (the interpolated fUS volume), 'VoxelSize' (voxel size in mm), 'Direction' (axes DV-VD, AP-PA, LR-RL)
    """

    data_interp_dic = interpolate3D(data, atlas_resolution, tf_matrix)
    data_interp = flip_dims(data_interp_dic)
    data_interp = np.swapaxes(data_interp, 0, 1)

    mat, vec = nib.affines.to_matvec(np.linalg.inv(tf_matrix.T))
    shape_ = (atlas.shape[1], atlas.shape[0], atlas.shape[2])

    data_reg = np.ndarray(np.append(shape_, data_interp.shape[3]))
    # iterate over the temporal dimension
    for t in range(data_reg.shape[3]):
        data_reg[...,t] = affine_transform(data_interp[...,t], mat, vec, output=np.ndarray(shape_), output_shape=shape_, mode='constant', cval=np.nan)

    # swapping back the dimensions in the correct order
    data_reg = np.swapaxes(data_reg, 0, 1)

    if visualize_reg:
        visualize_volume(data_reg, atlas=atlas, orientation=visualize_reg, overlay=overlay)

    return(data_reg)
