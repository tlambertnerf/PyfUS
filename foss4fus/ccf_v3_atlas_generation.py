import numpy as np
import json
import nrrd
from skimage import measure


MAJOR_STRUCTURES = ['P', 'MY', 'HY', 'TH', 'MB', 'CB', 'CTXsp', 'HPF', 'Isocortex', 'OLF', 'STR', 'PAL']
_NUL = object()

def get_all_id_acr(hierarchy, structure='empty'):

    """
    Recursively find all the acronyms in hierarchy, and associate them to a major anatomical group.
    :param hierarchy: JSON object. The hierarchy of the Allen CCF V3 in the JSON format.
    :param structure: string. Hyperparameter tracking the major anatomical group the acronym is associated with.
        Refer to the general doc for a list of the major groups.
    """

    if isinstance(hierarchy, dict):

        key_value = hierarchy.get("id", _NUL)  # _NUL if key not present

        if key_value is not _NUL:
            acr = hierarchy.get("acronym")
            if acr in MAJOR_STRUCTURES:
                structure = acr
            yield(key_value, acr, structure)

        for jsonkey in hierarchy:
            jsonvalue = hierarchy[jsonkey]
            for v in get_all_id_acr(jsonvalue, structure):  # recursive
                yield v

    elif isinstance(hierarchy, list):

        for item in hierarchy:
            for v in get_all_id_acr(item, structure):  # recursive
                yield v


def generate_regions_file(src, dst):

    # src: json file containing the Allen ccf v3 hierarchy
    # dst: filename of the txt file in which the regions list will be saved

    with open(src, 'r') as f:

        ccf_hierarchy = json.load(f)
        hierarchy = []

        l = get_all_id_acr(ccf_hierarchy)
        for elt in l:
            hierarchy.append(elt)
        hierarchy.sort(key = lambda tup: (tup[2], tup[0]))
        hierarchy_text_ordered = '\n'.join("%s %s %s" %tup for tup in hierarchy)

        with open(dst, 'w') as dst:
            dst.write(hierarchy_text_ordered)

        return(np.array(hierarchy))


def load_atlas(atlas_resolution, save_npy=True, extract_contours=False):

    atlas, meta_atlas = nrrd.read(F'../atlases/atlases_nrrd/annotation_{atlas_resolution}.nrrd')
    atlas = np.swapaxes(atlas, 0, 1)
    print(atlas.shape)
    #atlas = np.swapaxes(atlas, 1, 2)
    #atlas = np.flip(atlas, axis=1)

    if extract_contours:
        contours = np.zeros(atlas.shape)
        for i in range(contours.shape[2]):
            contours_i = measure.find_contours(atlas[:,:,i], level=1)
            for elt in contours_i:
                for coord in elt:
                    contours[int(coord[0]), int(coord[1]), i] = 1

    if save_npy:
        np.save(F'../atlases/atlases_npy/atlas_ccf_v3_{atlas_resolution}.npy', atlas)
        if extract_contours:
            np.save(F'../atlases/atlases_npy/atlas_ccf_v3_{atlas_resolution}_contours.npy', contours)

    return(atlas)


def get_regions_in_atlas(atlas, hierarchy, atlas_resolution, ext=''):

    hierarchy_ids = hierarchy[:,0].astype('int')
    atlas_ids = np.unique(atlas)

    regions_ids = np.argwhere(np.in1d(hierarchy_ids, atlas_ids))
    regions_arr = np.squeeze(np.array(hierarchy[regions_ids]))
    #conversion into list of tuples for easy sorting (group first, acronym second)
    regions_arr = np.array(sorted(list(zip(regions_arr[:,0], regions_arr[:,1], regions_arr[:,2])), key = lambda tup: (tup[2], tup[1])))

    with open(F'regions_ccf_v3_{atlas_resolution}{ext}.txt', 'w') as dst:

        text = '\n'.join(F"{x[0]} {x[1]} {x[2]}" for x in regions_arr)
        dst.write(text)


def generate_simplified_atlas(atlas_resolution, simplified_regions_list_path, ext):

    atlas = np.load(F"../atlases/atlases_npy/atlas_ccf_v3_{atlas_resolution}.npy")

    with open(simplified_regions_list_path, 'r') as f:

        for line in f:

            elt = line.rstrip().split(' ')
            if len(elt) == 3:
                from_, to_ = int(elt[0]), int(elt[2])
                atlas[atlas == from_] = to_

    np.save(F"../atlases/atlases_npy/atlas_ccf_v3_{atlas_resolution}_{ext}.npy", atlas)
    return(atlas)
