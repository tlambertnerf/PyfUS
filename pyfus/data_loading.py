"""
@author: Théo Lambert

This module regroups all the functions and classes related to data loading.
"""

import os, sys, random, typing, mat73
from typing import Union
import numpy as np
import pyfus.registration as reg
import pandas as pd
import pyfus.utils as u
import pyfus.quality_control as qc
import warnings
from scipy.io import loadmat
from time import time


# TO COMMENT WHEN IN DEVELOPMENT
warnings.filterwarnings('ignore')


class DataLoaderAndPreproc:

    """
    Object handling the whole data loading and preprocessing step.

    Parameters
    ----------
    root_folder : str
        Path to the folder in which are stored the data for all the experiments. See doc for more info on folder structure.
    experiment_ID : str
        Name/identifier of the experiment.
    expgroup_ID : list of str
        List providing the expgroups to be processed.
    subject_ID : list of str
        List providing the subject to be processed.
    session_ID : list of str
        List providing the sessions to be processed.
    stim_ID : list of str
        List providing the stimuli to be processed.
    mode : string
        Loading mode, to select in folder_tree or excel. See general doc for more information.
    excel_source : string
        Optional, used only if mode=='excel'. Path to the excel source file.
    reduction : string
        To select how to aggregate the data: single_trial, median or mean.
    level_avg : string
        Set the level at which the reduction will be performed: all, expgroup, subject, session.
    register : bool
        Whether or not to register the data.
    atlas_resolution : int
        Resolution of the atlas to which the data will be registered in µm (ex: 100).
    baseline : iterable
        Baseline period on which to baseline the data (ex: range(0,20)).
    make_reliability_maps : bool
        Whether or not to make reliability maps. Check the quality_control module for more info.
    remove_unreliable : float | None
        Whether or not to filter the loaded data based on the reliability map. Excluded values will be set to NaN.
    trial_preprocessing : object | None
        If None, will be ignored. If object, will be used to preprocess trials. Check quality_control.py for further details.
    """


    def __init__(self,
        root_folder: str,
        experiment_ID: str,
        expgroup_ID: list[str] = None,
        subject_ID: list[str] = None,
        session_ID: list[str] = None,
        stim_ID: list[str] = None,
        mode: str = 'folder_tree',
        excel_source: str = None,
        reduction: str = 'median',
        level_avg: str = 'stim',
        register: bool = True,
        atlas_resolution: int = 100,
        baseline: Union[None, typing.Iterable[int]] = None,
        make_reliability_maps: bool = False,
        remove_unreliable: Union[float, None] = None,
        trial_preprocessing: Union[object, None] = None
        ) -> None:

        self.root_folder = root_folder
        self.experiment_ID = experiment_ID
        self.expgroup_ID = expgroup_ID
        self.subject_ID = subject_ID
        self.session_ID = session_ID
        self.stim_ID = stim_ID

        self.reduction = reduction
        self.init_data_reduction_func()
        if self.reduction == 'single_trial':
            print("You selected single trial processing, no reduction and average will be performed (level_avg param will be ignored).")

        key = '%08x' % random.randrange(16**8)
        reg_ext = 'reg' if register else 'noreg'
        if self.reduction == 'single_trial':
            self.ext = F'{reduction}_{reg_ext}_{key}' #### DONT FORGET TO PROPERLY DEFINE THIS EXTENSION!!!!
        else:
            self.ext = F'{reduction}_{level_avg}_{reg_ext}_{key}' #### DONT FORGET TO PROPERLY DEFINE THIS EXTENSION!!!!

        self.output_folder = os.path.join(self.root_folder, "output", F"{self.experiment_ID}_{self.ext}")
        os.makedirs(self.output_folder, exist_ok=True)

        self.register = register
        self.atlas_resolution = atlas_resolution
        path_atlas, path_atlas_contours, _ = u.get_atlas_and_info_paths(atlas_resolution, "_nolayersnoparts")
        self.atlas, self.atlas_contours = np.load(path_atlas), np.load(path_atlas_contours)

        self.mode = mode

        if self.mode == 'folder_tree':
            self.scan_folder_tree()
            self.iterator = iter(self.data_iterator_folder_tree())

        elif self.mode == 'excel':
            self.excel_source = excel_source
            #only assumption: all files from a same session are in the same folder
            self.extract_info_excel_file(self.excel_source)
            self.iterator = iter(self.data_iterator_excel_file())

        else:
            print('Wrong mode (choose folder_tree or excel) --> failed initialization. Exiting...')
            sys.exit()

        self.levels = {'all':1, 'expgroup':2, 'subject':3, 'session':4}
        if level_avg not in ['all', 'expgroup', 'subject', 'session']:
            print('Unrecognized averaging level (choose expgroup, subject, session or all) --> failed initialization. Exiting...')
            sys.exit()
        self.level_avg = level_avg if reduction != 'single_trial' else None

        self.baseline = baseline

        assert not((make_reliability_maps == True) and (self.reduction == 'single_trial' or self.register == 'False')), "Impossible to make reliability maps if data are not registered or if the extraction is done at single trial level"
        self.make_reliability_maps = make_reliability_maps
        self.remove_unreliable = remove_unreliable

        self.trial_preprocessing = trial_preprocessing

        print(F"Processed data will be stored in {self.experiment_ID}_{self.ext}.")


    def check_ids(self):

        """
        Method for checking that the different IDs that were extracted from the folder structure or the excel file do not have the character '_' in their string, as it is used later to parse filenames.
        """

        for elt in self.expgroup_ID:
            assert len(elt.split('_')) == 1, F"Invalid ID: character '_' detected in {elt}. Please remove this character. Exiting."
        for elt in self.subject_ID:
            assert len(elt.split('_')) == 1, F"Invalid ID: character '_' detected in {elt}. Please remove this character. Exiting."
        for elt in self.session_ID:
            assert len(elt.split('_')) == 1, F"Invalid ID: character '_' detected in {elt}. Please remove this character. Exiting."
        for elt in self.stim_ID:
            assert len(elt.split('_')) == 1, F"Invalid ID: character '_' detected in {elt}. Please remove this character. Exiting."


    def check_transf(self,
        session_path: str
        ) -> Union[None, str]:

        """
        Method for checking if a transformation matrix for the session can be found in the folder structure.
        It must be located in the 'other' folder, and have either 'transf' or 'Transf' separated by '_'. Examples of valid names are: 'tmp_Transf_001.mat', 'Transf.mat' or 'transf_XXX.mat'.

        Parameters
        ----------
        session_path : str
            Path to the session for which the transform matrix must be searched.

        Returns
        -------
        transf_path : None | str
            Returns None if no transformation matrix was found, or the path to the transform matrix.
        """

        for file in os.listdir(os.path.join(session_path, 'other')):

            filename = os.fsdecode(file)
            filename_s = filename[:-4].split('_')

            if ('transf' in filename_s) or ('Transf' in filename_s):
                return(os.path.join(session_path, 'other', filename))

        return(None)


    def scan_folder_tree(
        self,
        print_skip_session: bool = False
        ) -> None:

        """
        Method for checking whether all the requested data and associated transform matrix have been found in the folder structure.

        Parameters
        ----------
        print_skip_session : bool
            Whether or not to print the requested sessions that have not been found.
        """

        nothing_skipped, dir_count = True, 0

        exp_path = os.path.join(self.root_folder, self.experiment_ID)
        dirs_expgroup = self.expgroup_ID if self.expgroup_ID is not None else os.listdir(exp_path)

        expgroups, subject, sessions, stims = [], [], [], []

        for dir_expgroup in dirs_expgroup:

            if os.path.isdir(os.path.join(exp_path, dir_expgroup)):
                pass
            else:
                print(F"Expgroup {dir_expgroup} was not found. Skipped.")
                nothing_skipped = False
                continue

            expgroups.append(dir_expgroup)

            expgroup_path = os.path.join(exp_path, dir_expgroup)
            dirs_subject = self.subject_ID if self.subject_ID is not None else os.listdir(expgroup_path)

            for dir_subject in dirs_subject:

                if os.path.isdir(os.path.join(exp_path, dir_expgroup, dir_subject)):
                    pass
                else:
                    print(F"subject {dir_subject} was not found for expgroup {dir_expgroup}. Skipped.")
                    nothing_skipped = False
                    continue

                subject.append(dir_subject)

                subject_path = os.path.join(expgroup_path, dir_subject)
                dirs_session = self.session_ID if self.session_ID is not None else os.listdir(subject_path)

                for dir_session in dirs_session:

                    if os.path.isdir(os.path.join(subject_path, dir_session)):
                        pass
                        dir_count += 1
                    else:
                        if print_skip_session:
                            print(F"Session {dir_session} was not found for subject {dir_subject} {dir_expgroup}. Skipped.")
                        continue

                    sessions.append(dir_session)

                    session_path = os.path.join(subject_path, dir_session)
                    dirs_stim = self.stim_ID if self.stim_ID is not None else os.listdir(os.path.join(session_path, 'fus'))

                    transf_path = self.check_transf(session_path)
                    if transf_path is None and self.register:
                        print(F"/!\ No transformation matrix was found for {dir_session} {dir_subject} {dir_expgroup}. Will be ignored during registration.")

                    for dir_stim in dirs_stim:

                        if os.path.isdir(os.path.join(session_path, 'fus', dir_stim)):
                            pass
                        else:
                            print(F"Stim {dir_stim} was not found for subject {dir_subject} {dir_expgroup}, session {dir_session}. Skipped.")
                            nothing_skipped = False
                            continue

                        stims.append(dir_stim)

                        stim_path = os.path.join(session_path, 'fus', dir_stim)

        self.expgroup_ID = expgroups
        self.subject_ID = list(set(subject))
        self.session_ID = list(set(sessions))
        self.stim_ID = list(set(stims))
        self.check_ids()

        if nothing_skipped:
            print("All the data you requested have been found.")

        if dir_count == 0:
            print("No session was found... please check your paths and requested sessions.")
            sys.exit()

        proceed = input("Proceed? [Y/n]")
        if proceed not in ['Y', 'y']:
            print("Exiting.")
            sys.exit()


    def replicate_folder_tree_structure(
        self,
        level: Union[str, None] = None
        ) -> None:

        """
        Method for replicating the folder tree structure in order to store the processed data.

        Parameters
        ----------
        level: str | None
            Level at which the reduction will be performed. Set to None if single trials are requested, then the full folder tree will be reproduced.
        """

        level = self.level_avg if level == None else level
        if self.reduction == 'single_trial':
            level_num = 5
        else:
            level_num = self.levels[level]

        path_exp = os.path.join(self.root_folder, 'loaded_data', F"{self.experiment_ID}_{self.ext}")

        for expgroup in self.expgroup_ID:

            path = os.path.join(path_exp, expgroup)
            os.makedirs(os.path.join(path_exp, expgroup), exist_ok=True)

            if level_num > 2:

                for subject in self.subject_ID:

                    path = os.path.join(path_exp, expgroup, subject)
                    os.makedirs(path, exist_ok=True)

                    # to avoid creating a folder all subjects
                    if self.mode == 'folder_tree':
                        cond_sub = os.path.isdir(os.path.join(self.root_folder, self.experiment_ID, expgroup, subject))
                    elif self.mode == 'excel':
                        cond_sub = len(self.structure_dict[expgroup][subject]) != 0
                    else:
                        pass

                    if not cond_sub:
                        continue

                    if level_num > 3:

                        for session in self.session_ID:

                            # to avoid creating all sessions for all subject
                            if self.mode == 'folder_tree':
                                cond_ses = os.path.isdir(os.path.join(self.root_folder, self.experiment_ID, expgroup, subject, session))
                            elif self.mode == 'excel':
                                cond_ses = len(self.structure_dict[expgroup][subject][session][self.stim_ID[0]]) != 0
                            else:
                                pass

                            if cond_ses:

                                path = os.path.join(path_exp, expgroup, subject, session)
                                os.makedirs(path, exist_ok=True)

                                if level_num >= 4:

                                    for stim in self.stim_ID:

                                        if self.mode == 'folder_tree':
                                            cond_stim = os.path.isdir(os.path.join(self.root_folder, self.experiment_ID, expgroup, subject, session, 'fus', stim))
                                        elif self.mode == 'excel':
                                            cond_stim = len(self.structure_dict[expgroup][subject][session][stim]) != 0
                                        else:
                                            pass

                                        if cond_stim:
                                            path = os.path.join(path_exp, expgroup, subject, session, stim)
                                            os.makedirs(path, exist_ok=True)



    def data_iterator_folder_tree(self,
        level_avg: Union[str, None] = None,
        ) -> None:

        """
        Method for generating an iterator over the folder tree structure and yield destination filenames as well as raw data paths.

        Parameters
        ----------
        level_avg: str | None
            Level at which the reduction will be performed. Set to None if single trials are requested, then the full folder tree will be reproduced.
        """

        # not using directly self.level_avg to facilitate the list_selected_data method
        level_avg = self.level_avg if level_avg is None else level_avg

        exp_path = os.path.join(self.root_folder, self.experiment_ID)
        exp_path_dst = os.path.join(self.root_folder, 'loaded_data', F'{self.experiment_ID}_{self.ext}')

        for dir_stim in self.stim_ID:

            filelist_all, ext = [], dir_stim

            for dir_expgroup in self.expgroup_ID:

                expgroup_path, expgroup_path_dst = os.path.join(exp_path, dir_expgroup), os.path.join(exp_path_dst, dir_expgroup)

                for dir_subject in self.subject_ID:

                    subject_path, subject_path_dst = os.path.join(expgroup_path, dir_subject), os.path.join(expgroup_path_dst, dir_subject)

                    for dir_session in self.session_ID:

                        if os.path.isdir(os.path.join(self.root_folder, self.experiment_ID, dir_expgroup, dir_subject, dir_session)):

                            session_path, session_path_dst = os.path.join(subject_path, dir_session), os.path.join(subject_path_dst, dir_session)

                            transf_path = self.check_transf(session_path)

                            stim_path, stim_path_dst = os.path.join(session_path, 'fus', dir_stim), os.path.join(session_path_dst, dir_stim)
                            if os.path.isdir(os.path.join(stim_path)):

                                filelist, filenames = [], []

                                for file in os.listdir(stim_path):

                                    filename = os.path.join(stim_path, os.fsdecode(file))

                                    if self.reduction == 'single_trial':
                                        filename_dst = os.path.join(session_path_dst, dir_stim, os.fsdecode(file)[:-4]+".npy")
                                        yield filename_dst, [(F"{dir_expgroup} {dir_subject} {dir_session} {dir_stim} {os.fsdecode(file)}", [filename], transf_path)]
                                    else:
                                        filelist.append(filename)
                                        filenames.append(os.fsdecode(file))

                                filenames = [F"{dir_expgroup}_{dir_subject}_{dir_session}_{dir_stim}"] + filenames
                                if self.reduction != 'single_trial' and level_avg == 'session':
                                    filename_dst = os.path.join(stim_path_dst, filenames[0]+"_avg.npy")
                                    yield filename_dst, [('\n'.join(filenames), filelist, transf_path)]

                                if level_avg in ['expgroup', 'subject']:
                                    filelist_all.append(('\n'.join(filenames), filelist, transf_path))

                    if level_avg == 'subject':
                        filename_dst = os.path.join(subject_path_dst, F"{dir_expgroup}_{dir_subject}_{dir_stim}_avg.npy")
                        yield filename_dst, filelist_all
                        filelist_all = []

                if level_avg == 'expgroup':
                    filename_dst = os.path.join(expgroup_path_dst, F"{dir_expgroup}_{dir_stim}_avg.npy")
                    yield filename_dst, filelist_all
                    filelist_all = []

            if level_avg == 'all':
                filename_dst = os.path.join(exp_path_dst, F"{dir_stim}_avg.npy")
                yield filename_dst, filelist_all
                filelist_all = []


    def extract_info_excel_file(self,
        excel_source: str
        ):

        """
        Function for extracting the data structure (subject, sessions, etc) from the excel file and create a dictionary representing this structure.
        If the data associated with a key is empty, it means this session or stimulus does not exist for this animal.

        Parameters
        ----------
        excel_source: str
            Path to the excel file from which to extract the structure.
        """

        info = pd.read_excel(excel_source)

        # first abstracting out the folder structure
        expgroup_ID, subject_ID, session_ID, stim_ID = [], [], [], []
        nothing_skipped, dir_count = True, 0

        for idx, row in info.iterrows():

            expgroup, subject, session, stim = row['expgroup_ID'], row['subject_ID'], row['session_ID'], row['stim_ID']

            expgroup_ID.append(str(expgroup))
            subject_ID.append(str(subject))
            session_ID.append(str(session))
            stim_ID.append(str(stim))

        self.expgroup_ID, self.subject_ID, self.session_ID, self.stim_ID = list(set(expgroup_ID)), list(set(subject_ID)), list(set(session_ID)), list(set(stim_ID))
        self.check_ids()

        # creating and filling the dictionary
        structure_dict = {expgroup_ID: {subject_ID: {session_ID: {stim_ID:[] for stim_ID in self.stim_ID} for session_ID in self.session_ID} for subject_ID in self.subject_ID} for expgroup_ID in self.expgroup_ID}

        for idx, row in info.iterrows():

            session_stim_path = row['session_and_stim_path']
            expgroup, subject, session, stim = str(row['expgroup_ID']), str(row['subject_ID']), str(row['session_ID']), str(row['stim_ID'])
            transf_path = row['transf_path']

            if os.path.isdir(session_stim_path):
                dir_count += 1
            else:
                print(F"Stim {stim} was not found for subject {subject} {expgroup}, session {session}. Skipped.")
                nothing_skipped = False
                continue

            if self.register and type(transf_path) != str or not os.path.isfile(transf_path):
                print(F"/!\ No transform matrix for {expgroup} {subject} {session} was found. Will be ignored during registration.")

            files = []
            for f in os.listdir(session_stim_path):
                filename = os.fsdecode(f)
                files.append(os.path.join(session_stim_path, filename))

            structure_dict[expgroup][subject][session][stim] = (files, transf_path)

        if nothing_skipped:
            print("All the data you requested have been found.")

        if dir_count == 0:
            print("No session was found... please check your paths and requested sessions.")
            sys.exit()

        proceed = input("Proceed? [Y/n]")
        if proceed not in ['Y', 'y']:
            print("Exiting.")
            sys.exit()

        self.structure_dict = structure_dict


    def data_iterator_excel_file(self,
        level_avg: Union[str, None] = None,
        ):

        """
        Method for generating an iterator from the info extracted of an excel file and yield destination filenames as well as raw data paths.
        Note: the code is quite similar to the iterator of folder tree, but they were not merged for the sake of clarity since multiple lines differ.

        Parameters
        ----------
        level_avg : string | None
            Level at which the reduction will be performed. Set to None if single trials are requested, then the full folder tree will be reproduced.
        """

        # not using directly self.level_avg to facilitate the list_selected_data method
        level_avg = self.level_avg if level_avg is None else level_avg

        exp_path_dst = os.path.join(self.root_folder, 'loaded_data', F'{self.experiment_ID}_{self.ext}')

        for dir_stim in self.stim_ID:

            filelist_all, ext = [], dir_stim

            for dir_expgroup in self.expgroup_ID:

                expgroup_path_dst = os.path.join(exp_path_dst, dir_expgroup)

                for dir_subject in self.subject_ID:

                    subject_path_dst = os.path.join(expgroup_path_dst, dir_subject)

                    for dir_session in self.session_ID:

                        if len(self.structure_dict[dir_expgroup][dir_subject][dir_session][dir_stim]) != 0:

                            session_path_dst = os.path.join(subject_path_dst, dir_session)

                            transf_path = self.structure_dict[dir_expgroup][dir_subject][dir_session][dir_stim][1]
                            if self.register and type(transf_path) != str or not os.path.isfile(transf_path):
                                transf_path = None

                            stim_path_dst = os.path.join(session_path_dst, dir_stim)

                            filelist, filenames = [], []
                            filelist = self.structure_dict[dir_expgroup][dir_subject][dir_session][dir_stim][0]

                            if self.reduction == 'single_trial':
                                for file in filelist:
                                    fname = file.split("\\")[-1][:-4]
                                    filename_dst = os.path.join(session_path_dst, dir_stim, fname+".npy")
                                    yield filename_dst, [(F"{dir_expgroup} {dir_subject} {dir_session} {dir_stim} {fname}", [file], transf_path)]

                            filenames = [F"{dir_expgroup}_{dir_subject}_{dir_session}_{dir_stim}"] + filenames
                            if self.reduction != 'single_trial' and level_avg == 'session':
                                filename_dst = os.path.join(stim_path_dst, filenames[0]+"_avg.npy")
                                yield filename_dst, [('\n'.join(filenames), filelist, transf_path)]

                            if level_avg in ['expgroup', 'subject']:
                                filelist_all.append(('\n'.join(filenames), filelist, transf_path))

                    if level_avg == 'subject':
                        filename_dst = os.path.join(subject_path_dst, F"{dir_expgroup}_{dir_subject}_{dir_stim}_avg.npy")
                        yield filename_dst, filelist_all
                        filelist_all = []

                if level_avg == 'expgroup':
                    filename_dst = os.path.join(expgroup_path_dst, F"{dir_expgroup}_{dir_stim}_avg.npy")
                    yield filename_dst, filelist_all
                    filelist_all = []

            if level_avg == 'all':
                filename_dst = os.path.join(exp_path_dst, F"{dir_stim}_avg.npy")
                yield filename_dst, filelist_all
                filelist_all = []


    def init_data_reduction_func(self) -> None:

        """
        Method for initializing the data reduction function depending on the 'reduction' param defined in the init of the class.
        """

        if self.reduction == 'median':
            self.red_func = lambda x: np.nanmedian(x, 0)

        elif self.reduction == 'mean':
            self.red_func = lambda x: np.nanmean(x, 0)

        elif self.reduction == 'single_trial':
            self.red_func = lambda x : x[0]

        else:
            print('Unrecognized reduction method: choose between mean, median, single_trial')
            sys.exit()


    def list_selected_data(self) -> str:

        """
        Method for generating the log file, containing all the info about the data that were selected and the processing that was applied.

        Parameters
        ----------
        data : ndarray
            Input data, 3D volume in time.
        outliers : ndarray
            Array of outliers as output by the 'get_outliers' method.

        Returns
        -------
        header : str
            A string containing the log.
        """

        txt_header = (
        F"{self.experiment_ID}_{self.ext}\n\n"
        F"INPUT PARAMETERS\n"
        F"root_folder = {self.root_folder}\n"
        F"experiment_ID = {self.experiment_ID}\n"
        F"expgroup_ID = {self.expgroup_ID}\n"
        F"subject_ID = {self.subject_ID}\n"
        F"session_ID = {self.session_ID}\n"
        F"stim_ID = {self.stim_ID}\n"
        F"mode = {self.mode}\n"
        F"level_avg = {self.level_avg}\n"
        F"reduction = {self.reduction}\n"
        F"baseline = {self.baseline}\n"
        F"register = {self.register}\n"
        F"atlas_resolution = {self.atlas_resolution}\n"
        F"make_reliability_maps = {self.make_reliability_maps}\n"
        F"remove_unreliable = {self.remove_unreliable}\n"
        F"trial_preprocessing = {self.trial_preprocessing}\n\n"
        F"LIST OF SELECTED FILES\n\n"
        )

        txt = []

        if self.mode == 'folder_tree':
            iterator = iter(self.data_iterator_folder_tree(level_avg='session'))
        elif self.mode == 'excel':
            iterator = iter(self.data_iterator_excel_file(level_avg='session'))
        else:
            pass

        for _, elt in iterator:
            info, transf = elt[0][0], elt[0][2]
            txt.append(F"{info}\nTransform matrix: {transf}\n")
        txt = '\n'.join(txt)

        return(txt_header+txt)


    def load_data_and_transf(
        self,
        filelist: list[str],
        transf_path: str,
        name: str,
        block_size: int = 5,
        trial_preprocessing = Union[None, object]
        ) -> (dict, dict):

        """
        Method for replacing the frames identified as outliers with an interpolation of neighbouring frames.

        Parameters
        ----------
        filelist : list of str
            List of paths to files to be loaded.
        transf_path : str
            Path to the list of files to be loaded.
        name : str
            Name of the output file. Only necessary if required by a trial selection object.
        block_size : int
            Used for low-memory option. NOT PROPERLY IMPLEMENTED YET.
        trial_preprocessing : None or object
            If None, no trial-based preprocessing. Otherwise, the '__call__' method of the object is applied to the list of data. See quality_control.OutlierFrameRemoval for an example.

        Returns
        -------
        data : dict
            A dictionary containing the reduced data, its size, voxel size and direction / anatomical orientation.
        M : ndarray
            The 4x4 transformation matrix.
        """

        #tmstp = time()
        data = []

        for k, file_path in enumerate(filelist):

            try:
                data_mat = loadmat(file_path, squeeze_me=True, simplify_cells=True)
            except NotImplementedError: # scipy loadmat does not support 7.3 files
                data_mat = mat73.loadmat(file_path)
                #data_mat['I'] = data_mat['I'][:,:,:,::2]

            ### data reshaping to get data in the order DV.AP.LR
            if 'size' in data_mat['md']:
                size_ = data_mat['md']['size'].astype('int').tolist()
                size_ = [size_[0], size_[2], size_[1]]
                data_shape = np.append(size_, data_mat['I'].shape[2])
                data_fus = data_mat['I'].reshape(data_shape).astype('float32')
            else:
                size_ = data_mat['md']['imageSize'].astype('int').tolist()
                size_ = [size_[i] for i in [0,2,1]]
                data_fus = np.swapaxes(data_mat['I'], 1, 2).astype('float32')

            data.append(data_fus)

        trials_len = [d.shape[-1] for d in data]
        assert trials_len.count(trials_len[0]) == len(trials_len), F"All trials must have the same lengths (issue with data {name}). Exiting..."

        if self.trial_preprocessing is not None:
            #tmstp_trial_selec = time()
            data = self.trial_preprocessing(data, name)
            #print(time()-tmstp_trial_selec)

        voxel_size = data_mat['md']['voxelSize'].tolist()
        direction = data_mat['md']['Direction']
        data = {'Data':self.red_func(data), 'Size':size_, 'VoxelSize':voxel_size, 'Direction':str(direction)}

        if transf_path is not None:
            try:
                tf_mat = loadmat(transf_path, squeeze_me=True)['Transf'].tolist()
                M = tf_mat[0]
            except NotImplementedError: # for fucking compatibility
                tf_mat = mat73.loadmat(transf_path)['Transf']
                M = tf_mat['M']
            # original estimate is made with 50µm CCF v3 atlas, so translation (last line) must adjusted if the atlas resolution changes
            M[3] /= (self.atlas_resolution/50)
            M[3,3] = 1

        else:
            M = None

        return(data, M)


    def normalize_data(
        self,
        data: np.ndarray
        ) -> np.ndarray:

        """
        Method for normalizing and centering the data using the self.baseline range.

        Parameters
        ----------
        data : ndarray
            The 4D data to be normalized.

        Returns
        -------
        data
            The normalized and centered data.
        """

        baseline = np.median(data[..., self.baseline], -1)
        baseline[baseline == 0] = 1
        data /= baseline[..., np.newaxis]
        data = (data-1).astype('float16')

        return(data)


    def process_data(
        self,
        print_reg_error: bool = True,
        block_size:int = 5
        ):

        """
        Main method for performing the data loading process.
        /!\\ maybe implement an iterative way of computing mean or median for heavy datasets

        Parameters
        ----------
        print_reg_error : bool
            If True, data that were skipped because no transformation matrix was found will be displayed in the terminal.
        block_size : int
            LOW-MEMORY IS NOT IMPLEMENTED YET SO NOT USED.

        """

        self.replicate_folder_tree_structure()
        txt = self.list_selected_data()
        with open(os.path.join(self.root_folder, 'loaded_data', F'{self.experiment_ID}_{self.ext}', 'info.txt'), 'w') as f:
            f.write(txt)

        if self.make_reliability_maps:
            _, _, regions_info_file = u.get_atlas_and_info_paths(self.atlas_resolution, '_nolayersnoparts')
            rm = qc.ReliabilityMaps(self.atlas, regions_info_file, self.atlas_contours, reliability_threshold=self.remove_unreliable)

        for dst, elt in self.iterator:

            avg, k, n_samples = [], 0, 0
            name = u.name_from_path(dst)

            for info, filelist, transf_path in elt:

                # to deal with the case of an empty folder
                if len(filelist) == 0:
                    continue

                tmstp = time()
                data, transf = self.load_data_and_transf(filelist, transf_path, name)
                n_samples += len(filelist)

                if self.register and transf is None:

                    if print_reg_error:
                        print(F"Warning: {info} was ignored during registration (no transf matrix).")
                    else:
                        pass

                else:

                    if self.register:
                        tmstp = time()
                        data = reg.register_data(self.atlas, data, transf, self.atlas_resolution)

                        if self.make_reliability_maps:
                            rm.compute_reliability_map(data)

                    else: ### this could me more elegant in the naming
                        data = data['Data']

                    if self.baseline is not None:
                        tmstp = time()
                        data = self.normalize_data(data)

                    avg.append(data)

            with open(os.path.join(self.root_folder, 'loaded_data', F'{self.experiment_ID}_{self.ext}', 'info.txt'), 'a') as f:
                f.write("\n{}   n_samples:   {}\n".format(dst.split('\\')[-1][:-4], n_samples))

            tmstp = time()
            avg = self.red_func(avg)

            if self.make_reliability_maps:
                rm.process(self.output_folder, name, plot=False)
                with open(os.path.join(self.root_folder, 'loaded_data', F'{self.experiment_ID}_{self.ext}', 'info.txt'), 'a') as f:
                    f.write(F"List of regions to exclude for region averaging analysis:")
                    f.write(F"\n{rm.list_excluded}\n")

                if self.remove_unreliable:
                    avg = rm.remove_unreliable(avg)

            np.save(dst, avg)

        print("All the data has been processed. Good luck with the analysis!")
