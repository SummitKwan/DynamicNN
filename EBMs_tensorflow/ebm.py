""" base class in this module """

import os
import re
import time
import h5py
import pickle

modelname = 'EBM'

class EnergyBasedModel():

    def read_parameters(self, filepathname=None,
                   filedir='../model_save', filename='', fileext='.h5', find_last=True):
        """
        read model parameters from a hdf5 (h5) or pickle (pkl) file
        :param filepathname: the complete file path and name of the file to store,
            if not given, generate using filedir, filename, fileext, find_last
        :param filedir:      directory of file
        :param filename:     name of file
        :param fileext:      extension of file, has to be either '.h5' or '.pkl'
        :param find_last:    whether to find the most recent model based on timecode
        :return:
        """

        if filepathname is not None:
            filedir, basename = os.path.split(filepathname)
            filename, fileext = os.path.splitext(basename)
        else:
            if find_last:
                files = os.listdir(filedir)
                file_keep = []
                date_keep = []
                for file in files:
                    if re.search(r'{}'.format(filename), file) and re.search(r'{}$'.format(fileext), file) \
                            and re.search('\d{8}_\d{6}', file):
                        timecode = re.search('\d{8}_\d{6}', file).group(0)
                        file_keep.append(file)
                        date_keep.append(timecode)
                filenameext = sorted(zip(date_keep, file_keep))[-1][1]
                filepathname = os.path.join(filedir, filenameext)
            else:
                filepathname = os.path.join(filedir, filename + fileext)

        print('reading model parameters from file {}'.format(filepathname))

        dict_params = dict()
        if fileext == '.h5':
            with h5py.File(filepathname, 'r') as f:
                for key in f.keys():
                    dict_params[key] = f.get(key).value
        elif fileext == '.pkl':
            with open(filepathname, 'rb') as f:
                dict_params = pickle.load(f)
        else:
            raise Exception('fileext must be one of (".h5", ".pkl")')

        return dict_params

    def write_parameters(self, dict_params, filepathname=None,
                   filedir='../model_save', filename=modelname, fileext='.h5', auto_timecode=True):
        """
        save model parameters to a hdf5 (h5) or pickle (pkl) file
        :param dict_params:   dictional of parameters to save
        :param filepathname:  the complete file path and name of the file to store,
            if not given, generate using filedir, filename, fileext, autotimecode
        :param filedir:       directory of file
        :param filename:      name of file
        :param fileext:       extension of file, has to be either '.h5' or '.pkl'
        :param auto_timecode: whether automatically add datetime code to the end filename, default to True
        :return:
        """

        if filepathname is not None:
            filedir, basename = os.path.split(filepathname)
            filename, fileext = os.path.splitext(basename)
        else:
            assert os.path.isdir(filedir),   'directory {} does not exist'.format(filedir)
            assert fileext in ('.h5', '.pkl'), 'fileext must be one of (".h5", ".pkl")'
            if auto_timecode:
                time_str = '_' + time.strftime("%Y%m%d_%H%M%S")
            else:
                time_str = ''

            filepathname = os.path.join(filedir, filename + time_str + fileext)

        print('writing model parameters to file {}'.format(filepathname))

        if fileext == '.h5':
            with h5py.File(filepathname, 'w') as f:
                for key in dict_params:
                    f.create_dataset(key, data=dict_params[key])
        elif fileext == '.pkl':
            with open(filepathname, 'wb') as f:
                pickle.dump(dict_params, f)
        else:
            raise Exception('fileext must be one of (".h5", ".pkl")')