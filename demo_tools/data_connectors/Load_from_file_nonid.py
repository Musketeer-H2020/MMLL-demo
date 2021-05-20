# -*- coding: utf-8 -*-
'''
A data connector that loads data from a file. Especific for the non id experiments.'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Mar. 2021"

import pickle


class Load_From_File_master():
    """
    This class implements a data connector, that loads the data from a file. This connector is specific for the Musketeer Library demonstration examples.
    """

    def __init__(self, data_path, dataset):
        """
        Create a :class:`Load_From_File` instance.

        Parameters
        ----------
        filename : string
            path + filename to the file containing the data for master and workers.

        """
        data_file = data_path + dataset + '_val_nonid.pkl'
        with open(data_file, 'rb') as f:
            [self.Xval, self.yval] = pickle.load(f)

        data_file = data_path + dataset + '_tst_nonid.pkl'
        with open(data_file, 'rb') as f:
            [self.Xtst, self.ytst] = pickle.load(f)


    def get_data_val(self):
        """
        Obtains validation and test data, to be used by the master.

        Parameters
        ----------
        None

        Returns
        -------
        Xval: ndarray
            2-D array containing the validation patterns, one pattern per row

        yval: ndarray
            1-D array containing the validation targets, one target per row

        """
        return self.Xval, self.yval

    def get_data_tst(self):
        """
        Obtains validation and test data, to be used by the master.

        Parameters
        ----------
        None

        Returns
        -------
        Xtst: ndarray
            2-D array containing the test patterns, one pattern per row

        ytst: ndarray
            1-D array containing the test targets, one target per row

        """
        return self.Xtst, self.ytst


class Load_From_File_worker():
    """
    This class implements a data connector, that loads the data from a file. This connector is specific for the Musketeer Library demonstration examples.
    """

    def __init__(self, data_path, dataset, Nworkers, kworker):
        """
        Create a :class:`Load_From_File` instance.

        Parameters
        ----------
        filename : string
            path + filename to the file containing the data for master and workers.

        """
        data_file = data_path + dataset + '_' + str(Nworkers) + '_' + str(kworker) + '_tr' + '_nonid.pkl'
        with open(data_file, 'rb') as f:
            [self.Xtr, self.ytr] = pickle.load(f)


    def get_data_train_Worker(self, kworker):
        """
        Obtains training data at a given worker

        Parameters
        ----------
        kworker: integer
            number of the worker to be read data for.

        Returns
        -------
        Xtr: ndarray
            2-D array containing the training patterns, one pattern per row

        ytr: ndarray
            1-D array containing the training targets, one target per row

        """
        return self.Xtr, self.ytr
