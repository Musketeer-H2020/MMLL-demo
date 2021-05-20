# -*- coding: utf-8 -*-
'''
A data connector that loads data from a file. Especific for the demos. Vertical partition.'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Nov. 2020"

import pickle


class Load_From_File():
    """
    This class implements a data connector, that loads the data from a file. This connector is specific for the Musketeer Library demonstration examples.
    """

    def __init__(self, data_file):
        """
        Create a :class:`Load_From_File` instance.

        Parameters
        ----------
        filename : string
            path + filename to the file containing the data for master and workers.

        """
        self.data_file = data_file
        with open(self.data_file, 'rb') as f:
            [self.Xtr_chunks, self.ytr_chunks, self.Xval, self.yval, self.Xtst, self.ytst, self.input_data_description_chunks, self.target_data_description_chunks] = pickle.load(f)


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
        return self.Xtr_chunks[kworker], self.ytr_chunks[kworker]

    def get_data_train_Worker_V(self, kworker):
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
        return self.Xtr_chunks[kworker], self.ytr_chunks[kworker], self.input_data_description_chunks[kworker], self.target_data_description_chunks[kworker]


    def get_all_data_Worker(self, kworker):
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
        return self.Xtr_chunks[kworker], self.ytr_chunks[kworker], self.Xval, self.yval, self.Xtst, self.ytst
