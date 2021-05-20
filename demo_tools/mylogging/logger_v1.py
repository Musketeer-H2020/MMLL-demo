# -*- coding: utf-8 -*-
'''
A logger.'''

__author__ = "Angel Navia-Vázquez"
__date__ = "June 2019"

import logging


class Logger():
    """
    This class implements the logging facilities as well as some print methods.
    """

    def __init__(self, output_filename):
        """
        Create a :class:`Logger` instance.

        Parameters
        ----------
        output_filename : string
            path + filename to the file containing the output logs

        """
        self.logger = logging.getLogger()            # logger
        fhandler = logging.FileHandler(filename=output_filename, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        self.logger.addHandler(fhandler)
        self.logger.setLevel(logging.INFO)

    def display(self, message, verbose=True):
        """
        Display on screen if verbose=True and prints to the log file

        Parameters
        ----------
        verbose : Boolean
            prints to screen if True
        """
        if verbose:
            print(message)
        self.logger.info(message)

    def info(self, message):
        self.logger.info(message)
