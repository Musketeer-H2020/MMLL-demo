# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
Dec. 2020
Demo on feature selection using frequencies
python3 pom5_feature_selection_frequency.py --dataset news20_all_sparse --verbose 1

'''
import argparse
import time
import json
import sys, os
import numpy as np
import codecs
import random
import pickle

# Add higher directory to python modules path.
sys.path.append("../../../../")

try:
    from MMLL.nodes.MasterNode import MasterNode
    from MMLL.common.MMLL_tools import display
    #from MMLL.comms.comms_pycloudmessenger import Comms_master as Comms
    from MMLL.comms.comms_pycloudmessenger import Comms_master as Comms
    from demo_tools.crypto.crypt_PHE import Crypto as CR
except Exception as err:
    if "No module named 'MMLL'" in str(err):
        print('\n' + 80 * '#')
        print('You need to install the MMLL library')
        print('pip install git+https://github.com/Musketeer-H2020/MMLL.git')
        print(80 * '#' + '\n')
    raise

from demo_tools.task_manager_pycloudmessenger import Task_Manager
from demo_tools.mylogging.logger_v1 import Logger
from demo_tools.data_connectors.Load_from_file import Load_From_File as DC                          # Data connector
from demo_tools.evaluation_tools import eval_classification, create_folders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default=None, help='The external names of the workers')
    parser.add_argument('--verbose', type=str, default='1', help='Print messages on screen when True')
    parser.add_argument('--dataset', type=str, default=None, help='The dataset to be used')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.verbose == '1':
        verbose = True
    else:
        verbose = False

    # Create the directories for storing relevant outputs if they do not exist
    create_folders("./results/")

    # Logging is optional, if you do not want to log messages, simply set logger=None
    logger = Logger('./results/logs/Master.log')

    pom = 5
    Nworkers = 5
    model_type = 'LR'
    dataset_name = FLAGS.dataset

    display('===========================================', logger, True)
    display('Creating Master... ', logger, True)
    display('Please wait until Master is ready before launching the workers...', logger, True)
    # ==================================================
    # Note: this part creates the task and waits for the workers to join. This code is
    # intended to be used only at the demos, in Musketeer this part must be done in the client. 
    credentials_filename = '../../musketeer.json'
    tm = Task_Manager(credentials_filename)
    # We need the aggregator to build comms object
    aggregator = tm.create_master_random_taskname(pom, Nworkers, user_org='UC3M')   
    display('Workers can be launched now!', logger, True)
    display('Waiting for the workers to join task name = %s' % tm.task_name, logger, True)
    tm.wait_for_workers()
    # ==================================================    
    
    #########################################
    display('Creating MasterNode under POM5, communicating through pycloudmessenger', logger, True)
    # Creating Comms object, needed by MMLL
    comms = Comms(aggregator)

    # Defining encryption object
    key_size = 512
    cr = CR(key_size=key_size)

    # Creating Masternode
    mn = MasterNode(pom, comms, logger, verbose)
    display('-------------------- Loading dataset %s --------------------------' % dataset_name, logger, True)
    # Warning: this data connector is only designed for the demos. In Musketeer, appropriate data
    # connectors must be provided
    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    dc = DC(data_file)

    #########################################
    input_data_description = None
    if dataset_name == "news20_all_sparse":
        input_data_description = {
                            "NI": 1355191, 
                            "input_types": [
                            {"type": "num", "name": "bow"}
                            ] * 1355191
                            }

        target_data_description= {
        "NT": 1, 
        "output_types": [
        {"type": "bin", "name": "news_group", "definition": "type of news"}
        ]
        }


    #---------------  Creating a ML model (Master side) ---------------------  
    ########################################
    # Parameters depending on the model_type
    ########################################
    if input_data_description is not None:
        model_parameters = {}
        model_parameters.update({'regularization': 0.001})
        model_parameters.update({'Nmaxiter': 10})
        model_parameters.update({'conv_stop': 0.005})
        model_parameters.update({'cr': cr})
        model_parameters.update({'input_data_description': input_data_description})
        model_parameters.update({'aggregation_type': 'direct'})
        #model_parameters.update({'aggregation_type': 'roundrobin'})
    else:
        display('\n' + '='*50 + '\nERROR: input_data_description is missing\n' + '='*50 + '\n', logger, True)
        sys.exit()

    mn.create_model_Master(model_type, model_parameters=model_parameters)
    display('MMLL model %s is ready for training!' % model_type, logger, True)

    Max_freq = 0.90
    NF = 1000
    feature_extractor, new_input_data_description = mn.get_feat_freq_transformer(input_data_description, Max_freq, NF)

    display('Computing feature extraction for Xval...', logger, True)
    [Xval, yval] = dc.get_data_val()

    display('------- Sample before Feature Frequency Selection ---\n', logger, True)
    display(Xval[0, :], logger, True)
    display(Xval.shape, logger, True)

    Xval_FE = feature_extractor.transform(Xval)

    errors_RP = mn.preprocess_data_at_workers(feature_extractor)

    display('------- Sample after Feature Frequency Selection ---\n', logger, True)
    display(Xval_FE[0, :], logger, True)
    display(Xval_FE.shape, logger, True)

    display('\n---------------------------------------------------------------------', logger, True)
    display('----------------------- Feature Frequency Selection completed ---------', logger, True)
    display('----------------------------------------------------------------------\n', logger, True)

    display('Terminating all worker nodes.', logger, True)
    mn.terminate_workers()
