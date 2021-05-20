# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
Dec. 2020
python3 pom4_image_reshape_preprocessing.py --dataset mnist_raw_matrix_binclass --verbose 1

'''
import argparse
import time
import json
import sys, os
import numpy as np

# Add higher directory to python modules path.
sys.path.append("../../../../")

try:
    from MMLL.nodes.MasterNode import MasterNode
    from MMLL.common.MMLL_tools import display
    #from MMLL.comms.comms_pycloudmessenger import Comms_master as Comms
    from MMLL.comms.comms_pycloudmessenger import Comms_master as Comms

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

    pom = 4
    Nworkers = 5 + 1
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
    display('Creating MasterNode under POM4, communicating through pycloudmessenger', logger, True)
    # Creating Comms object, needed by MMLL
    comms = Comms(aggregator)
    # Creating Masternode
    mn = MasterNode(pom, comms, logger, verbose)
    display('-------------------- Loading dataset %s --------------------------' % dataset_name, logger, True)
    # Warning: this data connector is only designed for the demos. In Musketeer, appropriate data
    # connectors must be provided
    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    dc = DC(data_file)

    #########################################
    input_data_description = None
    if dataset_name == "mnist_raw_matrix_binclass":
        input_data_description = {
                            "NI": 1, 
                            "input_types": [
                            {"type": "matrix", "name": "image"}]
                            }

        target_data_description= {
        "NT": 1, 
        "output_types": [
        {"type": "bin", "name": "even/odd", "definition": "if a number is even or odd"}
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
        model_parameters.update({'input_data_description': input_data_description})
        model_parameters.update({'aggregation_type': 'direct'})
        #model_parameters.update({'aggregation_type': 'roundrobin'})
    else:
        display('\n' + '='*50 + '\nERROR: input_data_description is missing\n' + '='*50 + '\n', logger, True)
        sys.exit()

    mn.create_model_Master(model_type, model_parameters=model_parameters)
    display('MMLL model %s is ready for training!' % model_type, logger, True)

    [Xval, yval] = dc.get_data_val()
    display('No. images = %d, Image size = %dx%d' % (Xval.shape[0], Xval.shape[2], Xval.shape[3]), logger, True)

    # Image reshaping
    from MMLL.preprocessors.image_reshape import image_reshape_model
    image_reshape_tranformer =   image_reshape_model(input_data_description, 50, 40)
    display('Preprocessing data at workers with image_reshape', logger, True)
    errors_image_reshape = mn.preprocess_data_at_workers(image_reshape_tranformer)

    Xval_reshape = image_reshape_tranformer.transform(Xval)

    display('No. images = %d, Image size = %dx%d' % (Xval_reshape.shape[0], Xval_reshape.shape[2], Xval_reshape.shape[3]), logger, True)

    from MMLL.preprocessors.image_to_vector import image_to_vector_model
    image_to_vector_tranformer =   image_to_vector_model(input_data_description)
    display('Preprocessing data at workers with image_to_vector', logger, True)
    errors_image_to_vector = mn.preprocess_data_at_workers(image_to_vector_tranformer)

    Xval_tovector, new_input_data_description = image_to_vector_tranformer.transform(Xval_reshape)

    display('No. images = %d, Image size = %dx%d' % (Xval_tovector.shape[0], 1, Xval_tovector.shape[1]), logger, True)

    display('\n---------------------------------------------------------------------', logger, True)
    display('----------------------- Image reshaping Preprocessing completed ---------', logger, True)
    display('----------------------------------------------------------------------\n', logger, True)

    display('Terminating all worker nodes.', logger, True)
    mn.terminate_workers()
