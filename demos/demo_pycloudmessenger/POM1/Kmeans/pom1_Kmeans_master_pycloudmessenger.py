# -*- coding: utf-8 -*-
'''
@author:  Marcos Fernandez Diaz
November 2020

Example of use: python pom1_Kmeans_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --normalization <normalization>
    
Parameters:
    - user: String with the name of the user. If the user does not exist in the pycloudmessenger platform a new one will be created
    - password: String with the password
    - task_name: String with the name of the task. If the task already exists, an error will be displayed
    - normalization: String indicating wether to apply standard normalization. Possible options are std or minmax. By default no normalization is used.

'''

# Import general modules
import argparse
import logging
import json
import time
import sys, os
import pickle
import numpy as np
import onnxruntime as rt

# Add higher directory to python modules path.
sys.path.append("../../../../")

# To be imported from MMLL (pip installed)
from MMLL.nodes.MasterNode import MasterNode
from MMLL.comms.comms_pycloudmessenger import Comms_master as Comms

# To be imported from demo_tools
from demo_tools.task_manager_pycloudmessenger import Task_Manager
from demo_tools.data_connectors.Load_from_file import Load_From_File as DC
from demo_tools.mylogging.logger_v1 import Logger
from demo_tools.evaluation_tools import display, Kmeans_plot, create_folders


# Set up logger
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, default=None, help='User')
    parser.add_argument('--password', type=str, default=None, help='Password')
    parser.add_argument('--task_name', type=str, default=None, help='Name of the task')
    parser.add_argument('--normalization', type=str, default='no', choices=['no', 'std', 'minmax'], help='Type of normalization')

    FLAGS, unparsed = parser.parse_known_args()
    user_name = FLAGS.user
    user_password = FLAGS.password
    task_name = FLAGS.task_name
    normalization = FLAGS.normalization

    dataset_name = 'pima'
    verbose = False
    pom = 1
    model_type = 'Kmeans'
    Nworkers = 2


    # Create the directories for storing relevant outputs if they do not exist
    create_folders("./results/")

    # Setting up the logger    
    logger = Logger('./results/logs/Master.log')

    # Task definition
    task_definition = {"quorum": Nworkers, 
                       "POM": pom, 
                       "model_type": model_type, 
                       "NC": 2, 
                       "Nmaxiter": 2, 
                       "tolerance": 0.001
                      }

    # Load credentials file to use pycloudmessenger
    # Note: this part creates the task and waits for the workers to join. This code is intended to be used only at the demos, in Musketeer this part must be done in the client. 
    credentials_filename = '../../musketeer.json'
    try:
        with open(credentials_filename, 'r') as f:
            credentials = json.load(f)
    except:
        display('Error - The file musketeer.json is not available, please put it under the following path: "' + os.path.abspath(os.path.join("","../../")) + '"', logger, verbose)
        sys.exit()

    display('===========================================', logger, verbose)
    display('Creating Master... ', logger, verbose)
    display('Please wait until Master is ready before launching the workers...', logger, verbose)

    # Create task and wait for workers to join
    tm = Task_Manager(credentials_filename)
    aggregator = tm.create_master_and_taskname(display, logger, task_definition, user_name=user_name, user_password=user_password, task_name=task_name)
    display('Waiting for the workers to join task name = %s' % tm.task_name, logger, verbose)
    tm.wait_for_workers_to_join(display, logger)
    
    # Creating the Comms object, needed by MMLL
    display('Creating MasterNode under POM %d, communicating through pycloudmessenger' %pom, logger, verbose)
    comms = Comms(aggregator)

    # Creating Masternode
    mn = MasterNode(pom, comms, logger, verbose)
    display('-------------------- Loading dataset %s --------------------------' % dataset_name, logger, verbose)
 
    # Load data
    # Warning: this data connector is only designed for the demos. In Musketeer, appropriate data
    # connectors must be provided
    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    try:
        dc = DC(data_file)
    except:
        display('Error - The file ' + dataset_name + '_demonstrator_data.pkl does not exist. Please download it from Box and put it under the following path: "' + os.path.abspath(os.path.join("","../../../../input_data/")) + '"', logger, verbose)
        sys.exit()

    # Normalization definition needed for preprocessing
    number_inputs = 8
    feature_description = {"type": "num"}
    feature_array = [feature_description for index in range(number_inputs)]
    data_description = {
                        "NI": number_inputs, 
                        "input_types": feature_array
                        }

  
    # Creating a ML model
    model_parameters = {}
    model_parameters['NC'] = int(task_definition['NC'])
    model_parameters['Nmaxiter'] = int(task_definition['Nmaxiter'])
    model_parameters['tolerance'] = float(task_definition['tolerance'])
    mn.create_model_Master(model_type, model_parameters=model_parameters)
    display('MMLL model %s is ready for training!' % model_type, logger, verbose)

    # Normalization of data in each worker before training
    if normalization=='std':
        normalizer = mn.normalizer_fit_transform_workers(data_description, 'global_mean_std')
    elif normalization=='minmax':
        normalizer = mn.normalizer_fit_transform_workers(data_description, 'global_min_max')


    # Start the training procedure.
    display('Training the model %s' % model_type, logger, verbose)
    t_ini = time.time()
    mn.fit()
    t_end = time.time()
    display('Training is complete: Training time = %s seconds' % str(t_end - t_ini)[0:6], logger, verbose)

    # Retrieving and saving the final model
    display('Retrieving the trained model from MasterNode', logger, verbose)
    model = mn.get_model()    
    # Warning: this save_model utility is only for demo purposes
    output_filename_model = './results/models/Master_' + dataset_name + '_model.pkl'
    model.save(output_filename_model)

    # Making predictions on test data
    display('-------------  Obtaining predictions----------------------------------\n', logger, verbose)
    [Xtst, ytst] = dc.get_data_tst()
    if normalization != 'no':
        Xtst = normalizer.transform(Xtst)
    preds_tst = model.predict(Xtst)

    # Evaluating the results
    display('-------------  Evaluating --------------------------------------------\n', logger, verbose)
    # Warning, these evaluation methods are not part of the MMLL library, they are only intended
    # to be used for the demos. Use them at your own risk.
    output_filename = 'Master_clusters_' + dataset_name + '.png'
    title = 'Kmeans clustering with 2 PCA components in test set master'
    Kmeans_plot(Xtst, preds_tst, title, output_filename, logger, verbose)


    # Load pkl model and check results
    with open(output_filename_model, 'rb') as f:
        model = pickle.load(f)
    preds_tst = model.predict(Xtst)


    # Model export to ONXX
    output_filename_model = './results/models/Master_' + dataset_name + '_model.onnx'
    model.save(output_filename_model)

    # Compute the prediction with ONNX Runtime
    sess = rt.InferenceSession(output_filename_model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: Xtst.astype(np.float32)})[0]
    err_onnx = np.mean((preds_tst.ravel() - pred_onx.ravel())**2)
    display('Error in ONNX predictions is %f' %err_onnx, logger, verbose)


    # Model export to PMML
    output_filename_model = './results/models/Master_' + dataset_name + '_model.pmml'
    model.save(output_filename_model)


    # Terminate workers
    display('Terminating all worker nodes.', logger, verbose)
    mn.terminate_workers()

    display('----------------------------------------------------------------------', logger, verbose)
    display('------------------------- END MMLL Procedure -------------------------', logger, verbose)
    display('----------------------------------------------------------------------\n', logger, verbose)
