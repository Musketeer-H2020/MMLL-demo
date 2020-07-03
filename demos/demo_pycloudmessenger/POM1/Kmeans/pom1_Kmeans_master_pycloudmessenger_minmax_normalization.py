# -*- coding: utf-8 -*-
'''
@author:  Marcos Fernandez Diaz
May 2020

Example of use: pom1_Kmeans_master_pycloudmessenger_minmax_normalization.py --user <user> --password <password> --task_name <task_name>
    
Parameters:
    - user: String with the name of the user. If the user does not exist in the pycloudmessenger platform a new one will be created
    - password: String with the password
    - task_name: String with the name of the task. If the task already exists, an error will be displayed

'''

# Import general modules
import argparse
import logging
import json
import time
import sys, os

# Add higher directory to python modules path.
sys.path.append("../../../../")

# To be imported from MMLL (pip installed)
from MMLL.nodes.MasterNode import MasterNode
from MMLL.comms.comms_pycloudmessenger import Comms_master as Comms

# To be imported from demo_tools
from demo_tools.task_manager_pycloudmessenger import Task_Manager
from demo_tools.data_connectors.Load_from_file import Load_From_File as DC                          # Data connector
from demo_tools.mylogging.logger_v1 import Logger
from demo_tools.evaluation_tools import display, Kmeans_plot


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

    FLAGS, unparsed = parser.parse_known_args()
    user_name = FLAGS.user
    user_password = FLAGS.password
    task_name = FLAGS.task_name

    dataset_name = 'pima'
    verbose = False
    comms_type = 'pycloudmessenger'
    pom = 1
    model_type = 'Kmeans'
    Nworkers = 2

    # Task definition
    task_definition = {"quorum": Nworkers, 
                       "POM": pom, 
                       "model_type": model_type,
                       "NC": 2, 
                       "Nmaxiter": 2, 
                       "tolerance": 0.001
                      }

    
    # Create the directories for storing relevant outputs if they do not exist
    if not os.path.exists("../results/logs/"):
        os.makedirs("../results/logs/") # Create directory for the logs
    if not os.path.exists("../results/figures/"):
        os.makedirs("../results/figures/") # Create directory for the figures
    if not os.path.exists("../results/models/"):
        os.makedirs("../results/models/") # Create directory for the models


    # Setting up the logger
    logger = Logger('../results/logs/pycloudmessenger_cloud_master_' + str(user_name) + '.log')


    display('===========================================', logger, verbose)
    display('Creating Master... ', logger, verbose)
    display('Please wait until Master is ready before launching the workers...', logger, verbose)
    # ==================================================
    # Note: this part creates the task and waits for the workers to join. This code is
    # intended to be used only at the demos, in Musketeer this part must be done in the client. 
    credentials_filename = '../../musketeer.json'
    try:
        with open(credentials_filename, 'r') as f:
            credentials = json.load(f)
    except:
        display('Error - The file musketeer.json is not available, please put it under the following path: "' + os.path.abspath(os.path.join("","../../")) + '"', logger, verbose)
        sys.exit()

    tm = Task_Manager(credentials_filename)
    # We need the aggregator to build comms object
    aggregator = tm.create_master_and_taskname(display, logger, task_definition, user_name=user_name, user_password=user_password, task_name=task_name)
    display('Waiting for the workers to join task name = %s' % tm.task_name, logger, verbose)
    tm.wait_for_workers_to_join(display, logger)
    # ==================================================
    
    #########################################
    display('Creating MasterNode under POM1, communicating through pycloudmessenger', logger, verbose)
    # Creating Comms object, needed by MMLL
    comms = Comms(aggregator)

    # Creating Masternode
    mn = MasterNode(pom, comms, logger, verbose)
    display('-------------------- Loading dataset %s --------------------------' % dataset_name, logger, verbose)
    # Warning: this data connector is only designed for the demos. In Musketeer, appropriate data
    # connectors must be provided
    data_description = {
                        "NI": 8, 
                        "input_types": [
                        {"type": "num"},
                        {"type": "num"},
                        {"type": "num"},
                        {"type": "num"},
                        {"type": "num"},
                        {"type": "num"},
                        {"type": "num"},
                        {"type": "num"}
                        ]
                        }

    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    try:
        dc = DC(data_file)
    except:
        display('Error - The file ' + dataset_name + '_demonstrator_data.pkl does not exist. Please download it from Box and put it under the following path: "' + os.path.abspath(os.path.join("","../../../../input_data/")) + '"', logger, verbose)
        sys.exit()

  
    #---------------  Creating a ML model (Master side) ---------------------  
    ########################################
    # Parameters depending on the model_type
    ########################################
    model_parameters = {}
    model_parameters['NC'] = int(task_definition['NC'])
    model_parameters['Nmaxiter'] = int(task_definition['Nmaxiter'])
    model_parameters['tolerance'] = float(task_definition['tolerance'])
    model_parameters.update({'data_description': data_description})

    mn.create_model_Master(model_type, model_parameters=model_parameters)
    display('MMLL model %s is ready for training!' % model_type, logger, verbose)


    ##############################################################################
    ######## Data normalization prior to training ################################
    ##############################################################################
    # We normalize the data before training
    # normalizes workers data and returns the normalizer object
    normalizer = mn.normalizer_fit_transform_workers(data_description, 'global_min_max')
        

    # We start the training procedure.
    display('Training the model %s' % model_type, logger, verbose)
    t_ini = time.time()
    mn.fit()
    t_end = time.time()
    display('Training is complete: Training time = %s seconds' % str(t_end - t_ini)[0:6], logger, verbose)
    display('----------------------------------------------------------------------', logger, verbose)

    display('Retrieving the trained model from MasterNode', logger, verbose)
    model = mn.get_model()
    
    # Warning: this save_model utility is only for demo purposes
    output_filename_model = '../results/models/POM' + str(pom) + '_' + model_type + '_master_' + dataset_name + '_model.pkl'
    mn.save_model(output_filename_model)

    display('-------------  Obtaining predictions----------------------------------\n', logger, verbose)
    [Xtst, ytst] = dc.get_data_tst()
    Xtst = normalizer.transform(Xtst)
    preds_tst = model.predict(Xtst)

    display('-------------  Evaluating --------------------------------------------\n', logger, verbose)
    # Warning, these evaluation methods are not part of the MMLL library, they are only intended
    # to be used for the demos. Use them at your own risk.
    Kmeans_plot(Xtst, preds_tst, 'Kmeans clustering with 2 PCA components in test set', model_type, dataset_name, logger, verbose)

    display('Terminating all worker nodes.', logger, verbose)
    mn.terminate_Workers()

    display('----------------------------------------------------------------------', logger, verbose)
    display('------------------------- END MMLL Procedure -------------------------', logger, verbose)
    display('----------------------------------------------------------------------\n', logger, verbose)