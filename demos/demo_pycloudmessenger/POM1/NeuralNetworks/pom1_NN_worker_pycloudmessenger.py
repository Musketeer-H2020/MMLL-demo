# -*- coding: utf-8 -*-
'''
@author:  Marcos Fernandez Diaz
May 2020

Example of use: python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id <id>

Parameters:
    - user: String with the name of the user. If the user does not exist in the pycloudmessenger platform a new one will be created
    - password: String with the password
    - task_name: String with the name of the task. If the task already exists, an error will be displayed
    - id: Integer representing the partition of data to be used by the worker. Each worker should use a different partition, possible values are 0 to 4.

'''

# Import general modules
import argparse
import logging
import json
import numpy as np
import sys, os

# Add higher directory to python modules path.
sys.path.append("../../../../")

# To be imported from MMLL (pip installed)
from MMLL.nodes.WorkerNode import WorkerNode
from MMLL.comms.comms_pycloudmessenger import Comms_worker as Comms

# To be imported from demo_tools
from demo_tools.task_manager_pycloudmessenger import Task_Manager
from demo_tools.data_connectors.Load_from_file import Load_From_File as DC                          # Data connector
from demo_tools.mylogging.logger_v1 import Logger
from demo_tools.evaluation_tools import display, plot_cm_seaborn


# Set up logger
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--credentials', type=str, default=None, help='Credentials for Muskeeter Server')
    parser.add_argument('--user', type=str, default=None, help='User')
    parser.add_argument('--password', type=str, default=None, help='Password')
    parser.add_argument('--task_name', type=str, default=None, help='Name of the task')
    #parser.add_argument('--dataset', type=str, default=None, help='The file with the data')
    #parser.add_argument('--verbose', type=str, default=False, help='If true print the messages on the console')
    parser.add_argument('--id', type=int, default=None, help='The address of the worker')

    FLAGS, unparsed = parser.parse_known_args()
    user_name = FLAGS.user
    user_password = FLAGS.password
    task_name = FLAGS.task_name
    data_partition_id = FLAGS.id # This integer identifies the data partition used for the worker

    dataset_name = 'mnist'
    verbose = False
    comms_type = 'pycloudmessenger'
    pom = 1
    model_type = 'NN'


    # Create the directories for storing relevant outputs if they do not exist
    if not os.path.exists("../results/logs/"):
        os.makedirs("../results/logs/") # Create directory for the logs
    if not os.path.exists("../results/figures/"):
        os.makedirs("../results/figures/") # Create directory for the figures
    if not os.path.exists("../results/models/"):
        os.makedirs("../results/models/") # Create directory for the models


    # Setting up the logger    
    logger = Logger('../results/logs/pycloudmessenger_cloud_worker_' + str(user_name) + '.log')


    display('===========================================', logger, verbose)
    display('Creating Worker...', logger, verbose)
    # ==================================================
    # Note: this part creates the worker (participant) and it joins the task. This code is
    # intended to be used only at the demos, in Musketeer this part must be done in the client. 
    # ==================================================
    credentials_filename = '../../musketeer.json'
    try:
        with open(credentials_filename, 'r') as f:
            credentials = json.load(f)
    except:
        display('Error - The file musketeer.json is not available, please put it under the following path: "' + os.path.abspath(os.path.join("","../../")) + '"', logger, verbose)
        sys.exit()

    tm = Task_Manager(credentials_filename)
    # We need the participant to build comms object
    participant = tm.create_worker_and_join_task(user_name, user_password, task_name, display, logger)
    display("Worker %s has joined task %s" %(user_name, task_name), logger, verbose)
    # ==================================================


    display('Creating WorkerNode under POM1, communicating through pycloudmessenger', logger, verbose)
    #########################################
    # Creating Comms object, needed by MMLL
    comms = Comms(participant, user_name)
    #########################################
    
    # Creating Workernode
    wn = WorkerNode(pom, comms, logger, verbose)
    display('-------------------- Loading dataset %s --------------------------' % dataset_name, logger, verbose)

    # Warning: this data connector is only designed for the demos. In Musketeer, appropriate data
    # connectors must be provided
    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    try:
        dc = DC(data_file)
    except:
        display('Error - The file ' + dataset_name + '_demonstrator_data.pkl does not exist. Please download it from Box and put it under the following path: "' + os.path.abspath(os.path.join("","../../../../input_data/")) + '"', logger, verbose)
        sys.exit()
    [Xtr, ytr, Xval, yval, Xtst, ytst] = dc.get_all_data_Worker(int(data_partition_id))
    wn.set_training_data(dataset_name, Xtr, ytr)
    wn.set_validation_data(dataset_name, Xval, yval)
    wn.set_test_data(dataset_name, Xtst, ytst)
    display('WorkerNode loaded %d patterns for training' % wn.NPtr, logger, verbose)
    display('WorkerNode loaded %d patterns for validation' % wn.NPval, logger, verbose)
    display('WorkerNode loaded %d patterns for test' % wn.NPtst, logger, verbose)
    #########################################

    #---------------  Creating a ML model (Worker side) ---------------------  
    wn.create_model_worker(model_type)
    display('MMLL model %s is ready for training!' %model_type, logger, verbose)
    display('Worker_' + model_type + ' %s is running...' %user_name, logger, verbose)
    wn.run()
    display('Worker_' + model_type + ' %s: EXIT' %user_name, logger, verbose)

    # Retrieving and saving the trained model
    display('Retrieving the trained model from WorkerNode', logger, verbose)
    model = wn.get_model()
    
    # Warning: this save_model utility is only for demo purposes
    output_filename_model = '../results/models/POM' + str(pom) + '_' + model_type + '_worker_' + dataset_name + '_model.pkl'
    wn.save_model(output_filename_model)

    display('-------------  Obtaining predictions------------------------------------\n', logger, verbose)
    preds_tst = model.predict(Xtst)
    y = np.argmax(ytst, axis=-1) # Convert to labels
    classes = np.arange(ytst.shape[1]) # 0 to 9

    display('-------------  Evaluating --------------------------------------------\n', logger, verbose)
    # Warning, these evaluation methods are not part of the MMLL library, they are only intended
    # to be used for the demos. Use them at your own risk.
    plot_cm_seaborn(preds_tst, y, classes, 'NN confusion matrix in test set worker', model_type, dataset_name, logger, verbose, normalize=True)
