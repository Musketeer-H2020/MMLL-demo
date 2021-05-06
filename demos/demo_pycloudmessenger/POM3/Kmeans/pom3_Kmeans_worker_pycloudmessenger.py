# -*- coding: utf-8 -*-
'''
@author:  Marcos Fernandez Diaz
October 2020

Example of use: python pom3_Kmeans_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id <id>

Parameters:
    - user: String with the name of the user. If the user does not exist in the pycloudmessenger platform a new one will be created.
    - password: String with the password.
    - task_name: String with the name of the task.
    - id: Integer representing the partition of data to be used by the worker. Each worker should use a different partition, possible values are 0 to 4.

'''


# Import general modules
import argparse
import logging
import json
import sys, os

# Add higher directory to python modules path.
sys.path.append("../../../../")

# To be imported from MMLL (pip installed)
from MMLL.nodes.WorkerNode import WorkerNode
from MMLL.comms.comms_pycloudmessenger import Comms_worker as Comms

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
    parser.add_argument('--id', type=int, default=None, choices=[0, 1, 2, 3, 4], help='The data partition of the worker')

    FLAGS, unparsed = parser.parse_known_args()
    user_name = FLAGS.user
    user_password = FLAGS.password
    task_name = FLAGS.task_name
    data_partition_id = FLAGS.id # This integer identifies the data partition used for the worker

    # Set basic configuration
    dataset_name = 'pima'
    verbose = False
    pom = 3
    model_type = 'Kmeans'


    # Create the directories for storing relevant outputs if they do not exist
    create_folders("./results/")

    # Setting up the logger    
    logger = Logger('./results/logs/Worker_' + str(user_name) + '.log')

    # Load credentials file to use pycloudmessenger
    # Note: this part creates the task and waits for the workers to join. This code is intended to be used only at the demos, in Musketeer this part must be done in the client.
    display('===========================================', logger, verbose)
    credentials_filename = '../../musketeer.json'
    try:
        with open(credentials_filename, 'r') as f:
            credentials = json.load(f)
    except:
        display('Error - The file musketeer.json is not available, please put it under the following path: "' + os.path.abspath(os.path.join("","../../")) + '"', logger, verbose)
        sys.exit()

    # Create user and join the task
    tm = Task_Manager(credentials_filename)
    participant = tm.create_worker_and_join_task(user_name, user_password, task_name, display, logger)

    # Creating the comms object
    display('Creating WorkerNode under POM %d, communicating through pycloudmessenger' %pom, logger, verbose)
    comms = Comms(participant, user_name)
    
    # Creating Workernode
    wn = WorkerNode(pom, comms, logger, verbose)
    display('-------------------- Loading dataset %s --------------------------' % dataset_name, logger, verbose)

    # Load data
    # Warning: this data connector is only designed for the demos. In Musketeer, appropriate data connectors must be provided
    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    try:
        dc = DC(data_file)
    except:
        display('Error - The file ' + dataset_name + '_demonstrator_data.pkl does not exist. Please download it from Box and put it under the following path: "' + os.path.abspath(os.path.join("","../../../../input_data/")) + '"', logger, verbose)
        sys.exit()

    # Get train/test data and set training data
    [Xtr, ytr, _, _, Xtst, ytst] = dc.get_all_data_Worker(int(data_partition_id))
    wn.set_training_data(dataset_name, Xtr, ytr)
    display('WorkerNode loaded %d patterns for training' % wn.NPtr, logger, verbose)

    # Creating a ML model and start training procedure
    wn.create_model_worker(model_type)
    display('MMLL model %s is ready for training!' %model_type, logger, verbose)
    display('Worker_' + model_type + ' %s is running...' %user_name, logger, verbose)
    wn.run()
    display('Worker_' + model_type + ' %s: EXIT' %user_name, logger, verbose)

    # Retrieving and saving the trained model
    display('Retrieving the trained model from WorkerNode', logger, verbose)
    model = wn.get_model()    
    # Warning: this save_model utility is only for demo purposes
    output_filename_model = './results/models/Worker_' + str(user_name) + '_' + dataset_name + '_model.pkl'
    model.save(output_filename_model)

    # Making predictions on test data
    display('-------------  Obtaining predictions------------------------------------\n', logger, verbose)
    preprocessors = wn.get_preprocessors()
    if preprocessors is not None:
        for prep_model in preprocessors: # Apply stored preprocessor sequentially (in the same order received)
            Xtst = prep_model.transform(Xtst)
            display('Test data transformed using %s' %prep_model.name, logger, verbose)
    preds_tst = model.predict(Xtst)

    # Evaluating the results
    display('-------------  Evaluating --------------------------------------------\n', logger, verbose)
    # Warning, these evaluation methods are not part of the MMLL library, they are only intended to be used for the demos. Use them at your own risk.
    output_filename = 'Worker_' + str(user_name) + '_clusters_' + dataset_name + '.png'
    title = 'Kmeans clustering with 2 PCA components in test set worker'
    Kmeans_plot(Xtst, preds_tst, title, output_filename, logger, verbose)