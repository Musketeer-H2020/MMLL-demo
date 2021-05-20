# -*- coding: utf-8 -*-
'''
@author:  Marcos Fernández Díaz
December 2020
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset <dataset> --id <id>

'''
import argparse
import json
import logging
import sys, os
# Add higher directory to python modules path.
sys.path.append("../../../../")

try:
    from MMLL.nodes.WorkerNode import WorkerNode
    from MMLL.common.MMLL_tools import display
    from MMLL.comms.comms_pycloudmessenger import Comms_worker as Comms
except Exception as err:
    if "No module named 'MMLL'" in str(err):
        print('\n' + 80 * '#')
        print('You need to install the MMLL library')
        print('pip install git+https://github.com/Musketeer-H2020/MMLL.git')
        print(80 * '#' + '\n')
    raise

from demo_tools.task_manager_pycloudmessenger import Task_Manager
from demo_tools.mylogging.logger_v1 import Logger
from demo_tools.data_connectors.Load_from_file_V import Load_From_File as DC_V
from demo_tools.evaluation_tools import create_folders


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
    parser.add_argument('--id', type=int, default=None, choices=[0, 1, 2, 3, 4], help='The address of the worker')
    parser.add_argument('--dataset', type=str, default=None, help='The dataset to be used')

    FLAGS, unparsed = parser.parse_known_args()
    user_name = FLAGS.user
    user_password = FLAGS.password
    task_name = FLAGS.task_name
    data_partition_id = FLAGS.id # This integer identifies the data partition used for the worker
    dataset_name = FLAGS.dataset

    # Set basic configuration
    pom = 1
    model_type = 'SVM'
    verbose = False


    # Create the directories for storing relevant outputs if they do not exist
    create_folders("./results/")

    # Setting up the logger
    logger = Logger('./results/logs/Worker_' + str(user_name) + '.log')


    # Load the credentials for pycloudmessenger
    display('===========================================', logger, verbose)
    display('Creating Worker...', logger, verbose)
    # Note: this part creates the worker (participant) and it joins the task. This code is
    # intended to be used only at the demos, in Musketeer this part must be done in the client. 
    credentials_filename = '../../musketeer.json'
    try:
        with open(credentials_filename, 'r') as f:
            credentials = json.load(f)
    except:
        display('Error - The file musketeer.json is not available, please put it under the following path: "' + os.path.abspath(os.path.join("","../../")) + '"', logger, verbose)
        sys.exit()

    # Create user and join task
    tm = Task_Manager(credentials_filename)
    participant = tm.create_worker_and_join_task(user_name, user_password, task_name, display, logger)
    display("Worker %s has joined task %s" %(user_name, task_name), logger, verbose)

    # Creating the comms object
    display('Creating WorkerNode under POM %d, communicating through pycloudmessenger' %pom, logger, verbose)
    comms = Comms(participant, user_name)
    
    # Creating Workernode
    wn = WorkerNode(pom, comms, logger, verbose)
    display('-------------------- Loading dataset %s --------------------------' % dataset_name, logger, verbose)

    # Load data
    # Warning: this data connector is only designed for the demos. In Musketeer, appropriate data
    # connectors must be provided
    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    try:
        dc = DC_V(data_file)
    except:
        display('Error - The file ' + dataset_name + '_demonstrator_data.pkl does not exist. Please download it from Box and put it under the following path: "' + os.path.abspath(os.path.join("","../../../../input_data/")) + '"', logger, verbose)
        sys.exit()

    # Get train and set training data
    [Xtr, ytr, input_data_description, target_data_description] = dc.get_data_train_Worker_V(int(data_partition_id))
    wn.set_training_data(dataset_name, Xtr, ytr, input_data_description, target_data_description)
    display('WorkerNode loaded %d patterns for train' % wn.NPtr, logger, verbose)


    # Creating a ML model and start training procedure
    wn.create_model_worker(model_type)
    display('MMLL model %s is ready for training!' %model_type, logger, verbose)
    display('Worker_' + model_type + ' %s is running...' %user_name, logger, verbose)
    wn.run()
    display('Worker_' + model_type + ' %s: EXIT' %user_name, logger, verbose)
