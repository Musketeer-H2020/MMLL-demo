# -*- coding: utf-8 -*-
'''
@author:  Marcos Fernández Díaz
December 2020
python pom2_natural_language_processing.py --user <user> --password <password> --task_name <task_name> --dataset <dataset>

'''
import argparse
import time
import logging
import json
import sys, os
import numpy as np
import codecs
import random

# Add higher directory to python modules path.
sys.path.append("../../../../")

try:
    from MMLL.nodes.MasterNode import MasterNode
    from MMLL.common.MMLL_tools import display
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
from demo_tools.data_connectors.Load_from_file import Load_From_File as DC
from demo_tools.evaluation_tools import eval_classification, create_folders


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
    parser.add_argument('--dataset', type=str, default=None, help='The dataset to be used')

    FLAGS, unparsed = parser.parse_known_args()
    user_name = FLAGS.user
    user_password = FLAGS.password
    task_name = FLAGS.task_name
    dataset_name = FLAGS.dataset

    # Set basic configuration
    pom = 2
    Nworkers = 2
    model_type = 'SVM'
    verbose = False


    # Create the directories for storing relevant outputs if they do not exist
    create_folders("./results/")

    # Logging is optional, if you do not want to log messages, simply set logger=None
    logger = Logger('./results/logs/Master_' + str(user_name) + '.log')


    # Task definition
    task_definition = {"quorum": Nworkers, 
                       "POM": pom, 
                       "model_type": model_type, 
                       "NC": 200, 
                       "Nmaxiter": 6, 
                       "tolerance": 0.0001,
                       "sigma": 2.55,
                       "C": 1,
                       "NmaxiterGD": 20,
                       "eta": 0.05
                      }


    # Load the credentials for pycloudmessenger
    display('===========================================', logger, verbose)
    display('Creating Master... ', logger, verbose)
    display('Please wait until Master is ready before launching the workers...', logger, verbose)
    # Note: this part creates the task and waits for the workers to join. This code is
    # intended to be used only at the demos, in Musketeer this part must be done in the client.
    credentials_filename = '../../musketeer.json'
    try:
        with open(credentials_filename, 'r') as f:
            credentials = json.load(f)
    except:
        display('Error - The file musketeer.json is not available, please put it under the following path: "' + os.path.abspath(os.path.join("","../../")) + '"', logger, verbose)
        sys.exit()

    # Create task and wait for participants to join
    tm = Task_Manager(credentials_filename)
    aggregator = tm.create_master_and_taskname(display, logger, task_definition, user_name=user_name, user_password=user_password, task_name=task_name)   
    display('Waiting for the workers to join task name = %s' % tm.task_name, logger, verbose)
    tm.wait_for_workers_to_join(display, logger)
   
    
    # Creating the comms object
    display('Creating MasterNode under POM %d, communicating through pycloudmessenger' %pom, logger, verbose)
    comms = Comms(aggregator)

    # Creating Masternode
    mn = MasterNode(pom, comms, logger, verbose)
    display('-------------------- Loading dataset %s --------------------------' %dataset_name, logger, verbose)

    # Load data
    # Warning: this data connector is only designed for the demos. In Musketeer, appropriate data
    # connectors must be provided
    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    try:
        dc = DC(data_file)
    except:
        display('Error - The file ' + dataset_name + '_demonstrator_data.pkl does not exist. Please download it from Box and put it under the following path: "' + os.path.abspath(os.path.join("","../../../../input_data/")) + '"', logger, verbose)
        sys.exit()


    # Input and output data description needed for preprocessing
    input_data_description = None
    if dataset_name == "20news_bow_bin":
        input_data_description = {
                            "NI": 1, 
                            "input_types": [
                            {"type": "bow", "name": "bow"}
                            ]
                            }

        target_data_description= {
        "NT": 1, 
        "output_types": [
        {"type": "bin", "name": "tech_group", "definition": "identifies if the topic is about technology"}
        ]
        }


    # Creating a ML model
    if input_data_description is not None:
        model_parameters = {}
        model_parameters['NC'] = int(task_definition['NC'])
        model_parameters['Nmaxiter'] = int(task_definition['Nmaxiter'])
        model_parameters['tolerance'] = float(task_definition['tolerance'])
        model_parameters['sigma'] = float(task_definition['sigma'])
        model_parameters['C'] = float(task_definition['C'])
        model_parameters['NmaxiterGD'] = int(task_definition['NmaxiterGD'])
        model_parameters['eta'] = float(task_definition['eta'])
        mn.create_model_Master(model_type, model_parameters=model_parameters)
        display('MMLL model %s is ready for training!' % model_type, logger, verbose)
    else:
        display('ERROR: input_data_description is missing', logger, verbose)
        sys.exit()


    # Common vocabulary identification
    # https://www.mit.edu/~ecprice/wordlist.10000
    # https://github.com/dwyl/english-words/blob/master/words_alpha.txt

    vocab_file = '../../../../input_data/wordlist_10000.txt'
    with codecs.open(vocab_file, 'r') as f:
        words = f.readlines()
    words = [w.replace('\n', '') for w in words]
    random.shuffle(words)
    selected_words = words[0:1000]
    init_vocab_dict = {}
    for word in selected_words:
        init_vocab_dict.update({word: int(np.random.uniform(1000, 2000))})

    vocabulary, df_dict = mn.get_vocabulary_workers(input_data_description, init_vocab_dict) 
    # pack in a single call???
    # [tfidf_extractor, new_input_data_description, errors_tfidf_workers] = mn.get_tfidf_workers(vocabulary, df_dict, input_data_description)

    from MMLL.preprocessors.tfidf_matrix import tfidf_matrix_model
    tfidf_extractor = tfidf_matrix_model(vocabulary, df_dict, input_data_description)
    new_input_data_description  = tfidf_extractor.new_input_data_description
    errors_tfidf = mn.preprocess_data_at_workers(tfidf_extractor)

    display('Computing Xval tfidf...', logger, verbose)
    [Xval, yval] = dc.get_data_val()
    display('------- Sample before Natural Language preprocessing (TFIDF) ---', logger, verbose)
    display(Xval[0, :], logger, verbose)
    display('Shape of original data: %s' %str(Xval.shape), logger, verbose)

    Xval_tfidf = tfidf_extractor.transform(Xval)
    display('------- Sample after Natural Language preprocessing (TFIDF) ---', logger, verbose)
    display(Xval_tfidf[0, :], logger, verbose)
    display('Shape after NLP: %s' %str(Xval_tfidf.shape), logger, verbose)

    display('---------------------------------------------------------------------', logger, verbose)
    display('--------- Natural language preprocessing (TFIDF) completed  ---------', logger, verbose)
    display('---------------------------------------------------------------------\n', logger, verbose)

    display('Terminating all worker nodes.', logger, verbose)
    mn.terminate_workers()

