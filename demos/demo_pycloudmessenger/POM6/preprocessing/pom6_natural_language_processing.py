# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
November 2020
Demo on Natural Language Processing
python3 pom6_natural_language_processing.py --dataset 20news_bow_bin --verbose 1

'''
import argparse
import time
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

    pom = 6
    Nworkers = 5
    model_type = 'LC'
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
    display('Creating MasterNode under POM6, communicating through pycloudmessenger', logger, True)
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

    # Common vocabulary identification
    #https://www.mit.edu/~ecprice/wordlist.10000
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

    display('Computing Xval tfidf...', logger, True)
    [Xval, yval] = dc.get_data_val()

    display('------- Sample before Natural Language preprocessing (TFIDF) ---\n', logger, True)
    display(Xval[0, :], logger, True)
    display(Xval.shape, logger, True)

    Xval_tfidf = tfidf_extractor.transform(Xval)

    display('------- Sample after Natural Language preprocessing (TFIDF) ---\n', logger, True)
    display(Xval_tfidf[0, :], logger, True)
    display(Xval_tfidf.shape, logger, True)

    display('\n---------------------------------------------------------------------', logger, True)
    display('---------- Natural Language preprocessing (TFIDF) completed  -----------', logger, True)
    display('----------------------------------------------------------------------\n', logger, True)

    display('Terminating all worker nodes.', logger, True)
    mn.terminate_workers()

