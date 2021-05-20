# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
Nov 2020
python3 pom6_data_alignment_estimation.py --dataset income_raw --verbose 1

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
    if dataset_name in ["income_raw", "income_raw_bad24"]:
        input_data_description = {
                            "NI": 14, 
                            "input_types": [
                            {"type": "num", "name": "age"},
                            {"type": "cat", "name": "workclass",  "values": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked", "?"]},
                            {"type": "num", "name": "fnlwgt"},
                            {"type": "cat", "name": "education",  "values": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool", "?"]},
                            {"type": "num", "name": "education-num"},
                            {"type": "cat", "name": "marital-status",  "values": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse", "?"]},
                            {"type": "cat", "name": "occupation",  "values": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "?"]},
                            {"type": "cat", "name": "relationship",  "values": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried", "?"]},
                            {"type": "cat", "name": "race",  "values": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black", "?"]},
                            {"type": "bin", "name": "sex"},          # binary cat are transformed to bin
                            {"type": "num", "name": "capital-gain"},
                            {"type": "num", "name": "capital-loss"},
                            {"type": "num", "name": "hours-per-week"},
                            {"type": "cat", "name": "native-country",  "values": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?"]}
                            ]
                            }

        target_data_description= {
        "NT": 1, 
        "output_types": [
        {"type": "bin", "name": "income", "definition": ">50K, <=50K"}
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

    ##############################################################################
    ######## Data preprocessing prior to DV estimation: convert to numeric  ######
    ##############################################################################
    preprocessor, new_input_data_description, worker_errors = mn.data2num_transform_workers(input_data_description)
    
    # check
    [Xval, yval] = dc.get_data_val()

    Xval = preprocessor.transform(Xval)
    stats_list = ['rxy', 'meanx', 'staty']
    ref_stats_dict = mn.compute_statistics(Xval, yval, stats_list)

    stats_dict_workers = mn.get_statistics_workers(stats_list)

    for worker in list(stats_dict_workers.keys()):
        display('=' * 50, logger, True)
        display('Data alignment for worker %s:' % worker, logger, True)
        for stat in stats_list:
            da = np.dot(ref_stats_dict[stat].ravel(), stats_dict_workers[worker][stat].ravel())        
            display('\tAlignment in %s is %f:' % (stat, da), logger, True)

    display('=' * 50, logger, True)

    display('\n---------------------------------------------------------------------', logger, True)
    display('---------- Data alignment estimation completed  ----------------------', logger, True)
    display('----------------------------------------------------------------------\n', logger, True)

    display('Terminating all worker nodes.', logger, True)
    mn.terminate_workers()

