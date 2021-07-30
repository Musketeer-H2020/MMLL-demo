# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
Jan 2021
python3 pom6_MBSVM_master_pycloudmessenger.py --dataset M-Iris --verbose 1

'''
import argparse
import time
import json
import sys, os
import numpy as np
import pickle

# Add higher directory to python modules path.
sys.path.append("../../../../")

try:
    from MMLL.nodes.MasterNode import MasterNode
    from MMLL.common.MMLL_tools import display, estimate_centroids
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
from demo_tools.evaluation_tools import eval_multiclass_classification, create_folders

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
    model_type = 'MBSVM'
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

    # We create centroids at random, and filter them
    NC = 80
    NI = 4
    NCcandidates = 10 * NC
    NCini = NC
    minvalue = 0
    maxvalue = 8.0
    C = estimate_centroids(NCini, NI, NCcandidates, minvalue, maxvalue, False)
    
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

    input_data_description = None
    if dataset_name in ['M-mnist-dlp100', 'M-mnist-dlp100-small']:
        input_data_description = {
                    "NI": 100, 
                    "input_types": [{"type": "num"}] * 100
                    }
        target_data_description = {
                                "NT": 1, 
                                "output_type": [
                                {"type": "cat", "values": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]},
                                ]
                                }    

    if dataset_name == 'M-iris':
        input_data_description = {
                    "NI": 4, 
                    "input_types": [{"type": "num"}] * 4
                    }
        target_data_description = {
                                "NT": 1, 
                                "output_type": [
                                {"type": "cat", "values": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]},
                                ]
                                }    


    #---------------  Creating a ML model (Master side) ---------------------  
    ########################################
    # Parameters depending on the model_type
    ########################################
    if dataset_name == 'M-iris':
        fsigma = 1.0
        Nmaxiter = 10
        Csvm = 10
        landa = 0.5
    if dataset_name == 'M-mnist-dlp100':
        fsigma = 0.2
        Nmaxiter = 20
        Csvm = 10
        landa = 0.5
    if dataset_name == 'M-mnist-dlp100-small':
        fsigma = 0.2
        Nmaxiter = 20
        Csvm = 10
        landa = 0.5

    if input_data_description is not None and target_data_description is not None:
        model_parameters = {}
        model_parameters.update({'Nmaxiter': Nmaxiter})
        model_parameters.update({'conv_stop': 0.01})
        model_parameters.update({'input_data_description': input_data_description})
        model_parameters.update({'target_data_description': target_data_description})
        model_parameters.update({'C': C})
        model_parameters.update({'NC': NC})
        model_parameters.update({'fsigma': fsigma})
        model_parameters.update({'Csvm': Csvm})
        model_parameters.update({'conv_stop': 0.01})
        model_parameters.update({'landa': landa})
    else:
        display('\n' + '='*50 + '\nERROR: input_data_description or target_data_description is missing\n' + '='*50 + '\n', logger, True)
        sys.exit()

    mn.create_model_Master(model_type, model_parameters=model_parameters)
    display('MMLL model %s is ready for training!' % model_type, logger, True)

    # We start the training procedure.
    display('Training the model %s' % model_type, logger, True)
    t_ini = time.time()

    # We can train the model without a validation set: 
    #mn.fit()

    # OR, if we can provide a validation set to Masternode, training can usually speed up
    [Xval, yval] = dc.get_data_val()
    mn.fit(Xval=Xval, yval=yval)

    t_end = time.time()
    display('Training is complete: Training time = %s seconds' % str(t_end - t_ini)[0:6], logger, True)
    display('----------------------------------------------------------------------', logger, True)

    if mn.model_is_trained:
        display('Retrieving the trained model from MasterNode', logger, True)
        model = mn.get_model()
        
        # Saving the model
        output_filename_model = './results/models/POM' + str(pom) + '_' + model_type + '_' + dataset_name + '_model.pkl'
        #output_filename_model = mn.save_model(output_filename_model)
        model.save(output_filename_model)

        display('-------------  Evaluating --------------------------------------------\n', logger, True)
        # Warning, these evaluation methods are not part of the MMLL library, they are only intended
        # to be used for the demos. Use them at your own risk.   

        # We check the saved model
        display('Loading the saved model', logger, True)
        with open(output_filename_model, 'rb') as f:
            model_loaded = pickle.load(f)

        display('-------------  Evaluating --------------------------------------------\n', logger, True)
        # Warning, these evaluation methods are not part of the MMLL library, they are only intended
        # to be used for the demos. Use them at your own risk.   

        display('-------------  Obtaining predictions------------------------------------\n', logger, True)
        
        try:
            [Xval, yval] = dc.get_data_val()
            preds_val = model_loaded.predict(Xval)
            e_val = np.array(yval.ravel() != np.array(preds_val).ravel()).astype(float)
            CE_val = np.mean(e_val) * 100.0
            display('Master_' + model_type + ': CE(%%) on validation set =  %s' % str(CE_val)[0:6], logger, True)
        except:
            raise
            '''
            preds_val = None
            print('ERROR while computing predictions on validation data')
            print('STOP AT ')
            import code
            code.interact(local=locals())
            '''
        try:
            [Xtst, ytst] = dc.get_data_tst()
            preds_tst = model_loaded.predict(Xtst)
            e_tst = np.array(ytst.ravel() != np.array(preds_tst).ravel()).astype(float)
            CE_tst = np.mean(e_tst) * 100.0
            display('Master_' + model_type + ': CE(%%) on test set =  %s' % str(CE_tst)[0:6], logger, True)
        except:
            raise
            '''
            preds_tst = None
            print('ERROR while computing predictions on test data')
            '''
        figures_folder = './results/figures/'
    
        classes = model.classes
        Xval_b = mn.add_bias(Xval).astype(float)
        Xtst_b = mn.add_bias(Xtst).astype(float)

        o_val_dict = model_loaded.predict_soft(Xval)
        o_tst_dict = model_loaded.predict_soft(Xtst)

        eval_multiclass_classification(pom, model_type, dataset_name, Xval_b, yval, Xtst_b, ytst, logger, True, mn, classes, o_val_dict, o_tst_dict, preds_val, preds_tst, figures_folder)
        #roc_auc_val, roc_auc_tst = eval_classification(pom, model_type, dataset_name, Xval_b, yval, Xtst_b, ytst, preds_val, preds_tst, logger, True, mn, figures_folder)

        display('Terminating all worker nodes.', logger, True)
        mn.terminate_workers()

        try:
            os.remove('current_taskname.txt')
        except:
            pass

        display('\n---------------------------------------------------------------------', logger, True)
        display('------------------------- END MMLL Procedure -------------------------', logger, True)
        display('----------------------------------------------------------------------\n', logger, True)
    else:
        display('\n---------------------------------------------------------------------', logger, True)
        display('------------------------- Training not completed ----------------------', logger, True)
        display('----------------------------------------------------------------------\n', logger, True)
