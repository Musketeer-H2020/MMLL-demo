# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
May 2020
python3 pom6_LC_pm_master_pycloudmessenger.py --dataset pima --verbose 1

'''
import argparse
import time
import json
import sys, os

# Add higher directory to python modules path.
sys.path.append("../../../../")

try:
    from MMLL.nodes.MasterNode import MasterNode
    from MMLL.common.MMLL_tools import display
    from MMLL.comms.comms_pycloudmessenger import Comms_master as Comms
except:
    print('\n' + 80 * '#')
    print('You need to install the MMLL library')
    print('pip install git+https://github.com/Musketeer-H2020/MMLL.git')
    print(80 * '#' + '\n')
    sys.exit()

from demo_tools.task_manager_pycloudmessenger import Task_Manager
from demo_tools.mylogging.logger_v1 import Logger
from demo_tools.data_connectors.Load_from_file import Load_From_File as DC                          # Data connector
from demo_tools.evaluation_tools import eval_classification

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

    # Logging is optional, if you do not want to log messages, simply set logger=None
    logger = Logger('./results/logs/Master.log')

    pom = 6
    Nworkers = 5
    model_type = 'LC_pm'
    dataset_name = FLAGS.dataset
    
    display('===========================================', logger, True)
    display('Creating Master... ', logger, True)
    display('Please wait until Master is ready before launching the workers...', logger, True)
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
    [Xval, yval] = dc.get_data_val()
    mn.set_validation_data(dataset_name, Xval, yval)
    display('MasterNode loaded %d patterns for validation' % mn.NPval, logger, True)
    [Xtst, ytst] = dc.get_data_tst()
    mn.set_test_data(dataset_name, Xtst, ytst)
    display('MasterNode loaded %d patterns for test' % mn.NPtst, logger, True)
    #########################################
  
    #---------------  Creating a ML model (Master side) ---------------------  
    ########################################
    # Parameters depending on the model_type
    ########################################
    model_parameters = {}
    model_parameters.update({'regularization': 0.001})
    model_parameters.update({'Nmaxiter': 10})
    model_parameters.update({'conv_stop': 0.005})

    mn.create_model_Master(model_type, model_parameters=model_parameters)
    display('MMLL model %s is ready for training!' % model_type, logger, True)

    # We start the training procedure.
    display('Training the model %s' % model_type, logger, True)
    t_ini = time.time()
    mn.fit()
    t_end = time.time()
    display('Training is complete: Training time = %s seconds' % str(t_end - t_ini)[0:6], logger, True)
    display('----------------------------------------------------------------------', logger, True)

    display('Retrieving the trained model from MasterNode', logger, True)
    model = mn.get_model()
    
    # Warning: this save_model utility is only for demo purposes
    output_filename_model = './results/models/POM' + str(pom) + '_' + model_type + '_' + dataset_name + '_model.pkl'
    mn.save_model(output_filename_model)

    display('-------------  Evaluating --------------------------------------------\n', logger, True)
    # Warning, these evaluation methods are not part of the MMLL library, they are only intended
    # to be used for the demos. Use them at your own risk.   

    # Adding bias...
    mn.Xval_b = mn.add_bias(mn.Xval_b)
    mn.Xtst_b = mn.add_bias(mn.Xtst_b)

    display('-------------  Obtaining predictions------------------------------------\n', logger, True)
    try:
        preds_tst = model.predict(mn.Xtst_b)
    except:
        preds_tst = None
        print('ERROR while computing predictions on test data')

    try:
        preds_val = model.predict(mn.Xval_b)
    except:
        preds_val = None
        print('ERROR while computing predictions on validation data')

    figures_folder = './results/figures/'
    roc_auc_val, roc_auc_tst = eval_classification(pom, model_type, dataset_name, mn.Xval_b, mn.yval, mn.Xtst_b, mn.ytst, preds_val, preds_tst, logger, verbose, mn, figures_folder)

    display('Terminating all worker nodes.', logger, True)
    mn.terminate_Workers()

    try:
        os.remove('current_taskname.txt')
    except:
        pass

    display('\n---------------------------------------------------------------------', logger, True)
    display('------------------------- END MMLL Procedure -------------------------', logger, True)
    display('----------------------------------------------------------------------\n', logger, True)