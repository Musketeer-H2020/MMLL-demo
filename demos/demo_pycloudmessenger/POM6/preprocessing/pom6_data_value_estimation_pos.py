# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
Dec 2020
python3 pom6_data_value_estimation.py --dataset income_dv_small --verbose 1

'''
import argparse
import time
import json
import sys, os
import numpy as np
from sklearn.metrics import roc_curve, auc

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
    if dataset_name in ["income_dv_small"]:
        input_data_description = {
                            "NI": 107, 
                            "input_types": [{"type": "num"}] * 107
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

    [Xval, yval] = dc.get_data_val()

    dv, best_workers = mn.get_data_value_aposteriori(Xval, yval, baseline_auc=0.7)

    Nworkers = len(best_workers)
    for kworker in range(Nworkers):
        display('Data value for worker %s is %f' % (best_workers[kworker], dv[kworker]), logger, True)


    '''
    # Training with selected workers
    selected_workers = ['0', '2', '3']
    display('Training the model %s with best workers' % model_type, logger, True)
    t_ini = time.time()
    mn.fit(Xval=Xval, yval=yval, selected_workers=selected_workers)
    t_end = time.time()

    display('Training is complete: Training time = %s seconds' % str(t_end - t_ini)[0:6], logger, True)
    model = mn.get_model()
    Xval_b = mn.add_bias(Xval)
    preds_val = model.predict(Xval_b).ravel()
    fpr_val, tpr_val, thresholds_val = roc_curve(list(yval), preds_val)
    roc_auc_val = auc(fpr_val, tpr_val)
    display('AUC with good workers = %f' % roc_auc_val, logger, True)
    display('----------------------------------------------------------------------', logger, True)


    # Training with worker 0
    selected_workers = ['0']
    # Training model with all workers
    display('Training the model %s with worker %s' % (model_type, selected_workers[0]), logger, True)
    t_ini = time.time()
    mn.fit(Xval=Xval, yval=yval, selected_workers=selected_workers)
    t_end = time.time()

    display('Training is complete: Training time = %s seconds' % str(t_end - t_ini)[0:6], logger, True)
    model = mn.get_model()
    Xval_b = mn.add_bias(Xval)
    preds_val = model.predict(Xval_b).ravel()
    fpr_val, tpr_val, thresholds_val = roc_curve(list(yval), preds_val)
    roc_auc_val = auc(fpr_val, tpr_val)
    display('AUC with worker %s = %f' % (selected_workers[0], roc_auc_val), logger, True)
    display('----------------------------------------------------------------------', logger, True)

    # Training model with all workers
    display('Training the model %s with all workers' % model_type, logger, True)
    t_ini = time.time()
    mn.fit(Xval=Xval, yval=yval)
    t_end = time.time()

    display('Training is complete: Training time = %s seconds' % str(t_end - t_ini)[0:6], logger, True)
    model = mn.get_model()
    Xval_b = mn.add_bias(Xval)
    preds_val = model.predict(Xval_b).ravel()
    fpr_val, tpr_val, thresholds_val = roc_curve(list(yval), preds_val)
    roc_auc_val = auc(fpr_val, tpr_val)
    display('AUC with all workers = %f' % roc_auc_val, logger, True)
    display('----------------------------------------------------------------------', logger, True)
    

    print('STOP AT ')
    import code
    code.interact(local=locals())

    stats_list = ['rxy', 'mx']
    ref_stats_dict = mn.compute_statistics(Xval, yval, stats_list)

    stats_dict_workers = mn.get_statistics_workers(stats_list + ['npatterns'])

    dv = np.zeros((5, 1))
    NP = np.zeros((5, 1))
    workers = ['0', '1', '2', '3', '4']
    Nworkers = len(workers)

    for kworker in range(Nworkers):
        worker = workers[kworker]
        das = []
        for stat in stats_list:
            das.append(np.dot(ref_stats_dict[stat].ravel(), stats_dict_workers[worker][stat].ravel()))        
        dv[kworker] = np.mean(np.array(das))
        NP[kworker] = stats_dict_workers[worker]['npatterns']

    dv = dv * NP
    dv = dv /np.sum(dv)

    for kworker in range(Nworkers):
        display('Data value for worker %s is %f' % (workers[kworker], dv[kworker]), logger, True)
    '''

    display('\n---------------------------------------------------------------------', logger, True)
    display('---------- Data value estimation completed  ----------------------', logger, True)
    display('----------------------------------------------------------------------\n', logger, True)

    display('Terminating all worker nodes.', logger, True)
    mn.terminate_workers()



