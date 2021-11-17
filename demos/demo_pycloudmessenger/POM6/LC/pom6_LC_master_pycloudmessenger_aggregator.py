# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
Jan 2021
python3 pom6_LC_master_pycloudmessenger.py --dataset pima --verbose 1

'''
import argparse
import time
import json
import sys, os
import numpy as np
import pickle
import onnxruntime as rt  # pip install onnxruntime

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
from demo_tools.data_connectors.Load_from_file import Load_From_File as DC                          # Data connector
from demo_tools.evaluation_tools import eval_classification, create_folders
from MMLL.aggregators.sample_aggregator_POMs456 import SGD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default=None, help='The external names of the workers')
    parser.add_argument('--verbose', type=str, default='1', help='Print messages on screen when True')
    parser.add_argument('--use_dve', type=str, default='0', help='Online Data Value Estimation')
    parser.add_argument('--dve_weight', type=str, default='0', help='Weight workers contributions by DV')
    parser.add_argument('--dataset', type=str, default=None, help='The dataset to be used')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.verbose == '1':
        verbose = True
    else:
        verbose = False

    if FLAGS.use_dve == '1':
        use_dve = True
    else:
        use_dve = False

    if FLAGS.dve_weight == '1':
        dve_weight = True
    else:
        dve_weight = False

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

    input_data_description = None
    if dataset_name == 'pima':
        input_data_description = {
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
    if dataset_name == 'Bmnist':
        input_data_description = {
                    "NI": 784, 
                    "input_types": [
                    {"type": "num"} 
                    ] * 784
                    }
        
    if dataset_name in ["income_dv_small", "income_dv", "income_DV"]:
        input_data_description = {
                            "NI": 107, 
                            "input_types": [{"type": "num"}] * 107
                            }

    if dataset_name in ["pima_DV"]:
        input_data_description = {
                            "NI": 8, 
                            "input_types": [{"type": "num"}] * 8
                            }

        target_data_description= {
        "NT": 1, 
        "output_types": [
        {"type": "bin", "name": "income", "definition": ">50K, <=50K"}
        ]
        }

    if dataset_name in ["mnist_binclass_bad34"]:
        input_data_description = {
                            "NI": 784, 
                            "input_types": [{"type": "num"}] * 784
                            }

        target_data_description= {
        "NT": 1, 
        "output_types": [
        {"type": "bin", "name": "even_odd", "definition": "Even or odd digit"}
        ]
        }

    #---------------  Creating a ML model (Master side) ---------------------  
    ########################################
    # Parameters depending on the model_type
    ########################################
    if input_data_description is not None:
        model_parameters = {}
        model_parameters.update({'regularization': 0.001})
        model_parameters.update({'Nmaxiter': 20})
        model_parameters.update({'conv_stop': 0.01})
        model_parameters.update({'input_data_description': input_data_description})
        model_parameters.update({'mu': 0.3})
        if use_dve:
            model_parameters.update({'use_dve': True})           
            if dve_weight:
                model_parameters.update({'dve_weight': True})

        aggregator_adversarial = SGD(0.3)
        model_parameters.update({'aggregator': aggregator_adversarial})
    else:
        display('\n' + '='*50 + '\nERROR: input_data_description is missing\n' + '='*50 + '\n', logger, True)
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
        except:
            raise
            '''
            preds_val = None
            print('ERROR while computing predictions on validation data')
            import code
            code.interact(local=locals())
            '''
        try:
            [Xtst, ytst] = dc.get_data_tst()
            preds_tst = model_loaded.predict(Xtst)
        except:
            raise
            '''
            preds_tst = None
            print('ERROR while computing predictions on test data')
            import code
            code.interact(local=locals())
            '''

        figures_folder = './results/figures/'
        
        Xval_b = mn.add_bias(Xval)
        Xtst_b = mn.add_bias(Xtst)
        roc_auc_val, roc_auc_tst = eval_classification(pom, model_type, dataset_name, Xval_b, yval, Xtst_b, ytst, preds_val, preds_tst, logger, True, mn, figures_folder)

        # Model export to ONXX
        output_filename_model = './results/models/POM' + str(pom) + '_' + model_type + '_' + dataset_name + '_model.onnx'
        model.save(output_filename_model)

        # Compute the prediction with ONNX Runtime
        sess = rt.InferenceSession(output_filename_model)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        #pred_onx = sess.run([label_name], {input_name: Xtst.astype(np.float32)})[0]
        prob_name = sess.get_outputs()[1].name
        pred_onx = sess.run([prob_name], {input_name: Xtst.astype(np.float32)})[0]
        pred_onx_postproc = np.array([p[1] for p in pred_onx])
        err_onnx = np.mean((preds_tst.ravel() - pred_onx_postproc.ravel())**2)
        print('Error in ONNX predictions is %f' %err_onnx )
        print('=' * 80)

        # Model export to PMML
        output_filename_model = './results/models/POM' + str(pom) + '_' + model_type + '_' + dataset_name + '_model.pmml'
        model.save(output_filename_model)

        display('Terminating all worker nodes.', logger, True)
        mn.terminate_workers()

        try:
            os.remove('current_taskname.txt')
        except:
            pass

        display('\n---------------------------------------------------------------------', logger, True)
        display('------------------------- END MMLL Procedure -------------------------', logger, True)
        display('----------------------------------------------------------------------\n', logger, True)

        '''
        print('STOP AT END')
        import code
        code.interact(local=locals())
        '''
    else:
        display('\n---------------------------------------------------------------------', logger, True)
        display('------------------------- Training not completed ----------------------', logger, True)
        display('----------------------------------------------------------------------\n', logger, True)
    
