# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
Febr. 2021
python3 pom4_MLC_master_pycloudmessenger.py --dataset M-iris --verbose 1

'''
import argparse
import time
import json
import sys, os
import numpy as np
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

    pom = 4
    Nworkers = 5 + 1
    model_type = 'MLC'
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
    # For debugging purposes
    comms.pycloudmessenger_timeout_POMs456 = 0.1

    # Creating Masternode
    mn = MasterNode(pom, comms, logger, verbose)
    display('-------------------- Loading dataset %s --------------------------' % dataset_name, logger, True)
    # Warning: this data connector is only designed for the demos. In Musketeer, appropriate data
    # connectors must be provided
    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    dc = DC(data_file)
    #########################################
  
    #---------------  Creating a ML model (Master side) ---------------------  
    ########################################
    # Parameters depending on the model_type
    ########################################
    input_data_description = None
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
    if dataset_name == 'M-iris':
        mu = 0.05
        Nmaxiter = 20

    if input_data_description is not None:
        model_parameters = {}
        model_parameters.update({'Nmaxiter': Nmaxiter})
        model_parameters.update({'conv_stop': 0.005})
        model_parameters.update({'mu': mu})
        model_parameters.update({'momentum': 0})
        model_parameters.update({'regularization': 0.001})
        model_parameters.update({'input_data_description': input_data_description})
        model_parameters.update({'target_data_description': target_data_description})
    else:
        display('\n' + '='*50 + '\nERROR: input_data_description is missing\n' + '='*50 + '\n', logger, True)
        sys.exit()

    mn.create_model_Master(model_type, model_parameters=model_parameters)
    display('MMLL model %s is ready for training!' % model_type, logger, True)

    # We start the training procedure.
    display('Training the model %s' % model_type, logger, True)
    t_ini = time.time()

    mn.fit()

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
            o_val_dict, preds_val = model_loaded.predict(Xval)
            e_val = np.array(yval.ravel() != np.array(preds_val).ravel()).astype(float)
            CE_val = np.mean(e_val) * 100.0
            display('Master_' + model_type + ': CE(%%) on validation set =  %s' % str(CE_val)[0:6], logger, True)
        except:
            raise
            '''
            print('ERROR while computing predictions on validation data')
            import code
            code.interact(local=locals())
            '''

        try:
            [Xtst, ytst] = dc.get_data_tst()
            o_tst_dict, preds_tst = model_loaded.predict(Xtst)
            e_tst = np.array(ytst.ravel() != np.array(preds_tst).ravel()).astype(float)
            CE_tst = np.mean(e_tst) * 100.0
            display('Master_' + model_type + ': CE(%%) on test set =  %s' % str(CE_tst)[0:6], logger, True)
        except:
            raise
            '''
            print('ERROR while computing predictions on test data')
            import code
            code.interact(local=locals())
            '''

        figures_folder = './results/figures/'
        Xval_b = mn.add_bias(Xval).astype(float)
        Xtst_b = mn.add_bias(Xtst).astype(float)
        try:
            classes = target_data_description['output_type'][0]['values']
            eval_multiclass_classification(pom, model_type, dataset_name, Xval_b, yval, Xtst_b, ytst, logger, True, mn, classes, o_val_dict, o_tst_dict, preds_val, preds_tst, figures_folder)
        except:
            raise
            '''
            print('STOP AT eval_multiclass_classification')
            import code
            code.interact(local=locals())
            '''

        # Model export to ONXX
        output_filename_model = './results/models/POM' + str(pom) + '_' + model_type + '_' + dataset_name + '_model.onnx'
        model.save(output_filename_model)

        # Compute the prediction with ONNX Runtime
        sess = rt.InferenceSession(output_filename_model)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run([label_name], {input_name: Xtst.astype(np.float32)})[0]

        err_onnx = np.sum((np.array(preds_tst).ravel() != pred_onx.ravel()))
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
    else:
        display('\n---------------------------------------------------------------------', logger, True)
        display('------------------------- Training not completed ----------------------', logger, True)
        display('----------------------------------------------------------------------\n', logger, True)
