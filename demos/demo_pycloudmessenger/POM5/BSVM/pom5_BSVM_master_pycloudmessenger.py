# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
Apr 2021
python3 pom5_BSVM_master_pycloudmessenger.py --dataset synth2D-class --verbose 1

'''
import argparse
import time
import json
import sys, os
import pickle
import onnxruntime as rt  # pip install onnxruntime
import numpy as np

# Add higher directory to python modules path.
sys.path.append("../../../../")

try:
    from MMLL.nodes.MasterNode import MasterNode
    from MMLL.common.MMLL_tools import display, estimate_centroids
    from MMLL.comms.comms_pycloudmessenger import Comms_master as Comms
    from demo_tools.crypto.crypt_PHE import Crypto as CR
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

    pom = 5
    Nworkers = 5
    model_type = 'BSVM'
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
    NC = 30
    NI = 2
    NCcandidates = 10 * NC  # Initial randomm number
    NCini = 2 * NC          # Initial number after pruning by distance
    minvalue = -1.5
    maxvalue = 1.5

    # We retrieve double number of centroids, to finally select those with maximal projection
    C = estimate_centroids(NCini, NI, NCcandidates, minvalue, maxvalue, False)

    #########################################
    display('Creating MasterNode under POM6, communicating through pycloudmessenger', logger, True)
    # Creating Comms object, needed by MMLL
    comms = Comms(aggregator)

    # Defining encryption object
    key_size = 512
    cr = CR(key_size=key_size)

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
    if dataset_name == 'synth2D-class':
        input_data_description = {
                    "NI": 2, 
                    "input_types": [
                    {"type": "num"}
                    ] * 2
                    }

    if input_data_description is not None:
        model_parameters = {}
        model_parameters.update({'C': C})
        model_parameters.update({'NC': NC})
        model_parameters.update({'conv_stop': 0.005})
        model_parameters.update({'input_data_description': input_data_description})
        model_parameters.update({'cr': cr})

        if dataset_name == 'synth2D-class':
            model_parameters.update({'regularization': 0.001})
            model_parameters.update({'fsigma': 0.15})
            model_parameters.update({'Csvm': 100})
            model_parameters.update({'landa': 0.5})
            model_parameters.update({'Nmaxiter': 20})
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
            #Xval_b = mn.add_bias(Xval)
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
            #Xtst_b = mn.add_bias(Xtst)
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
        roc_auc_val, roc_auc_tst = eval_classification(pom, model_type, dataset_name, Xval, yval, Xtst, ytst, preds_val, preds_tst, logger, True, model_loaded, figures_folder)

        # Model export to ONXX
        output_filename_model = './results/models/POM' + str(pom) + '_' + model_type + '_' + dataset_name + '_model.onnx'
        model.save(output_filename_model)

        # Compute the prediction with ONNX Runtime
        sess = rt.InferenceSession(output_filename_model)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run([label_name], {input_name: Xtst.astype(np.float32)})[0]
        err_onnx = np.sum((np.array(preds_tst).ravel() - pred_onx.ravel())**2)
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


