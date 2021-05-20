# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
November 2020
python3 pom5_LC_pm_master_pycloudmessenger_V.py --dataset income_raw --verbose 1

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
from demo_tools.data_connectors.Load_from_file_V import Load_From_File as DC_V                          # Data connector
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
    model_type = 'LC_pm'
    dataset_name = FLAGS.dataset

    '''
    import pickle
    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    with open(data_file, 'rb') as f:
        [Xtr_chunks, ytr_chunks, Xval, yval, Xtst, ytst, input_data_description_chunks, target_data_description_chunks] = pickle.load(f)

    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    dc = DC_V(data_file)
    [Xval, yval] = dc.get_data_val()

    print('STOP AT data CHECK')
    import code
    code.interact(local=locals())
    '''

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
    display('Creating MasterNode under POM5, communicating through pycloudmessenger', logger, True)
    # Creating Comms object, needed by MMLL
    comms = Comms(aggregator)
    # Creating Masternode
    mn = MasterNode(pom, comms, logger, verbose)
    display('-------------------- Loading dataset %s --------------------------' % dataset_name, logger, True)
    # Warning: this data connector is only designed for the demos. In Musketeer, appropriate data
    # connectors must be provided
    data_file = '../../../../input_data/' + dataset_name + '_demonstrator_data.pkl'
    dc = DC_V(data_file)

    #########################################
    input_data_description = None
    if dataset_name == "income_raw_V":
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

    if dataset_name == "mnist_binclass":
        input_data_description = {
                            "NI": 28*28, 
                            "input_types": [{"type": "num", "name": "pixel value"}] * 28*28
                            }

        target_data_description= {
        "NT": 1, 
        "output_types": [
        {"type": "bin", "name": "digit class", "definition": "even/odd number"}
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

    # We process data at workers, such that feature are correctly linked across workers
    # we obtain the data description at every worker
    input_data_description_dict, target_data_description_dict = mn.record_linkage_transform_workers() 


    ##############################################################################
    ######## Vertical Data normalization prior to training ################################
    ##############################################################################
    # We transform the data to numeric before training, if needed
    # transforms data at workers data and returns the transformer object
    display('Converting data at workers to numeric', logger, True)
    [d2num_transformer, new_input_data_description, errors_data2num] = mn.data2num_transform_workers_V(input_data_description) 

    [Xval, yval] = dc.get_data_val()
    if d2num_transformer is not None:
        Xval = d2num_transformer.transform(Xval)

    # Logarithmic preprocessing
    from MMLL.preprocessors.logscale import logscale_model
    prep_logscale = logscale_model(new_input_data_description)
    display('Preprocessing data at workers with logscale', logger, True)
    errors_logscale = mn.preprocess_data_at_workers_V(prep_logscale)
   
    if prep_logscale is not None:
        Xval = prep_logscale.transform(Xval)

    # outlier clipping preprocessing
    from MMLL.preprocessors.outlier_clipping import outlier_clipping_model
    times_sigma = 1.5
    prep_outlier_clipping = outlier_clipping_model(new_input_data_description, times_sigma)
    display('Preprocessing data at workers with outlier_clipping', logger, True)
    errors_outlier_clipping = mn.preprocess_data_at_workers_V(prep_outlier_clipping)
    Xval = prep_outlier_clipping.transform(Xval)

    from MMLL.preprocessors.normalizer import normalize_model
    normalizer = normalize_model(input_data_description=new_input_data_description, method='global_mean_std')
    display('Normalizing data at workers', logger, True)
    errors_normalization, normalizer = mn.preprocess_data_at_workers_V(normalizer)

    Xval = normalizer.transform(Xval)


    print('STOP AT vertical partition demo, dataset is ready for training.')
    import code
    code.interact(local=locals())






    # We start the training procedure.
    display('Training the model %s' % model_type, logger, True)
    t_ini = time.time()
    display('MasterNode loaded %d patterns for validation' % Xval.shape[0], logger, True)
    #mn.fit(Xval=Xval, yval=yval)
    mn.fit()
    t_end = time.time()
    display('Training is complete: Training time = %s seconds' % str(t_end - t_ini)[0:6], logger, True)
    display('----------------------------------------------------------------------', logger, True)

    if mn.model_is_trained:
        display('Retrieving the trained model from MasterNode', logger, True)
        model = mn.get_model()
        
        # Warning: this save_model utility is only for demo purposes
        output_filename_model = './results/models/POM' + str(pom) + '_' + model_type + '_' + dataset_name + '_model.pkl'
        mn.save_model(output_filename_model)

        display('-------------  Evaluating --------------------------------------------\n', logger, True)
        # Warning, these evaluation methods are not part of the MMLL library, they are only intended
        # to be used for the demos. Use them at your own risk.   

        display('-------------  Obtaining predictions------------------------------------\n', logger, True)
        
        try:
            [Xval, yval] = dc.get_data_val()
            if d2num_transformer is not None:
                Xval = d2num_transformer.transform(Xval)
            Xval = normalizer.transform(Xval)
            Xval = pca_transformer.transform(Xval)
            Xval_b = mn.add_bias(Xval)
            preds_val = model.predict(Xval_b)
        except:
            preds_val = None
            print('ERROR while computing predictions on validation data')

        try:
            [Xtst, ytst] = dc.get_data_tst()
            if d2num_transformer is not None:
                Xtst = d2num_transformer.transform(Xtst)
            Xtst = normalizer.transform(Xtst)
            Xtst = pca_transformer.transform(Xtst)            
            Xtst_b = mn.add_bias(Xtst)
            display('MasterNode loaded %d patterns for test' % Xtst.shape[0], logger, True)
            preds_tst = model.predict(Xtst_b)
        except:
            preds_tst = None
            print('ERROR while computing predictions on test data')

        figures_folder = './results/figures/'
        roc_auc_val, roc_auc_tst = eval_classification(pom, model_type, dataset_name, Xval_b, yval, Xtst_b, ytst, preds_val, preds_tst, logger, True, mn, figures_folder)

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

    print('STOP AT ')
    import code
    code.interact(local=locals())
