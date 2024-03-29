# -*- coding: utf-8 -*-
'''
@author:  Marcos Fernandez Diaz
February 2020

Example of use: python pom1_NN_model_averaging_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --normalization <normalization>  --implementation <implementation>

Parameters:
    - user: String with the name of the user. If the user does not exist in the pycloudmessenger platform a new one will be created
    - password: String with the password
    - task_name: String with the name of the task. If the task already exists, an error will be displayed
    - normalization: String indicating whether to apply normalization. Possible options are std or minmax. By default no normalization is used
    - implementation: String indicating whether to use gradient_averaging or model_averaging implementation. By default the latter is used.
    - optimizer: String indicating the type of optimizer to use (only valid when gradient implementation=gradient_descent).

'''

# Import general modules
import argparse
import logging
import json
import time
import numpy as np
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disables tensorflow warnings
import tensorflow as tf
import onnxruntime
from sklearn.metrics import accuracy_score

# Add higher directory to python modules path.
sys.path.append("../../../../")
os.environ['KMP_WARNINGS'] = 'off' # Remove KMP_AFFINITY logs

# To be imported from MMLL (pip installed)
from MMLL.nodes.MasterNode import MasterNode
from MMLL.comms.comms_pycloudmessenger import Comms_master as Comms
from MMLL.aggregators.aggregator import ModelAveraging, SGD

# To be imported from demo_tools 
from demo_tools.task_manager_pycloudmessenger import Task_Manager
from demo_tools.data_connectors.Load_from_file import Load_From_File as DC
from demo_tools.mylogging.logger_v1 import Logger
from demo_tools.evaluation_tools import display, plot_cm_seaborn, create_folders


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
    parser.add_argument('--normalization', type=str, default='no', choices=['no', 'std', 'minmax'], help='Type of normalization')
    parser.add_argument('--implementation', type=str, default='model_averaging', choices=['model_averaging', 'gradient_descent'], help='Type of implementation')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['Adam', 'RMSprop', 'SGD'], help='Gradient descent optimizer')

    FLAGS, unparsed = parser.parse_known_args()
    user_name = FLAGS.user
    user_password = FLAGS.password
    task_name = FLAGS.task_name
    normalization = FLAGS.normalization
    implementation = FLAGS.implementation
    optimizer = FLAGS.optimizer


    # Set basic configuration
    dataset_name = 'mnist'
    verbose = False
    pom = 1
    model_type = 'NN'
    Nworkers = 2


    # Create the directories for storing relevant outputs if they do not exist
    create_folders("./results/")

    # Setting up the logger 
    logger = Logger('./results/logs/Master.log')   


    # Load the model architecture as defined by Keras model.to_json()
    keras_filename = 'keras_model_CNN_mnist.json'
    try:
        with open('./' + keras_filename, 'r') as json_file:
            model_architecture = json_file.read()
    except:
        display('Error - The file ' + keras_filename + ' defining the neural network architecture is not available, please put it under the following path: "' + os.path.abspath(os.path.join("","./")) + '"', logger, verbose)
        sys.exit()

    # Task definition
    if implementation.lower() == 'model_averaging':
        model_averaging = 'True'
    else:
        model_averaging = 'False'

    task_definition = {"quorum": Nworkers, 
                       "POM": pom, 
                       "model_type": model_type, 
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
    data_description = {"NI": 1, "input_types": [{"type": "matrix", "name": "image"}]}

  
    # Creating a ML model
    model_parameters = {}
    model_parameters['learning_rate'] = 0.15
    model_parameters['Nmaxiter'] = 5
    model_parameters['model_architecture'] = model_architecture
    model_parameters['optimizer'] = optimizer
    model_parameters['momentum'] = 1
    model_parameters['nesterov'] = 'False'
    model_parameters['loss'] = 'categorical_crossentropy'
    model_parameters['metric'] = 'accuracy'
    model_parameters['batch_size'] = 128
    model_parameters['num_epochs'] = 2
    model_parameters['model_averaging'] = model_averaging
    model_parameters['aggregator'] = None


    mn.create_model_Master(model_type, model_parameters=model_parameters)
    display('MMLL model %s is ready for training!' % model_type, logger, verbose) 


    # Normalization of data in each worker before training
    if normalization=='std':
        normalizer = mn.normalizer_fit_transform_workers(data_description, 'global_mean_std')
    elif normalization=='minmax':
        normalizer = mn.normalizer_fit_transform_workers(data_description, 'global_min_max')

    # Start the training procedure.
    display('Training the model %s' % model_type, logger, verbose)
    t_ini = time.time()
    [Xval, yval] = dc.get_data_val()
    Xval = Xval.reshape((Xval.shape[0], 28, 28, 1))

    if normalization != 'no':
        Xval = normalizer.transform(Xval)

    mn.fit(Xval=Xval, yval=yval)
    t_end = time.time()
    display('Training is complete: Training time = %s seconds' % str(t_end - t_ini)[0:6], logger, verbose)


   # Retrieving and saving the final model
    display('Retrieving the trained model from MasterNode', logger, verbose)
    model = mn.get_model()    
    # Warning: this save_model utility is only for demo purposes
    output_filename_model = './results/models/Master_' + dataset_name + '_model'
    model.save(output_filename_model)
    
    # Making predictions on test data
    display('-------------  Obtaining predictions----------------------------------\n', logger, verbose)
    [Xtst, ytst] = dc.get_data_tst()
    # Reshape data for training with CNNs
    Xtst = Xtst.reshape((Xtst.shape[0], 28, 28, 1))

    if normalization != 'no':
        Xtst = normalizer.transform(Xtst)
    preds_tst = model.predict(Xtst)
    preds_tst = np.argmax(preds_tst, axis=-1) # Convert to labels
    y = np.argmax(ytst, axis=-1) # Convert to labels
    classes = np.arange(ytst.shape[1]) # 0 to 9

    # Evaluating the results
    display('-------------  Evaluating --------------------------------------------\n', logger, verbose)
    # Warning, these evaluation methods are not part of the MMLL library, they are only intended
    # to be used for the demos. Use them at your own risk.
    output_filename = 'Master_NN_confusion_matrix_' + dataset_name + '.png'
    title = 'NN confusion matrix in test set master'
    plot_cm_seaborn(preds_tst, y, classes, title, output_filename, logger, verbose, normalize=True)


    # Load Tf SavedModel and check results
    model_loaded = tf.keras.models.load_model(output_filename_model)
    preds_tst = model_loaded.predict(Xtst)
    preds_tst = np.argmax(preds_tst, axis=-1) # Convert to labels


    # Model export to ONXX
    output_filename_model = './results/models/Master_' + dataset_name + '_model.onnx'
    model.save(output_filename_model)

    # Compute the prediction with ONNX Runtime
    onnx_session = onnxruntime.InferenceSession(output_filename_model)
    onnx_inputs = {onnx_session.get_inputs()[0].name: Xtst}
    onnx_output = onnx_session.run(None, onnx_inputs)[0]
    onnx_output = np.argmax(onnx_output, axis=-1) # Convert to labels
    err_onnx = accuracy_score(y,onnx_output)
    display('Test accuracy in ONNX model is %f' %err_onnx, logger, verbose)


    # Terminate workers
    display('Terminating all worker nodes.', logger, verbose)
    mn.terminate_workers()

    display('----------------------------------------------------------------------', logger, verbose)
    display('------------------------- END MMLL Procedure -------------------------', logger, verbose)
    display('----------------------------------------------------------------------\n', logger, verbose)
