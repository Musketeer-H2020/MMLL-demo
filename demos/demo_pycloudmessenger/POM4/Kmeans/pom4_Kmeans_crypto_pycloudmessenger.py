# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
Jan. 2020
python pom4_Kmeans_crypto_pycloudmessenger.py --dataset synth2D --verbose 1

'''
import argparse
import json
import sys
# Add higher directory to python modules path.
sys.path.append("../../../../")

try:
    from MMLL.nodes.CryptoNode import CryptoNode
    from MMLL.common.MMLL_tools import display
    from MMLL.comms.comms_pycloudmessenger import Comms_worker as Comms
except Exception as err:
    if "No module named 'MMLL'" in str(err):
        print('\n' + 80 * '#')
        print('You need to install the MMLL library')
        print('pip install git+https://github.com/Musketeer-H2020/MMLL.git')
        print(80 * '#' + '\n')
    raise

# Add higher directory to python modules path.
sys.path.append("../../../../")
from demo_tools.task_manager_pycloudmessenger import Task_Manager
from demo_tools.crypto.crypt_PHE import Crypto as CR
from demo_tools.mylogging.logger_v1 import Logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default=None, help='The addresses of the workers')
    parser.add_argument('--verbose', type=str, default='1', help='Print on screen if True')
    #parser.add_argument('--dataset', type=str, default=None, help='The dataset to be used')
    #parser.add_argument('--platform', type=str, default='pycloudmessenger', help='The comms plaftorm')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.verbose == '1':
        verbose = True
    else:
        verbose = False

    pom = 4
    # This string identifies the worker in the real world. Unused by now...
    # pycloudmessenger assigns internal pseudo-ids for communicating
    cryptonode_real_name = str(FLAGS.id)   

    #master_address = 'ma'   
    model_type = 'Kmeans'

    # Defining encryption object
    key_size = 512
    cr = CR(key_size=key_size)

    logger = Logger('./results/logs/crypto.log')
        
    display('===========================================', logger, True)
    display('Creating CryptoNode...', logger, True)
    credentials_filename = '../../musketeer.json'
    tm = Task_Manager(credentials_filename)
    task_name = tm.get_current_task_name()
    # We need the participant to build comms object
    participant = tm.create_worker_join_task(cryptonode_real_name)
    display("Worker has joined task %s" % task_name, logger, True)
    # ==================================================

    display('===========================================', logger, True)
    display('Creating CryptoNode...', logger, True)

    display('Creating CryptoNode under POM4, communicating through pycloudmessenger', logger, True)
    #########################################
    # Creating Comms object, needed by MMLL
    comms = Comms(participant, cryptonode_real_name)
    # For debugging purposes
    comms.pycloudmessenger_timeout_POMs456 = 0.1

    #########################################
    # Creating Cryptonode
    cn = CryptoNode(pom, comms, logger, verbose=verbose, cr=cr)

    #---------------  Creating a ML model (Worker side) ---------------------  
    display('Crypto: Creating ML model of type %s' % model_type, logger, verbose)
    cn.create_model_crypto(model_type)

    display('Cryptonode is running %s...' % model_type, logger, True)
    cn.run()

    display('Crypto ' + model_type + ' : EXIT', logger, True)