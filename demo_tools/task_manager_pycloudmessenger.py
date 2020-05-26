# -*- coding: utf-8 -*-
'''
Task managing utilities
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
try:
    import pycloudmessenger.ffl.abstractions as ffl
    import pycloudmessenger.ffl.fflapi as fflapi
    import pycloudmessenger.serializer as serializer
except:
    print("pycloudmessenger is not installed, use:")
    print("pip install https://github.com/IBM/pycloudmessenger/archive/v0.3.0.tar.gz")

class Task_Manager:
    """
    """

class Task_Manager:
    """
    """

    def __init__(self, credentials_filename):
        """
        """
        self.credentials_filename = credentials_filename

    def create_master_random_taskname(self, pom, Nworkers, user_name=None, user_password='Tester', user_org='Musketeer', task_name='Test', random_taskname=True):
        self.pom = pom
        self.Nworkers = Nworkers
        config = 'cloud'
        if random_taskname:
            rword = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
            version = '_' + rword
        else:
            version = '_v2'
        task_name += version
        self.task_name = task_name
        user_password  += version
        
        if user_name is None:
            user_name = 'ma' + version
        
        ffl.Factory.register(config, fflapi.Context, fflapi.User, fflapi.Aggregator, fflapi.Participant)
        context_master = ffl.Factory.context(config, self.credentials_filename)

        ffl_user_master = ffl.Factory.user(context_master)
        with ffl_user_master:
            try:
                ffl_user_master.create_user(user_name, user_password, user_org)
            except Exception as err:
                print(str(err).split(':')[1])

        context_master = ffl.Factory.context(config, self.credentials_filename, user_name, user_password, encoder = serializer.Base64Serializer)
        ffl_user_master = ffl.Factory.user(context_master)

        # Create task
        task_definition = {"task_name": task_name,
                            "owner": user_name, 
                           "quorum": self.Nworkers, 
                           "POM": self.pom,
                           "model_type": "None", 
                          }
        with ffl_user_master:
            try:
                result = ffl_user_master.create_task(task_name, ffl.Topology.star, task_definition)
            except Exception as err:
                print(str(err).split(':')[1])

        # We write to disk the name of the task, to be read by the workers. In the real system, 
        # the task_name must be communicated by other means.
        with open('current_taskname.txt', 'w') as f:
            f.write(task_name)
        self.aggregator = ffl.Factory.aggregator(context_master, task_name=task_name)
        #return context_master, task_name
        return self.aggregator


    def create_master_and_taskname(self, display, logger, task_definition, user_name=None, user_password='Tester', task_name='Test', user_org='TREE', verbose=False):
        self.task_name = task_name
        self.Nworkers = task_definition['quorum']
        config = 'cloud'

        # Create context for the cloud communications
        ffl.Factory.register(config, fflapi.Context, fflapi.User, fflapi.Aggregator, fflapi.Participant)
        context = ffl.Factory.context(config, self.credentials_filename)

        # Create user
        ffl_user = ffl.Factory.user(context)
        with ffl_user:
            try:
                ffl_user.create_user(user_name, user_password, user_org)
                display('User %s has registered in pycloudmessenger platform!' %user_name, logger, verbose)
            except Exception as err:
                display('The user %s is already registered in pycloudmessenger platform.' %user_name, logger, verbose)
        context = ffl.Factory.context(config, self.credentials_filename, user_name, user_password, encoder=serializer.Base64Serializer)

        try:
            user = ffl.Factory.user(context)
            with user:
                import json
                result = user.create_task(task_name, ffl.Topology.star, task_definition)
                task_definition = json.loads(user.task_info(task_name)['definition'])
        except Exception as err:
            display('Error: %s' %err, logger, verbose)
            import sys
            sys.exit()
 
        self.aggregator = ffl.Factory.aggregator(context, task_name=task_name)
        return self.aggregator


    def get_current_task_name(self):
        task_available = False
        while not task_available:
            try:
                with open('current_taskname.txt', 'r') as f:
                    self.task_name = f.read()
                task_available = True
            except:
                print('No available task yet...')
                time.sleep(1)
                pass
        return self.task_name

    def create_worker_join_task(self, id, user_password='Tester', user_org='Musketeer'):
        created = False
        while not created:
            try:
                self.task_name = self.get_current_task_name()
                print(self.task_name)
                config = 'cloud'

                version = self.task_name.split('_')[1]
                worker_name = 'worker_' + str(id) + '_' + version
                user_password  += version

                ffl.Factory.register(config, fflapi.Context, fflapi.User, fflapi.Aggregator, fflapi.Participant)
                context_w = ffl.Factory.context(config, self.credentials_filename)
                ffl_user_worker = ffl.Factory.user(context_w)
                with ffl_user_worker:
                    try:
                        ffl_user_worker.create_user(worker_name, user_password, user_org)
                    except Exception as err:
                        print(str(err).split(':')[1])
                context_w = ffl.Factory.context('cloud', self.credentials_filename, worker_name, user_password, encoder = serializer.Base64Serializer)
                user_worker = ffl.Factory.user(context_w)
                with user_worker:
                    try:
                        result = user_worker.join_task(self.task_name)
                        print('Worker %s has joined task %s' % (worker_name, self.task_name))
                        created = True
                    except Exception as err:
                        print(str(err).split(':')[1])
            except:
                print('waiting for Master...')
                time.sleep(1)
                pass

        participant = ffl.Factory.participant(context_w, task_name=self.task_name)
        
        return participant


    def create_worker_and_join_task(self, user_name, user_password, task_name, display, logger, user_org='TREE', verbose=False):
        config = 'cloud'

        # Create context for the cloud communications
        ffl.Factory.register(config, fflapi.Context, fflapi.User, fflapi.Aggregator, fflapi.Participant)
        context = ffl.Factory.context(config, self.credentials_filename)

        # Create user
        ffl_user = ffl.Factory.user(context)
        with ffl_user:
            try:
                ffl_user.create_user(user_name, user_password, user_org)
                display('User %s has registered in pycloudmessenger platform!' %user_name, logger, verbose)
            except Exception as err:
                display('The user %s is already registered in pycloudmessenger platform.' %user_name, logger, verbose)
        context = ffl.Factory.context(config, self.credentials_filename, user_name, user_password, encoder=serializer.Base64Serializer)

        # Join task
        try:
            user = ffl.Factory.user(context)
            with user:
                user.join_task(task_name)
        except Exception as err:
            import sys
            display('Error: %s' %err, logger, verbose)
            sys.exit()

        # Create the comms object
        participant = ffl.Factory.participant(context, task_name=task_name)
        return participant


    def wait_for_workers(self):

        stop = False
        workers = self.aggregator.get_participants()

        while not stop: 
            try:
                with self.aggregator:
                    resp = self.aggregator.receive(1)
                participant = resp.notification['participant']
                workers.append(participant)
                print('Task %s: participant %s has joined' % (self.task_name, participant))
            except Exception as err:
                print("Task %s: joined %d participants out of %d" % (self.task_name, len(workers), self.Nworkers))
                #print(err)
                #print('Check here: error')
                #import code
                #code.interact(local=locals())
                pass

            if len(workers) == self.Nworkers:
                stop = True

        workers = self.aggregator.get_participants()
        return list(workers.keys())


    def wait_for_workers_to_join(self, display, logger, verbose=False):
        """
        Wait for workers to join until quorum is met.
        """
        with self.aggregator:
            workers = self.aggregator.get_participants()

        if workers:
            if len(workers) == self.Nworkers:
                display('Participants have already joined', logger, verbose)
                return workers

        display('Waiting for workers to join (%d of %d present)' %(len(workers), self.Nworkers), logger, verbose)

        ready = False
        while not ready:
            try:
                with self.aggregator:
                    resp = self.aggregator.receive(300)
                    participant = resp.notification['participant']
                display('Participant %s joined' %participant, logger, verbose)
            except Exception as err:
                raise err

            if len(workers) == self.Nworkers:
                ready = True

        return workers
