==================================================================
 Demo execution instructions using pycloudmessenger under Linux OS
==================================================================

Open three bash terminals and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

-------------------------------------------
Execute these lines, one at every terminal. Start launching the master, and wait until it is ready to receive workers before launching the workers.

Once the training is completed, these demo scripts produce the output files in the results/ folder (models, figures, logs)
-------------------------------------------

Parameters:
    - user: String with the name of the user. If the user does not exist in the pycloudmessenger platform a new one will be created.
    - password: String with the password.
    - task_name: String with the name of the task. If the task already exists, an error will be displayed.
    - id: Integer representing the partition of data to be used by the worker. Each worker should use a different partition, possible values are 0 to 4.
    - normalization: String indicating wether to apply normalization. Possible options are std, minmax or no. By default no normalization is used.
    - implementation: String indicating whether to use gradient_averaging or model_averaging implementation. By default the latter is used.
    - optimizer: String indicating the type of optimizer to use (only valid when gradient implementation=gradient_descent). By default Stochastic Gradient Descent is used.

Important notes:
    - The task_name should be the same for all terminals.
    - Each user should have a different id, otherwise they will be training using the same dataset partition.
    - The architecture of the Keras model to use is defined inside this folder. If you want to try a different architecture use the script model_definition_keras.py and define a new architecture using the sequential or functional API provided by Keras. This new filename should be updated at the beginning of pom1_NN_master_pycloudmessenger.py in order for the changes to take place.
-------------------------------------------


==================================================================
 Model averaging without normalization
==================================================================
python pom1_NN_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name>
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1


==================================================================
 Model averaging with minmax normalization
==================================================================
python pom1_NN_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --normalization minmax
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1


==================================================================
 Model averaging with std normalization
==================================================================
python pom1_NN_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --normalization std
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1


==================================================================
 Gradient averaging without normalization
==================================================================
python pom1_NN_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --implementation gradient_descent --optimizer SGD
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1


==================================================================
 Gradient averaging with minmax normalization
==================================================================
python pom1_NN_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --implementation gradient_descent --optimizer SGD --normalization minmax
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1


==================================================================
 Gradient averaging with std normalization
==================================================================
python pom1_NN_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name>  --implementation gradient_descent --optimizer SGD --normalization std
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1


==================================================================
 SGD with momentum
==================================================================
python pom1_NN_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --implementation gradient_descent --optimizer SGD
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom1_NN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1


==================================================================
 Convolutional Neural Networks example
==================================================================
python pom1_CNN_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name>
python pom1_CNN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom1_CNN_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1

