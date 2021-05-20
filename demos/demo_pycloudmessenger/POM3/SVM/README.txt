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
    - task_name: String with the name of the task.
    - id: Integer representing the partition of data to be used by the worker. Each worker should use a different partition, possible values are 0 to 4.
    - normalization: String indicating wether to apply standard normalization. Possible values are yes or no.

Important notes:
    - The task_name should be the same for all terminals.
    - Each user should have a different id, otherwise they will be training using the same dataset partition.
    - It takes around one hour and ten minutes to train the SVM on a machine with 8 cores and a modern processor due to the high processing times required for encrypting the weights of the model. In case you want to reduce the training time try a configuration with less centroids (NC) or less iterations (Nmaxiter).
-------------------------------------------


==================================================================
 Without normalization
==================================================================
python pom3_SVM_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name>
python pom3_SVM_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom3_SVM_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1


==================================================================
 With std normalization
==================================================================
python pom3_SVM_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --normalization yes
python pom3_SVM_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom3_SVM_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1

