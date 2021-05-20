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
    - normalization: String indicating wether to apply normalization. Possible options are std, minmax or no. By default no normalization is used.

Important notes:
    - The task_name should be the same for all terminals.
    - Each user should have a different id, otherwise they will be training using the same dataset partition.
-------------------------------------------


==================================================================
 Without normalization
==================================================================
python pom1_FBSVM_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name>
python pom1_FBSVM_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom1_FBSVM_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1


==================================================================
 With standard normalization
==================================================================
python pom1_FBSVM_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --normalization std
python pom1_FBSVM_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom1_FBSVM_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1


==================================================================
 With minmax normalization
==================================================================
python pom1_FBSVM_master_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --normalization minmax
python pom1_FBSVM_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 0
python pom1_FBSVM_worker_pycloudmessenger.py --user <user> --password <password> --task_name <task_name> --id 1



###### NON ID experiments  4W

mnist_binclass

taskset --cpu-list 5 python3 pom1_FBSVM_master_pycloudmessenger_nonid.py --user UC3Mm_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset mnist_binclass --Nworkers 4 &

taskset --cpu-list 1 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M0_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset mnist_binclass --Nworkers 4 --id 0 &

taskset --cpu-list 2 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M1_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset mnist_binclass --Nworkers 4  --id 1 & 

taskset --cpu-list 3 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M2_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset mnist_binclass --Nworkers 4 --id 2 & 

taskset --cpu-list 4 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M3_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset mnist_binclass --Nworkers 4 --id 3 & 

ijcnn1

taskset --cpu-list 5 python3 pom1_FBSVM_master_pycloudmessenger_nonid.py --user UC3Mm_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset ijcnn1 --Nworkers 4 &

taskset --cpu-list 1 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M0_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset ijcnn1 --Nworkers 4 --id 0 &

taskset --cpu-list 2 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M1_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset ijcnn1 --Nworkers 4  --id 1 & 

taskset --cpu-list 3 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M2_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset ijcnn1 --Nworkers 4 --id 2 & 

taskset --cpu-list 4 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M3_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset ijcnn1 --Nworkers 4 --id 3 & 


phishing

taskset --cpu-list 5 python3 pom1_FBSVM_master_pycloudmessenger_nonid.py --user UC3Mm_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset phishing --Nworkers 4 &

taskset --cpu-list 1 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M0_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset phishing --Nworkers 4 --id 0 &

taskset --cpu-list 2 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M1_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset phishing --Nworkers 4  --id 1 & 

taskset --cpu-list 3 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M2_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset phishing --Nworkers 4 --id 2 & 

taskset --cpu-list 4 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M3_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset phishing --Nworkers 4 --id 3 & 


income

taskset --cpu-list 5 python3 pom1_FBSVM_master_pycloudmessenger_nonid.py --user UC3Mm_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset income --Nworkers 4 &

taskset --cpu-list 1 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M0_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset income --Nworkers 4 --id 0 &

taskset --cpu-list 2 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M1_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset income --Nworkers 4  --id 1 & 

taskset --cpu-list 3 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M2_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset income --Nworkers 4 --id 2 & 

taskset --cpu-list 4 python3 pom1_FBSVM_worker_pycloudmessenger_nonid.py --user UC3M3_rskfnrt6 --password passUC3M --task_name TaskUC3M_rskfnrt6 --dataset income --Nworkers 4 --id 3 & 

