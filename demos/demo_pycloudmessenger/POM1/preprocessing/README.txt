==================================================================================
Pre-processing, normalization, data alignment estimation and data value estimation
==================================================================================
 Demo execution instructions using pycloudmessenger
==================================================================

Each execution line in this file represents one terminal (one participant), they can be in different machines.

-------------------------------------------
Execute these lines, one at every terminal. Start launching the master, and wait until it is ready to receive workers before launching the workers.

Once the training is completed, these demo scripts produce the output files in the results/ folder (models, figures, logs)
-------------------------------------------

Parameters:
    - user: String with the name of the user. If the user does not exist in the pycloudmessenger platform a new one will be created.
    - password: String with the password.
    - task_name: String with the name of the task.
    - id: Integer representing the partition of data to be used by the worker. Each worker should use a different partition, possible values are 0 to 4.
    - dataset: Name of the dataset to be used.

Important notes:
    - The task_name should be the same for all terminals in each demo.
    - Each user should have a different id, otherwise they will be training using the same dataset partition.
    - Each terminal should have a different user.
-------------------------------------------


==================================================================
 Check data demo with wrong data in worker 1
==================================================================
python pom1_check_data.py --user <user> --password <password> --task_name <task_name> --dataset income_raw
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset mnist --id 1


==================================================================
 Data conversion to numeric format demo
==================================================================
python pom1_data_to_numeric.py --user <user> --password <password> --task_name <task_name> --dataset income_raw
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw --id 1


==================================================================
 Data alignment estimation demo
==================================================================
python pom1_data_alignment_estimation.py --user <user> --password <password> --task_name <task_name> --dataset income_raw_bad24
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw_bad24 --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw_bad24 --id 1
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw_bad24 --id 2


==================================================================
 Ad-hoc preprocessing demo
==================================================================
python pom1_adhoc_preprocess.py --user <user> --password <password> --task_name <task_name> --dataset income_raw
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw --id 1


==================================================================
 Image reshape demo
==================================================================
python pom1_image_reshape_preprocessing.py --user <user> --password <password> --task_name <task_name> --dataset mnist_raw_matrix_binclass
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset mnist_raw_matrix_binclass --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset mnist_raw_matrix_binclass --id 1


==================================================================
 Normalization demo
==================================================================
python pom1_normalization.py --user <user> --password <password> --task_name <task_name> --dataset income_raw
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw --id 1


==================================================================
 Principal component analysis demo
==================================================================
python pom1_principal_component_analysis.py --user <user> --password <password> --task_name <task_name> --dataset income_raw
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw --id 1


==================================================================
 Greedy feature selection demo
==================================================================
python pom1_greedy_feature_selection.py --user <user> --password <password> --task_name <task_name> --dataset income_raw
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset income_raw --id 1


==================================================================
 Feature frequency selection demo
==================================================================
python pom1_feature_selection_frequency.py --user <user> --password <password> --task_name <task_name> --dataset news20_all_sparse
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset news20_all_sparse --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset news20_all_sparse --id 1


==================================================================
 Random projection demo
==================================================================
python pom1_random_projection.py --user <user> --password <password> --task_name <task_name> --dataset news20_all_sparse
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset news20_all_sparse --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset news20_all_sparse --id 1


==================================================================
 Deep Learning preprocessing demo
==================================================================
python pom1_deep_learning_preprocessing.py --user <user> --password <password> --task_name <task_name> --dataset mnist_raw_matrix_binclass
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset mnist_raw_matrix_binclass --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset mnist_raw_matrix_binclass --id 1


==================================================================
 Natural language processing (TFIDF) demo
==================================================================
python pom1_natural_language_processing.py --user <user> --password <password> --task_name <task_name> --dataset 20news_bow_bin
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset 20news_bow_bin --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset 20news_bow_bin --id 1


==================================================================
 Record linkage demo
==================================================================
python pom1_record_linkage.py --user <user> --password <password> --task_name <task_name> --dataset income_raw_V
python pom1_worker_V.py --user <user> --password <password> --task_name <task_name> --dataset income_raw_V --id 0
python pom1_worker_V.py --user <user> --password <password> --task_name <task_name> --dataset income_raw_V --id 1


==================================================================
 Missing data imputation demo
==================================================================
python pom1_missing_data_imputation.py --user <user> --password <password> --task_name <task_name> --dataset income_raw_V_missing
python pom1_worker_V.py --user <user> --password <password> --task_name <task_name> --dataset income_raw_V_missing --id 0
python pom1_worker_V.py --user <user> --password <password> --task_name <task_name> --dataset income_raw_V_missing --id 1


==================================================================
 Data value estimation demo (a priori)
==================================================================
python pom1_data_value_estimation.py --user <user> --password <password> --task_name <task_name> --dataset mnist
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset mnist --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset mnist --id 1
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset mnist --id 2


==================================================================
 Data value estimation demo (a posteriori)
==================================================================
python pom1_data_value_estimation_pos.py --user <user> --password <password> --task_name <task_name> --dataset mnist
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset mnist --id 0
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset mnist --id 1
python pom1_worker.py --user <user> --password <password> --task_name <task_name> --dataset mnist --id 2

