==================================================================================
Pre-processing, normalization, data alignment estimation and data value estimation
==================================================================================
 Demo execution instructions using pycloudmessenger
==================================================================


######################   Check_data DEMO
# Wrong data in workers 0 y 2

python3 pom4_check_data.py --dataset income_raw --verbose 1

python3 pom4_worker.py --dataset redwine_raw --verbose 1 --id 0

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 1 &
sleep 2
python3 pom4_worker.py --dataset redwine_raw --verbose 1 --id 2 &
sleep 2
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 3 &
sleep 2
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5


####################### Data conversion to numeric format DEMO

python3 pom4_data_to_numeric.py --dataset income_raw --verbose 1

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 0

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 1 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 2 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 3 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5


####################### Noise injection  DEMO

python3 pom4_noise_injection.py --dataset income_raw --verbose 1

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 0

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 1 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 2 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 3 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5

####################### Data alignment estimation DEMO

python3 pom4_data_alignment_estimation.py --dataset income_raw_bad24 --verbose 1

python3 pom4_worker.py --dataset income_raw_bad24 --verbose 1 --id 0

python3 pom4_worker.py --dataset income_raw_bad24 --verbose 1 --id 1 &
python3 pom4_worker.py --dataset income_raw_bad24 --verbose 1 --id 2 &
python3 pom4_worker.py --dataset income_raw_bad24 --verbose 1 --id 3 &
python3 pom4_worker.py --dataset income_raw_bad24 --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5


######################   Ad-hoc Preprocessing DEMO

python3 pom4_adhoc_preprocess.py --dataset income_raw --verbose 1

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 0 

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 1 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 2 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 3 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5


######################   Image reshape DEMO

python3 pom4_image_reshape_preprocessing.py --dataset mnist_raw_matrix_binclass --verbose 1

python3 pom4_worker.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 0

python3 pom4_worker.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 1 &
python3 pom4_worker.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 2 &
python3 pom4_worker.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 3 &
python3 pom4_worker.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5

####################### Data normalization DEMO

python3 pom4_normalization.py --dataset income_raw --verbose 1

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 0

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 1 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 2 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 3 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5


#######################   Principal Component Analysis DEMO 

python3 pom4_principal_component_analysis.py --dataset income_raw --verbose 1

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 0

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 1 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 2 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 3 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5


#######################   Greedy feature selection  DEMO

python3 pom4_greedy_feature_selection.py --dataset income_raw --verbose 1

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 0

python3 pom4_worker.py --dataset income_raw --verbose 1 --id 1 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 2 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 3 &
python3 pom4_worker.py --dataset income_raw --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5


######################   Feature frequency selection   DEMO

python3 pom4_feature_selection_frequency.py --dataset news20_all_sparse --verbose 1

python3 pom4_worker.py --dataset news20_all_sparse --verbose 1 --id 0

python3 pom4_worker.py --dataset news20_all_sparse --verbose 1 --id 1 &
python3 pom4_worker.py --dataset news20_all_sparse --verbose 1 --id 2 &
python3 pom4_worker.py --dataset news20_all_sparse --verbose 1 --id 3 &
python3 pom4_worker.py --dataset news20_all_sparse --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5


######################   Random_projection

python3 pom4_random_projection.py --dataset news20_all_sparse --verbose 1

python3 pom4_worker.py --dataset news20_all_sparse --verbose 1 --id 0

python3 pom4_worker.py --dataset news20_all_sparse --verbose 1 --id 1 &
python3 pom4_worker.py --dataset news20_all_sparse --verbose 1 --id 2 &
python3 pom4_worker.py --dataset news20_all_sparse --verbose 1 --id 3 &
python3 pom4_worker.py --dataset news20_all_sparse --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5


######################   Deep Learning preprocessing DEMO

python3 pom4_deep_learning_preprocessing.py --dataset mnist_raw_matrix_binclass --verbose 1

python3 pom4_worker.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 0

python3 pom4_worker.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 1 &
python3 pom4_worker.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 2 &
python3 pom4_worker.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 3 &
python3 pom4_worker.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5


######################   Natural Language Processing preprocessing (TFIDF) DEMO

python3 pom4_natural_language_processing.py --dataset 20news_bow_bin --verbose 1

python3 pom4_worker.py --dataset 20news_bow_bin --verbose 1 --id 0

python3 pom4_worker.py --dataset 20news_bow_bin --verbose 1 --id 1 &
python3 pom4_worker.py --dataset 20news_bow_bin --verbose 1 --id 2 &
python3 pom4_worker.py --dataset 20news_bow_bin --verbose 1 --id 3 &
python3 pom4_worker.py --dataset 20news_bow_bin --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5

#######################   Record Linkage DEMO

python3 pom4_record_linkage.py --dataset income_raw_V --verbose 1

python3 pom4_worker_V.py --dataset income_raw_V --verbose 1 --id 0

python3 pom4_worker_V.py --dataset income_raw_V --verbose 1 --id 1 &
sleep 2
python3 pom4_worker_V.py --dataset income_raw_V --verbose 1 --id 2 &
sleep 2
python3 pom4_worker_V.py --dataset income_raw_V --verbose 1 --id 3 &
sleep 2
python3 pom4_worker_V.py --dataset income_raw_V --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5


#######################   Missing Data Imputation DEMO 

python3 pom4_missing_data_imputation.py --dataset income_raw_V_missing --verbose 1

python3 pom4_worker_V.py --dataset income_raw_V_missing --verbose 1 --id 0 &
sleep 2
python3 pom4_worker_V.py --dataset income_raw_V_missing --verbose 1 --id 1 &
sleep 2
python3 pom4_worker_V.py --dataset income_raw_V_missing --verbose 1 --id 2 &
sleep 2
python3 pom4_worker_V.py --dataset income_raw_V_missing --verbose 1 --id 3 &


python3 pom4_worker_V.py --dataset income_raw_V_missing --verbose 1 --id 4

python3 pom4_crypto.py --verbose 1 --id 5

####################### Data value estimation DEMO (a priori)

python3 pom4_data_value_estimation.py --dataset income_dv_small --verbose 1

python3 pom4_worker.py --dataset income_dv_small --verbose 1 --id 0

python3 pom4_worker.py --dataset income_dv_small --verbose 1 --id 1 &
sleep 2
python3 pom4_worker.py --dataset income_dv_small --verbose 1 --id 2 &
sleep 2
python3 pom4_worker.py --dataset income_dv_small --verbose 1 --id 3 &
sleep 2
python3 pom4_worker.py --dataset income_dv_small --verbose 1 --id 4 &

python3 pom4_crypto.py --verbose 1 --id 5

####################### Data value estimation DEMO (a posteriori)

python3 pom4_data_value_estimation_pos.py --dataset income_dv_small --verbose 1

python3 pom4_worker.py --dataset income_dv_small --verbose 1 --id 0

python3 pom4_worker.py --dataset income_dv_small --verbose 1 --id 2 &
sleep 2
python3 pom4_worker.py --dataset income_dv_small --verbose 1 --id 3 &
sleep 2
python3 pom4_worker.py --dataset income_dv_small --verbose 1 --id 4 &
sleep 2
python3 pom4_worker.py --dataset income_dv_small --verbose 1 --id 1 

python3 pom4_crypto.py --verbose 1 --id 5
