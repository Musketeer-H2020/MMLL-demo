==================================================================
 Demo execution instructions using pycloudmessenger
==================================================================

---------------------
Single terminal demo with pima: 
---------------------

For a demo on a single terminal, with less messages shown on screen, simply paste these lines:

python3 pom6_LC_pm_master_pycloudmessenger.py --dataset pima --verbose 0 &
sleep 20
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 0 --dataset pima --verbose 0 &
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 1 --dataset pima --verbose 0 & 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 2 --dataset pima --verbose 0 & 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 3 --dataset pima --verbose 0 & 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 4 --dataset pima --verbose 0 & 

---------------------
Full detail demo: 
---------------------

Open six bash terminal and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.
Execute these lines, one at every terminal:

python3 pom6_LC_pm_master_pycloudmessenger.py --dataset pima --verbose 1

(wait for the master to start listening for workers...)

python3 pom6_LC_pm_worker_pycloudmessenger.py --id 0 --dataset pima --verbose 1 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 1 --dataset pima --verbose 1 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 2 --dataset pima --verbose 1 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 3 --dataset pima --verbose 1 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 4 --dataset pima --verbose 1 


---------------------
Single terminal demo with Bminst: 
---------------------

For a demo on a single terminal, with less messages shown on screen, simply paste these lines:

python3 pom6_LC_pm_master_pycloudmessenger.py --dataset Bmnist --verbose 0 &
sleep 20
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 0 --dataset Bmnist --verbose 0 &
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 1 --dataset Bmnist --verbose 0 & 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 2 --dataset Bmnist --verbose 0 & 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 3 --dataset Bmnist --verbose 0 & 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 4 --dataset Bmnist --verbose 0 & 

---------------------
Full detail demo: 
---------------------

Open six bash terminal and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.
Execute these lines, one at every terminal:

python3 pom6_LC_pm_master_pycloudmessenger.py --dataset Bmnist --verbose 1

(wait for the master to start listening for workers...)

python3 pom6_LC_pm_worker_pycloudmessenger.py --id 0 --dataset Bmnist --verbose 1 

python3 pom6_LC_pm_worker_pycloudmessenger.py --id 1 --dataset Bmnist --verbose 1 &
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 2 --dataset Bmnist --verbose 1 &
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 3 --dataset Bmnist --verbose 1 &
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 4 --dataset Bmnist --verbose 1 &


Debug: 

python3 pom6_LC_pm_master_pycloudmessenger.py --dataset pima --verbose 1

python3 pom6_LC_pm_worker_pycloudmessenger.py --id 0 --dataset pima --verbose 1

python3 pom6_LC_pm_worker_pycloudmessenger.py --id 1 --dataset pima --verbose 1 &
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 2 --dataset pima --verbose 1 & 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 3 --dataset pima --verbose 0 & 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 4 --dataset pima --verbose 0 & 






















######################   Check_data DEMO
# Metemos datos incorrectos en workers 0 y 2

python pom6_LC_pm_master_pycloudmessenger_check_data.py --dataset income_raw --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset redwine_raw --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 1 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger.py --dataset redwine_raw --verbose 1 --id 2 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 3 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 4 &


######################   Preprocess DEMO

python pom6_LC_pm_master_pycloudmessenger_preprocess.py --dataset income_raw --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 1 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 2 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 3 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 4 &

#######################   Feature selection  DEMO

python pom6_LC_pm_master_pycloudmessenger_gfs.py --dataset income_raw --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 1 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 2 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 3 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 4 &


MNIST

python pom6_LC_pm_master_pycloudmessenger_gfs.py --dataset mnist_binclass --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_binclass --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_binclass --verbose 1 --id 1 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_binclass --verbose 1 --id 2 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_binclass --verbose 1 --id 3 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_binclass --verbose 1 --id 4 &




#######################   Data alignment estimation  DEMO

with good workers: 

python pom6_LC_pm_master_pycloudmessenger_da.py --dataset income_raw --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 1 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 2 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 3 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 4 &

-------------------------------------------------------
with bad workers 2 and 4: 
python pom6_LC_pm_master_pycloudmessenger_da.py --dataset income_raw_bad24 --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw_bad24 --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw_bad24 --verbose 1 --id 1 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw_bad24 --verbose 1 --id 2 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw_bad24 --verbose 1 --id 3 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw_bad24 --verbose 1 --id 4 &


#######################   PCA  DEMO income_raw

python pom6_LC_pm_master_pycloudmessenger_pca.py --dataset income_raw --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 1 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 2 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 3 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset income_raw --verbose 1 --id 4 &


#######################   PCA  DEMO mnist

python pom6_LC_pm_master_pycloudmessenger_pca.py --dataset mnist_binclass --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_binclass --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_binclass --verbose 1 --id 1 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_binclass --verbose 1 --id 2 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_binclass --verbose 1 --id 3 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_binclass --verbose 1 --id 4 &

######################   Deep Learning preprocessing DEMO

python pom6_LC_pm_master_pycloudmessenger_dlp.py --dataset mnist_raw_matrix_binclass --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 1 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 2 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 3 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset mnist_raw_matrix_binclass --verbose 1 --id 4 &

######################   Natural Language Processing preprocessing DEMO

python pom6_LC_pm_master_pycloudmessenger_nlp.py --dataset 20news_bow_bin --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset 20news_bow_bin --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset 20news_bow_bin --verbose 1 --id 1 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset 20news_bow_bin --verbose 1 --id 2 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset 20news_bow_bin --verbose 1 --id 3 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset 20news_bow_bin --verbose 1 --id 4 &

######################   Random_projection + PCA  DEMO

python pom6_LC_pm_master_pycloudmessenger_rp.py --dataset news20_all_sparse --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset news20_all_sparse --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset news20_all_sparse --verbose 1 --id 1 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset news20_all_sparse --verbose 1 --id 2 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset news20_all_sparse --verbose 1 --id 3 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset news20_all_sparse --verbose 1 --id 4 &

######################   Feature freq selection   DEMO

python pom6_LC_pm_master_pycloudmessenger_feat_freq.py --dataset news20_all_sparse --verbose 1

python pom6_LC_pm_worker_pycloudmessenger.py --dataset news20_all_sparse --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger.py --dataset news20_all_sparse --verbose 1 --id 1 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset news20_all_sparse --verbose 1 --id 2 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset news20_all_sparse --verbose 1 --id 3 &
python pom6_LC_pm_worker_pycloudmessenger.py --dataset news20_all_sparse --verbose 1 --id 4 &


#######################   DEMO income_raw_V

python pom6_LC_pm_master_pycloudmessenger_V.py --dataset income_raw_V --verbose 1

python pom6_LC_pm_worker_pycloudmessenger_V.py --dataset income_raw_V --verbose 1 --id 0

python pom6_LC_pm_worker_pycloudmessenger_V.py --dataset income_raw_V --verbose 1 --id 1 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger_V.py --dataset income_raw_V --verbose 1 --id 2 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger_V.py --dataset income_raw_V --verbose 1 --id 3 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger_V.py --dataset income_raw_V --verbose 1 --id 4 &


#######################   DEMO income_raw_V with missing values

python pom6_LC_pm_master_pycloudmessenger_V_missing.py --dataset income_raw_V_missing --verbose 1

python pom6_LC_pm_worker_pycloudmessenger_V.py --dataset income_raw_V_missing --verbose 1 --id 0 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger_V.py --dataset income_raw_V_missing --verbose 1 --id 1 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger_V.py --dataset income_raw_V_missing --verbose 1 --id 2 &
sleep 2
python pom6_LC_pm_worker_pycloudmessenger_V.py --dataset income_raw_V_missing --verbose 1 --id 3 &


python pom6_LC_pm_worker_pycloudmessenger_V.py --dataset income_raw_V_missing --verbose 1 --id 4
