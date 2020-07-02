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
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 1 --dataset Bmnist --verbose 1 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 2 --dataset Bmnist --verbose 1 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 3 --dataset Bmnist --verbose 1 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 4 --dataset Bmnist --verbose 1 


=====================================================
Normalization Demo: training LC_pm with pima raw data
=====================================================

---------------------
Single terminal demo: 
---------------------

For a demo on a single terminal, with less messages shown on screen, simply paste these lines:

python3 pom6_LC_pm_master_pycloudmessenger_normalization.py --dataset pima_raw --verbose 0 &
sleep 20
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 0 --dataset pima_raw --verbose 0 & 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 1 --dataset pima_raw --verbose 0 &
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 2 --dataset pima_raw --verbose 0 & 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 3 --dataset pima_raw --verbose 0 & 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 4 --dataset pima_raw --verbose 0 & 

---------------------
Full detail demo: 
---------------------

python3 pom6_LC_pm_master_pycloudmessenger_normalization.py --dataset pima_raw --verbose 1

(wait for the master to start listening for workers...)

python3 pom6_LC_pm_worker_pycloudmessenger.py --id 0 --dataset pima_raw --verbose 1 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 1 --dataset pima_raw --verbose 1 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 2 --dataset pima_raw --verbose 1 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 3 --dataset pima_raw --verbose 1 
python3 pom6_LC_pm_worker_pycloudmessenger.py --id 4 --dataset pima_raw --verbose 1 

