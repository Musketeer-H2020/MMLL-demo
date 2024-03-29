==================================================================
 Demo execution instructions 
==================================================================

Open eight bash terminal and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

***************** NOTE ****************** 
POM4 algorithms require some preliminary data processing (dataset encrypting, transmitting deblinding, etc.) among Master, Cryptonode and Workers, so it may take a while before the actual learning takes place. Please be patient.

Execute these lines, one at every terminal:

python3 pom4_LC_master_pycloudmessenger.py --dataset pima --verbose 1

(wait for the master to start listening for workers/cryptonode...)

python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 0
python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 1 
python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 2  
python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 3 
python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 4 

python3 pom4_LC_crypto_pycloudmessenger.py --verbose 1 --id 5


---------------------
DEBUG 
---------------------

python3 pom4_LC_master_pycloudmessenger.py --dataset pima --verbose 1

python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 0 &
   
python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 1 &  
python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 2 &  
python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 3 & 
python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 4 & 

python3 pom4_LC_crypto_pycloudmessenger.py --verbose 1 --id 5


---------------------
Train + a posteriori DVE 
---------------------

python3 pom4_LC_master_pycloudmessenger.py --dataset pima_DV --verbose 1 --use_dve 1 --dve_weight 1 

python3 pom4_LC_worker_pycloudmessenger.py --id 0 --dataset pima_DV --verbose 1 &
sleep 5
python3 pom4_LC_worker_pycloudmessenger.py --id 1 --dataset pima_DV --verbose 1 &
sleep 5
python3 pom4_LC_worker_pycloudmessenger.py --id 2 --dataset pima_DV --verbose 1 &
sleep 5
python3 pom4_LC_worker_pycloudmessenger.py --id 3 --dataset pima_DV --verbose 1 &
sleep 5
python3 pom4_LC_worker_pycloudmessenger.py --id 4 --dataset pima_DV --verbose 1 &
sleep 5
python3 pom4_LC_crypto_pycloudmessenger.py --verbose 1 --id 5 &



python3 pom4_LC_master_pycloudmessenger.py --dataset income_DV --verbose 1 --use_dve 1 --dve_weight 1 

python3 pom4_LC_worker_pycloudmessenger.py --id 0 --dataset income_DV --verbose 1 &
sleep 5
python3 pom4_LC_worker_pycloudmessenger.py --id 1 --dataset income_DV --verbose 1 &
sleep 5
python3 pom4_LC_worker_pycloudmessenger.py --id 2 --dataset income_DV --verbose 1 &
sleep 5
python3 pom4_LC_worker_pycloudmessenger.py --id 3 --dataset income_DV --verbose 1 &
sleep 5
python3 pom4_LC_worker_pycloudmessenger.py --id 4 --dataset income_DV --verbose 1 &
sleep 5
python3 pom4_LC_crypto_pycloudmessenger.py --verbose 1 --id 5 &


---------------------
With Defenses
---------------------

python3 pom4_LC_master_pycloudmessenger_aggregator.py --dataset pima --verbose 1

(wait for the master to start listening for workers/cryptonode...)

python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 0

python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 1 &
python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 2 &
python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 3 &
python3 pom4_LC_worker_pycloudmessenger.py --dataset pima --verbose 1 --id 4 &
python3 pom4_LC_crypto_pycloudmessenger.py --verbose 1 --id 5

