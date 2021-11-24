==================================================================
 Demo execution instructions 
==================================================================

Open 6 bash terminals and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

Execute these lines, one at every terminal:

python3 pom6_LC_master_pycloudmessenger.py --dataset pima --verbose 1

(wait for the master to start listening for workers/cryptonode...)

python3 pom6_LC_worker_pycloudmessenger.py --id 0 --dataset pima --verbose 1

python3 pom6_LC_worker_pycloudmessenger.py --id 1 --dataset pima --verbose 1 
python3 pom6_LC_worker_pycloudmessenger.py --id 2 --dataset pima --verbose 1 
python3 pom6_LC_worker_pycloudmessenger.py --id 3 --dataset pima --verbose 1 
python3 pom6_LC_worker_pycloudmessenger.py --id 4 --dataset pima --verbose 1


---------------------
DEBUG 
---------------------

python3 pom6_LC_master_pycloudmessenger.py --dataset pima --verbose 1

python3 pom6_LC_worker_pycloudmessenger.py --id 0 --dataset pima --verbose 1

python3 pom6_LC_worker_pycloudmessenger.py --id 1 --dataset pima --verbose 1 &
python3 pom6_LC_worker_pycloudmessenger.py --id 2 --dataset pima --verbose 1 & 
python3 pom6_LC_worker_pycloudmessenger.py --id 3 --dataset pima --verbose 1 & 
python3 pom6_LC_worker_pycloudmessenger.py --id 4 --dataset pima --verbose 1 & 


---------------------
Train + a posteriori DVE 
---------------------

python3 pom6_LC_master_pycloudmessenger.py --dataset income_DV --use_dve 1 --dve_weight 1 --verbose 1

python3 pom6_LC_worker_pycloudmessenger.py --id 0 --dataset income_DV --verbose 1 &
sleep 5
python3 pom6_LC_worker_pycloudmessenger.py --id 1 --dataset income_DV --verbose 1 &
sleep 5
python3 pom6_LC_worker_pycloudmessenger.py --id 2 --dataset income_DV --verbose 1 & 
sleep 5
python3 pom6_LC_worker_pycloudmessenger.py --id 3 --dataset income_DV --verbose 1 & 
sleep 5
python3 pom6_LC_worker_pycloudmessenger.py --id 4 --dataset income_DV --verbose 1 & 


python3 pom6_LC_master_pycloudmessenger.py --dataset pima_DV --use_dve 1 --dve_weight 1 --verbose 1

python3 pom6_LC_worker_pycloudmessenger.py --id 0 --dataset pima_DV --verbose 1 &
sleep 5
python3 pom6_LC_worker_pycloudmessenger.py --id 1 --dataset pima_DV --verbose 1 &
sleep 5
python3 pom6_LC_worker_pycloudmessenger.py --id 2 --dataset pima_DV --verbose 1 & 
sleep 5
python3 pom6_LC_worker_pycloudmessenger.py --id 3 --dataset pima_DV --verbose 1 & 
sleep 5
python3 pom6_LC_worker_pycloudmessenger.py --id 4 --dataset pima_DV --verbose 1 & 


---------------------
With Defenses
---------------------

python3 pom6_LC_master_pycloudmessenger_aggregator.py --dataset pima --verbose 1

python3 pom6_LC_worker_pycloudmessenger.py --id 0 --dataset pima --verbose 1 &

python3 pom6_LC_worker_pycloudmessenger.py --id 1 --dataset pima --verbose 1 &
python3 pom6_LC_worker_pycloudmessenger.py --id 2 --dataset pima --verbose 1 & 
python3 pom6_LC_worker_pycloudmessenger.py --id 3 --dataset pima --verbose 1 & 
python3 pom6_LC_worker_pycloudmessenger.py --id 4 --dataset pima --verbose 1 & 


