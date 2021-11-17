==================================================================
 Demo execution instructions 
==================================================================

Open 6 bash terminals and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

Execute these lines, one at every terminal:

python3 pom6_MLC_master_pycloudmessenger.py --dataset M-iris --verbose 1

(wait for the master to start listening for workers/cryptonode...)

python3 pom6_MLC_worker_pycloudmessenger.py --id 0 --dataset M-iris --verbose 1

python3 pom6_MLC_worker_pycloudmessenger.py --id 1 --dataset M-iris --verbose 1 
python3 pom6_MLC_worker_pycloudmessenger.py --id 2 --dataset M-iris --verbose 1
python3 pom6_MLC_worker_pycloudmessenger.py --id 3 --dataset M-iris --verbose 1 
python3 pom6_MLC_worker_pycloudmessenger.py --id 4 --dataset M-iris --verbose 1 


---------------------
DEBUG 
---------------------

python3 pom6_MLC_master_pycloudmessenger.py --dataset M-iris_norm --use_dve 1 --dve_weight 0 --verbose 1

python3 pom6_MLC_worker_pycloudmessenger.py --id 0 --dataset M-iris_norm --verbose 1 &
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 1 --dataset M-iris_norm --verbose 1 &
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 2 --dataset M-iris_norm --verbose 1 & 
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 3 --dataset M-iris_norm --verbose 1 & 
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 4 --dataset M-iris_norm --verbose 1 & 


python3 pom6_MLC_master_pycloudmessenger.py --dataset M-anuran --use_dve 1 --dve_weight 1 --verbose 1

python3 pom6_MLC_worker_pycloudmessenger.py --id 0 --dataset M-anuran --verbose 1 &
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 1 --dataset M-anuran --verbose 1 &
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 2 --dataset M-anuran --verbose 1 & 
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 3 --dataset M-anuran --verbose 1 & 
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 4 --dataset M-anuran --verbose 1 & 

---------------------
Train + a posteriori DVE 
---------------------

python3 pom6_MLC_master_pycloudmessenger.py --dataset M-iris_norm_DV --use_dve 1 --dve_weight 1 --verbose 1

python3 pom6_MLC_worker_pycloudmessenger.py --id 0 --dataset M-iris_norm_DV --verbose 1 &
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 1 --dataset M-iris_norm_DV --verbose 1 &
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 2 --dataset M-iris_norm_DV --verbose 1 & 
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 3 --dataset M-iris_norm_DV --verbose 1 & 
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 4 --dataset M-iris_norm_DV --verbose 1 & 


python3 pom6_MLC_master_pycloudmessenger.py --dataset M-anuran_DV --use_dve 1 --dve_weight 1 --verbose 1

python3 pom6_MLC_worker_pycloudmessenger.py --id 0 --dataset M-anuran_DV --verbose 1 &
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 1 --dataset M-anuran_DV --verbose 1 &
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 2 --dataset M-anuran_DV --verbose 1 & 
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 3 --dataset M-anuran_DV --verbose 1 & 
sleep 5
python3 pom6_MLC_worker_pycloudmessenger.py --id 4 --dataset M-anuran_DV --verbose 1 & 


---------------------
With Defenses
---------------------

python3 pom6_MLC_master_pycloudmessenger_aggregator.py --dataset M-iris_norm --verbose 1

python3 pom6_MLC_worker_pycloudmessenger.py --id 0 --dataset M-iris_norm --verbose 1

python3 pom6_MLC_worker_pycloudmessenger.py --id 1 --dataset M-iris_norm --verbose 1 &
python3 pom6_MLC_worker_pycloudmessenger.py --id 2 --dataset M-iris_norm --verbose 1 & 
python3 pom6_MLC_worker_pycloudmessenger.py --id 3 --dataset M-iris_norm --verbose 1 & 
python3 pom6_MLC_worker_pycloudmessenger.py --id 4 --dataset M-iris_norm --verbose 1 & 



localflask

python3 local_flask_server.py

python3 pom6_MLC_master_localflask.py --dataset M-anuran --use_dve 1 --dve_weight 0 --verbose 1

python3 pom6_MLC_worker_localflask.py --id 0 --dataset M-anuran --verbose 1 &
sleep 5
python3 pom6_MLC_worker_localflask.py --id 1 --dataset M-anuran --verbose 1 &
sleep 5
python3 pom6_MLC_worker_localflask.py --id 2 --dataset M-anuran --verbose 1 & 
sleep 5
python3 pom6_MLC_worker_localflask.py --id 3 --dataset M-anuran --verbose 1 & 
sleep 5
python3 pom6_MLC_worker_localflask.py --id 4 --dataset M-anuran --verbose 1 & 

