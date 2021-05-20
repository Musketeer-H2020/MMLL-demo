==================================================================
 Demo execution instructions 
==================================================================

Open 6 bash terminals and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

Execute these lines, one at every terminal:

python3 pom5_LC_master_pycloudmessenger.py --dataset pima --verbose 1

(wait for the master to start listening for workers...)

python3 pom5_LC_worker_pycloudmessenger.py --id 0 --dataset pima --verbose 1

python3 pom5_LC_worker_pycloudmessenger.py --id 1 --dataset pima --verbose 1 
python3 pom5_LC_worker_pycloudmessenger.py --id 2 --dataset pima --verbose 1 
python3 pom5_LC_worker_pycloudmessenger.py --id 3 --dataset pima --verbose 1 
python3 pom5_LC_worker_pycloudmessenger.py --id 4 --dataset pima --verbose 1 

---------------------
DEBUG 
---------------------

python3 pom5_LC_master_pycloudmessenger.py --dataset pima --verbose 1

python3 pom5_LC_worker_pycloudmessenger.py --id 0 --dataset pima --verbose 1

python3 pom5_LC_worker_pycloudmessenger.py --id 1 --dataset pima --verbose 1 &
python3 pom5_LC_worker_pycloudmessenger.py --id 2 --dataset pima --verbose 1 &
python3 pom5_LC_worker_pycloudmessenger.py --id 3 --dataset pima --verbose 1 &
python3 pom5_LC_worker_pycloudmessenger.py --id 4 --dataset pima --verbose 1 &

