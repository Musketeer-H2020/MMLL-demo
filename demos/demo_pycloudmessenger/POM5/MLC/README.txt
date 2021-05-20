==================================================================
 Demo execution instructions 
==================================================================

Open 6 bash terminals and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

Execute these lines, one at every terminal:

python3 pom5_MLC_master_pycloudmessenger.py --dataset M-iris --verbose 1

(wait for the master to start listening for workers/cryptonode...)

python3 pom5_MLC_worker_pycloudmessenger.py --id 0 --dataset M-iris --verbose 1

python3 pom5_MLC_worker_pycloudmessenger.py --id 1 --dataset M-iris --verbose 1 
python3 pom5_MLC_worker_pycloudmessenger.py --id 2 --dataset M-iris --verbose 1 
python3 pom5_MLC_worker_pycloudmessenger.py --id 3 --dataset M-iris --verbose 1 
python3 pom5_MLC_worker_pycloudmessenger.py --id 4 --dataset M-iris --verbose 1 


---------------------
DEBUG 
---------------------

python3 pom5_MLC_master_pycloudmessenger.py --dataset M-iris --verbose 1

python3 pom5_MLC_worker_pycloudmessenger.py --id 0 --dataset M-iris --verbose 1

python3 pom5_MLC_worker_pycloudmessenger.py --id 1 --dataset M-iris --verbose 1 &
python3 pom5_MLC_worker_pycloudmessenger.py --id 2 --dataset M-iris --verbose 1 &
python3 pom5_MLC_worker_pycloudmessenger.py --id 3 --dataset M-iris --verbose 1 &
python3 pom5_MLC_worker_pycloudmessenger.py --id 4 --dataset M-iris --verbose 1 &

