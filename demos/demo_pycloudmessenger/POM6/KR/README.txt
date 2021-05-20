==================================================================
 Demo execution instructions 
==================================================================

Open 6 bash terminals and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

Execute these lines, one at every terminal:

python3 pom6_KR_master_pycloudmessenger.py --dataset sinc1D --verbose 1

(wait for the master to start listening for workers/cryptonode...)

python3 pom6_KR_worker_pycloudmessenger.py --dataset sinc1D --verbose 1 --id 0

python3 pom6_KR_worker_pycloudmessenger.py --dataset sinc1D --verbose 1 --id 1 
python3 pom6_KR_worker_pycloudmessenger.py --dataset sinc1D --verbose 1 --id 2 
python3 pom6_KR_worker_pycloudmessenger.py --dataset sinc1D --verbose 1 --id 3 
python3 pom6_KR_worker_pycloudmessenger.py --dataset sinc1D --verbose 1 --id 4 


---------------------
DEBUG 
---------------------

python3 pom6_KR_master_pycloudmessenger.py --dataset sinc1D --verbose 1

python3 pom6_KR_worker_pycloudmessenger.py --dataset sinc1D --verbose 1 --id 0

python3 pom6_KR_worker_pycloudmessenger.py --dataset sinc1D --verbose 1 --id 1 &
python3 pom6_KR_worker_pycloudmessenger.py --dataset sinc1D --verbose 1 --id 2 &
python3 pom6_KR_worker_pycloudmessenger.py --dataset sinc1D --verbose 1 --id 3 &
python3 pom6_KR_worker_pycloudmessenger.py --dataset sinc1D --verbose 1 --id 4 &

