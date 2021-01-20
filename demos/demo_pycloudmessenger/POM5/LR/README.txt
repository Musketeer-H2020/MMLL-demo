==================================================================
 Demo execution instructions using pycloudmessenger
==================================================================

---------------------
Single terminal demo: 
---------------------

For a demo on a single terminal, with less messages shown on screen, simply paste these lines:


python3 pom5_LR_master_pycloudmessenger.py --dataset redwine --verbose 0 &
sleep 20
python3 pom5_LR_worker_pycloudmessenger.py --id 0 --dataset redwine --verbose 0 &
python3 pom5_LR_worker_pycloudmessenger.py --id 1 --dataset redwine --verbose 0 &
python3 pom5_LR_worker_pycloudmessenger.py --id 2 --dataset redwine --verbose 0 &
python3 pom5_LR_worker_pycloudmessenger.py --id 3 --dataset redwine --verbose 0 &
python3 pom5_LR_worker_pycloudmessenger.py --id 4 --dataset redwine --verbose 0 &


---------------------
Full detail demo: 
---------------------

Open six bash terminal and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.
Execute these lines, one at every terminal:

python3 pom5_LR_master_pycloudmessenger.py --dataset redwine --verbose 1

(wait for the master to start listening for workers...)

python3 pom5_LR_worker_pycloudmessenger.py --dataset redwine --verbose 1 --id 0
python3 pom5_LR_worker_pycloudmessenger.py --dataset redwine --verbose 1 --id 1
python3 pom5_LR_worker_pycloudmessenger.py --dataset redwine --verbose 1 --id 2
python3 pom5_LR_worker_pycloudmessenger.py --dataset redwine --verbose 1 --id 3
python3 pom5_LR_worker_pycloudmessenger.py --dataset redwine --verbose 1 --id 4

windows

python pom5_LR_master_pycloudmessenger.py --dataset redwine --verbose 1
python pom5_LR_worker_pycloudmessenger.py --dataset redwine --verbose 1 --id 


===========================================================
Preprocessing demos
===========================================================
python3 pom5_LR_master_pycloudmessenger.py --dataset redwine --verbose 1

(wait for the master to start listening for workers...)

python3 pom5_LR_worker_pycloudmessenger.py --dataset redwine --verbose 1 --id 0
python3 pom5_LR_worker_pycloudmessenger.py --dataset redwine --verbose 1 --id 1
python3 pom5_LR_worker_pycloudmessenger.py --dataset redwine --verbose 1 --id 2
python3 pom5_LR_worker_pycloudmessenger.py --dataset redwine --verbose 1 --id 3
python3 pom5_LR_worker_pycloudmessenger.py --dataset redwine --verbose 1 --id 4


debug:

python3 pom5_LR_master_pycloudmessenger.py --dataset redwine --verbose 1

python3 pom5_LR_worker_pycloudmessenger.py --id 0 --dataset redwine --verbose 1

python3 pom5_LR_worker_pycloudmessenger.py --id 1 --dataset redwine --verbose 1 &
python3 pom5_LR_worker_pycloudmessenger.py --id 2 --dataset redwine --verbose 1 &
python3 pom5_LR_worker_pycloudmessenger.py --id 3 --dataset redwine --verbose 1 &
python3 pom5_LR_worker_pycloudmessenger.py --id 4 --dataset redwine --verbose 1 &

