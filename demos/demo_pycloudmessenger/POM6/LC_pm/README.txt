==================================================================
 Demo execution instructions using pycloudmessenger under Linux OS
==================================================================

Open six bash terminal and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

-------------------------------------------
Execute these lines, one at every terminal:
-------------------------------------------

python3 pom6_LC_pm_master_pycloudmessenger.py --dataset Bmnist --verbose 1

(wait for the master to start listening for workers...)

python pom6_LC_pm_worker_pycloudmessenger.py --id 0 --dataset Bmnist --verbose 1 
python pom6_LC_pm_worker_pycloudmessenger.py --id 1 --dataset Bmnist --verbose 1 
python pom6_LC_pm_worker_pycloudmessenger.py --id 2 --dataset Bmnist --verbose 1 
python pom6_LC_pm_worker_pycloudmessenger.py --id 3 --dataset Bmnist --verbose 1 
python pom6_LC_pm_worker_pycloudmessenger.py --id 4 --dataset Bmnist --verbose 1 

