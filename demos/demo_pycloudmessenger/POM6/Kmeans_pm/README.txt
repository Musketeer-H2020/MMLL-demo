==================================================================
 Demo execution instructions using pycloudmessenger under Linux OS
==================================================================

Open six bash terminal and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

-------------------------------------------
Execute these lines, one at every terminal:
-------------------------------------------

python3 pom6_Kmeans_pm_master_pycloudmessenger.py --dataset synth2D --verbose 1

(wait for the master to start listening for workers...)

python pom6_Kmeans_pm_worker_pycloudmessenger.py --id 0 --dataset synth2D --verbose 1 
python pom6_Kmeans_pm_worker_pycloudmessenger.py --id 1 --dataset synth2D --verbose 1 
python pom6_Kmeans_pm_worker_pycloudmessenger.py --id 2 --dataset synth2D --verbose 1 
python pom6_Kmeans_pm_worker_pycloudmessenger.py --id 3 --dataset synth2D --verbose 1 
python pom6_Kmeans_pm_worker_pycloudmessenger.py --id 4 --dataset synth2D --verbose 1 





python3 pom6_XC_master_pycloudmessenger.py --dataset redwine --verbose 1
(wait for the master to start listening for workers...)
python pom6_XC_worker_pycloudmessenger.py --id 0workerA --dataset redwine --verbose 1 
python pom6_XC_worker_pycloudmessenger.py --id 1workerB --dataset redwine --verbose 1 
python pom6_XC_worker_pycloudmessenger.py --id 2workerC --dataset redwine --verbose 1 
python pom6_XC_worker_pycloudmessenger.py --id 3workerD --dataset redwine --verbose 1 
python pom6_XC_worker_pycloudmessenger.py --id 4workerE --dataset redwine --verbose 1 

python3 pom6_RR_master_pycloudmessenger.py --dataset redwine --verbose 1
(wait for the master to start listening for workers...)
python pom6_RR_worker_pycloudmessenger.py --id 0workerA --dataset redwine --verbose 1 
python pom6_RR_worker_pycloudmessenger.py --id 1workerB --dataset redwine --verbose 1 
python pom6_RR_worker_pycloudmessenger.py --id 2workerC --dataset redwine --verbose 1 
python pom6_RR_worker_pycloudmessenger.py --id 3workerD --dataset redwine --verbose 1 
python pom6_RR_worker_pycloudmessenger.py --id 4workerE --dataset redwine --verbose 1 

python3 pom6_RR_master_pycloudmessenger.py --dataset redwine --verbose 1
(wait for the master to start listening for workers...)
python pom6_RR_worker_pycloudmessenger.py --id 0workerA --dataset redwine --verbose 1 
python pom6_RR_worker_pycloudmessenger.py --id 1workerB --dataset redwine --verbose 1 
python pom6_RR_worker_pycloudmessenger.py --id 2workerC --dataset redwine --verbose 1 
python pom6_RR_worker_pycloudmessenger.py --id 3workerD --dataset redwine --verbose 1 
python pom6_RR_worker_pycloudmessenger.py --id 4workerE --dataset redwine --verbose 1 

python3 pom6_LC_pm_master_pycloudmessenger.py --dataset pima --verbose 1
(wait for the master to start listening for workers...)
python pom6_LC_pm_worker_pycloudmessenger.py --id 0workerA --dataset pima --verbose 1 
python pom6_LC_pm_worker_pycloudmessenger.py --id 1workerB --dataset pima --verbose 1 
python pom6_LC_pm_worker_pycloudmessenger.py --id 2workerC --dataset pima --verbose 1 
python pom6_LC_pm_worker_pycloudmessenger.py --id 3workerD --dataset pima --verbose 1 
python pom6_LC_pm_worker_pycloudmessenger.py --id 4workerE --dataset pima --verbose 1 


