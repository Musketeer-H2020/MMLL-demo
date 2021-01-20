==================================================================
 Demo execution instructions using pycloudmessenger under Linux OS
==================================================================

---------------------
Single terminal demo: 
---------------------

For a demo on a single terminal, with less messages shown on screen, simply paste these lines:

python3 pom6_MLC_pm_master_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 0 &
sleep 20
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 0 --id 0 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 0  --id 1 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 0  --id 2 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 0  --id 3 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 0  --id 4 &

---------------------
Full detail demo: 
---------------------

Open seven bash terminal and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

-------------------------------------------
Execute these lines, one at every terminal:
-------------------------------------------

python3 pom6_MLC_pm_master_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1

(wait for the master to start listening for workers...)

python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1  --id 0
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1  --id 1
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1  --id 2
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1  --id 3
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1  --id 4


python pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1  --id 2


python3 pom6_MLC_pm_master_pycloudmessenger.py --dataset M-mnist-dlp100-noniid --verbose 1

(wait for the master to start listening for workers...)

python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100-noniid --verbose 1  --id 0 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100-noniid --verbose 1  --id 1 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100-noniid --verbose 1  --id 2 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100-noniid --verbose 1  --id 3 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100-noniid --verbose 1  --id 4


---------------------
Single terminal demo: 
---------------------

For a demo on a single terminal, with less messages shown on screen, simply paste these lines:

python3 pom6_MLC_pm_master_pycloudmessenger.py --dataset M-mnist --verbose 0 &
sleep 20
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 0 --id 0 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 0  --id 1 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 0  --id 2 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 0  --id 3 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 0  --id 4 &

---------------------
Full detail demo: 
---------------------

Open seven bash terminal and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

-------------------------------------------
Execute these lines, one at every terminal:
-------------------------------------------

python3 pom6_MLC_pm_master_pycloudmessenger.py --dataset M-mnist --verbose 1

(wait for the master to start listening for workers...)

python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 1  --id 0
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 1  --id 1
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 1  --id 2
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 1  --id 3
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 1  --id 4

windows
python pom6_MLC_pm_master_pycloudmessenger.py --dataset M-mnist --verbose 1
python pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 1 --id 

python pom6_MLC_pm_master_pycloudmessenger.py --dataset M-mnist-small --verbose 1
python pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-small --verbose 1 --id 

python pom6_MLC_pm_master_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1
python pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1 --id 

# Checking timeout=0.1 with new pycloudmessenger

python3 pom6_MLC_pm_master_pycloudmessenger.py --dataset M-mnist --verbose 1 &
sleep 20
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 1 --id 0 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 1  --id 1 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 1  --id 2 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 1  --id 3 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist --verbose 1  --id 4 &

Debug:

python3 pom6_MLC_pm_master_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1

python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1 --id 0

python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1 --id 1 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1 --id 2 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1 --id 3 &
python3 pom6_MLC_pm_worker_pycloudmessenger.py --dataset M-mnist-dlp100 --verbose 1 --id 4 &

