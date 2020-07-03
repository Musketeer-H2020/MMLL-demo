==================================================================
 Demo execution instructions using pycloudmessenger under Linux OS
==================================================================

---------------------
Single terminal demo: 
---------------------

For a demo on a single terminal, with less messages shown on screen, simply paste these lines:

** NOTE **: POM4 algorithms require some preliminary data processing (encrypting, transmitting deblinding, etc.) among Master, Cryptonode and workers, so it may take a while before the actual learning takes place.   

python3 pom4_Kmeans_master_pycloudmessenger.py --dataset synth2D --verbose 0 &
sleep 20
python3 pom4_Kmeans_crypto_pycloudmessenger.py --verbose 0 --id 5 &
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 0 --id 0 &
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 0 --id 1 &
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 0 --id 2 &
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 0 --id 3 &
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 0 --id 4 &


---------------------
Full detail demo: 
---------------------

Open eight bash terminal and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.
Execute these lines, one at every terminal:

** NOTE **: POM4 algorithms require some preliminary data processing (encrypting, transmitting deblinding, etc.) among Master, Cryptonode and workers, so it may take a while before the actual learning takes place.   

python3 pom4_Kmeans_master_pycloudmessenger.py --dataset synth2D --verbose 1

(wait for the master to start listening for workers/cryptonode...)

python3 pom4_Kmeans_crypto_pycloudmessenger.py --verbose 1 --id 5

python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 0  
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 1 
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 2 
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 3 
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 4 
