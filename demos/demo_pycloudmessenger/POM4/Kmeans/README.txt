==================================================================
 Demo execution instructions 
==================================================================

Open eight bash terminal and activate the environment in everyone of them (conda activate demo) and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

***************** NOTE ****************** 
POM4 algorithms require some preliminary data processing (dataset encrypting, transmitting deblinding, etc.) among Master, Cryptonode and Workers, so it may take a while before the actual learning takes place. Please be patient.

Execute these lines, one at every terminal:

python3 pom4_Kmeans_master_pycloudmessenger.py --dataset synth2D --verbose 1

(wait for the master to start listening for workers/cryptonode...)

python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 0  
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 1
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 2
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 3
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 4
python3 pom4_Kmeans_crypto_pycloudmessenger.py --verbose 1 --id 5

---------------------
DEBUG 
---------------------

python3 pom4_Kmeans_master_pycloudmessenger.py --dataset synth2D --verbose 1

python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 0  &

python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 1 &
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 2 &
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 3 &
python3 pom4_Kmeans_worker_pycloudmessenger.py --dataset synth2D --verbose 1 --id 4 &

python3 pom4_Kmeans_crypto_pycloudmessenger.py --verbose 1 --id 5

