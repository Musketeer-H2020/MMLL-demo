==================================================================
 Demo execution instructions using pycloudmessenger under Linux OS
==================================================================

Open six bash terminals and execute any of the following scripts to see the corresponding demo.

Every terminal represents one participant, they can be in different machines.

-------------------------------------------
Execute these lines, one at every terminal. Start launching the master, and wait until it is ready to receive workers before launching the workers.

Once the training is completed, these demo scripts produce the output files in the results/ folder (models, figures, logs)
-------------------------------------------

python3 pom6_Kmeans_pm_master_pycloudmessenger.py --dataset synth2D --verbose 1
python pom6_Kmeans_pm_worker_pycloudmessenger.py --id 0 --dataset synth2D --verbose 1 
python pom6_Kmeans_pm_worker_pycloudmessenger.py --id 1 --dataset synth2D --verbose 1 
python pom6_Kmeans_pm_worker_pycloudmessenger.py --id 2 --dataset synth2D --verbose 1 
python pom6_Kmeans_pm_worker_pycloudmessenger.py --id 3 --dataset synth2D --verbose 1 
python pom6_Kmeans_pm_worker_pycloudmessenger.py --id 4 --dataset synth2D --verbose 1 

