#!/bin/bash

#SBATCH -o /home/iml/hannah.eichhorn/workspace/iml-dl_framework/jobs_output/slurm_%j.job
#SBATCH -e /home/iml/hannah.eichhorn/workspace/iml-dl_framework/jobs_output/slurm_%j.job
#SBATCH -J recon_t2star
#SBATCH -p gpu_p

#SBATCH --qos gpu_reservation
#SBATCH --reservation=iml_user
#SBATCH --exclude=gpusrv41,supergpu02,supergpu03,supergpu05
#SBATCH --gres=gpu:1
#SBATCH --mem=48G #Memory required   #80GB for downstream tasks
#SBATCH -t 127:59:59 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.

echo "$HOME"
source "$HOME"/.bashrc

source "$HOME"/anaconda3/bin/activate #conda
chmod 600 "$HOME"/workspace/iml-dl_framework/jobs_output/slurm_"$SLURM_JOB_ID".job #create the job file
cd $HOME/workspace/iml-dl_framework/iml-dl/

# activate the conda environment
conda activate dev_hannah
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/iml/hannah.eichhorn/anaconda3/envs/iml_py308/lib/:/home/iml/hannah.eichhorn/anaconda3/lib/
# launch the experiment
echo 'Starting Script'
wandb online   # comment this out once wandb is allowed on cluster
python -u ./core/Main.py --config_path ./projects/recon_t2star/configs/config_train_miccai.yaml
