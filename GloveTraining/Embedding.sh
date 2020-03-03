#!/bin/bash                                                                                                                                                            
#SBATCH -p sched_mit_sloan_interactive
#SBATCH --output=embd.out                                                   
#SBATCH --error=embd.err                                                                                                                                          
#SBATCH --job-name embedding                                                                                                                                     
#SBATCH --mem-per-cpu 16G                                                                                                                                            
#SBATCH --cpus-per-task 8                                                                                                                                             
#SBATCH --time 2-00:00:00                                                                                                                                             
#SBATCH --mail-type END                                                                                                                                                
#SBATCH --mail-user fcaprass@mit.edu                                                                                                                                  
 
module load python/3.6.3
module load sloan/python/modules/3.6
srun python3 Embedding.py

