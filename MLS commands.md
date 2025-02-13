MLS:

connect to informatics VPN, run ssh <UUN>@mlp.inf.ed.ac.uk
remotely connect to DICE machine, run ssh UUN@ssh.inf.ed.ac.uk

access thru ssh and use SLURM commands

can run code - interactive job: srun --gres=gpu:1 --pty bash , run nvidia-smi   srun --gres=gpu:titan_x:1 --pty bash
batch job: sbatch --gres=gpu:1 <.sh_file>
#!/bin/bash

8 1060s, 4 titan x, 1 titan x pascal, 2 A600 max at one  time

check all types: scontril show node | grep gpu
check SLURM job status: squeue
cancel a job: scancel <job_id>
