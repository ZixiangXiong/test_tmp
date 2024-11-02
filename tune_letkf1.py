import os
import numpy as np
p1 = np.arange(0.1, 0.2, 0.1,dtype=np.float64)
p2 = np.arange(1000., 9001., 100.)
##create_shell_file:
for i in p1:
    for j in p2:
        with open(r"/gpfs/home/zx18/PycharmProjects/pythonProject/shell_file/20percent_arctan/my_script{}_{}.sh".format(int(j),int(1000*i)), "w") as f:
            print("#!/bin/bash", file=f)
            print("#SBATCH --job-name='sqg_enkf_obDensity_user_hpc_20p.py'", file=f)
            print("#SBATCH --nodes=1                     # Number of nodes", file=f)
            print("#SBATCH --ntasks-per-node=1           # Number of tasks per node", file=f)
            print("#SBATCH --cpus-per-task=1             # Number of CPU cores per task", file=f)
            print("#SBATCH -A chipilskigroup_q", file=f)
            print("#SBATCH --exclusive", file=f)
            print("#SBATCH -t 24:00:00", file=f)
            print("module load pycharm/2024.1.5", file=f)
            print("srun python sqg_enkf_obDensity_user_hpc_20p.py {} {} > /gpfs/home/zx18/PycharmProjects/pythonProject/output/output_{}_{}.txt".format(i, j,int(j),int(1000*i)), file=f)


##run_shell_file:
for i in p1:
    for j in p2:
        os.system("sbatch '/gpfs/home/zx18/PycharmProjects/pythonProject/shell_file/20percent_arctan/my_script{}_{}.sh'".format(int(j),int(1000*i)))

