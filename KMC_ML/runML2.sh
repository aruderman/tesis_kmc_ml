#!/bin/bash
#SBATCH --job-name=in-g0
#SBATCH --partition=short
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00-01:00:00

. /etc/profile
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

# Cargar módulos
module load gcc/12.3.0

# Definir listas
xi_values=(-4.0 -3.43307087 -2.86614173 -2.2992126 -1.73228346 -1.16535433 -0.5984252 -0.03149606 0.53543307 1.1023622 1.66929134 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2)
ell_values=(-1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -4.0 -3.43307087 -2.86614173 -2.2992126 -1.73228346 -1.16535433 -0.5984252 -0.03149606 0.53543307 1.1023622 1.66929134)
##ell_values=(-4.0 -3.43307087 -2.86614173 -2.2992126 -1.73228346)
##xi_values=(0.2 0.2 0.2 0.2 0.2)

# Máximo de trabajos simultáneos
MAX_PARALLEL=8

# Loop sobre índices
for i in "${!ell_values[@]}"; do
    xi=${xi_values[$i]}
    ell=${ell_values[$i]}

    echo "Lanzando cálculo con xi=$xi, ell=$ell"

    srun --ntasks=1 --cpus-per-task=32 ./kmc_galvanoT "$xi" "$ell" &

    # Control de concurrencia
    if [[ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL ]]; then
        # Esperar a que termine al menos 1
        wait -n
    fi
done

# Esperar a que terminen todos los que queden
wait

