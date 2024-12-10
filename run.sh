# add slurm specific code here:

echo "Running on GPU"

for i in {1..33}; do
  echo "Number of qubits currently being simulated: $i"
  if [ $i -eq 1 ] # use when the job terminates to start from a particular node and layer. Set the 
  # layer you want to start from on line 10.
  then
    for j in {1..3}; do
      echo "Number of layers currently being simulated: $j"
      # python qaoa.py $i $j nvidia COBYLA
      singularity exec --nv build/cudaq-env.sif python src/base/qaoa.py $i $j nvidia COBYLA
    done
  else
    for j in {1..3}; do
      echo "Number of layers currently being simulated: $j"
      # python qaoa.py $i $j nvidia COBYLA
      singularity exec --nv build/cudaq-env.sif python src/base/qaoa.py $i $j nvidia COBYLA
    done
  fi
done
