install: 
	singularity build --fakeroot cudaq-env.sif Apptainerfile.def
	mkdir build
	mv cudaq-env.sif build/

run:
	bash run.sh

analyze:
	python src/analysis/time_vs_numqubits.py


