all:
	@echo BPR

install:
	conda create -n bpr python=3.8 --yes
	conda env update --name bpr --file environment.yml

uninstall:
	conda env remove --name bpr

cython:
	@cd bpr/ && python setup.py build_ext --inplace

black:
	black .

cancel:
	scancel ucdbpr

submit:
	sbatch bpr.sh

push:
	git archive --output=./bpr.zip --format=zip HEAD
	scp bpr.zip khalil@kay.ichec.ie:/ichec/home/users/khalil
	rm bpr.zip

pull:
	scp -r khalil@kay.ichec.ie:/ichec/home/users/khalil/bpr/data/outputs .

peek:
	squeue -A EuroCC-AF-6

test:
	pytest -v ./tests