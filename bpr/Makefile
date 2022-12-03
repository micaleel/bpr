install:
	conda env create -f environment.yml

cython:
	python setup.py build_ext --inplace

run: cython
	python similarity.py

train: cython
	time python train_simple.py

train_ml100k: cython
	time python train_simple.py ml100k

train_ml1m: cython
	time python train_simple.py ml1m

train_mind: cython
	time python train_simple.py mind

train_amazon: cython
	time python train_simple.py amazon

train_gowalla: cython
	time python train_simple.py gowalla

train_lastfm: cython
	time python train_simple.py lastfm
