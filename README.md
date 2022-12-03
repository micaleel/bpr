# BPR

## Installation

```bash
conda create -n bpr python=3.8
conda env update --name bpr --file environment.yml
```

## Compiling Cython

```bash
conda activate bpr
make cython
```

## Running Experiments

```bash
conda activate bpr
python bpr/carpe_diem.py
python plot_more_results.py
```

## Datasets
#### CiteULike

- <https://github.com/js05212/citeulike-a>
- <https://github.com/js05212/citeulike-t>*
- <https://paperswithcode.com/sota/recommendation-systems-on-citeulike>
- <https://cornac.readthedocs.io/en/latest/_modules/cornac/datasets/citeulike.html>
- <http://www.wanghao.in/CDL.htm>
#### MIND 

- <https://www.kaggle.com/arashnic/mind-news-dataset?select=behaviors.tsv>