from multiprocessing import Pool
import os
import subprocess

n_workers = 5

expers = {
    "random": {},
    "sim": {"exponent": [0.0, 0.5, 1.0, 5.0, 10.0]},
    "RB": {
        "se_0": [1.0, 10.0, 0.5, 0.0],
        "warm_up": [0, 5, 10, 20],
        "window_size": [2, 5, 10, 20],
    },
    "Dual": {
        "se_0": [1.0, 10.0, 0.5, 0.0],
        "warm_up": [0],
        "window_size": [2, 5, 10, 20],
        "exponent": [0.0, 0.5, 1.0, 5.0, 10.0],
    },
    "Dual2": {
        "se_0": [1.0, 10.0, 0.5, 0.0],
        "warm_up": [0],
        "window_size": [2, 5, 10, 20],
        "exponent": [0.0, 0.5, 1.0, 5.0, 10.0],
    },
}


experiments = []


# experiments.append('random')
# for exponent in expers['sim']['exponent']:
#     experiments.append('sim {}'.format(exponent))

# for se_0 in expers['RB']['se_0']:
#     for warm_up in expers['RB']['warm_up']:
#         for window_size in expers['RB']['window_size']:

#             experiments.append('RB {} {} {}'.format(se_0, warm_up, window_size))

# for se_0 in expers['Dual']['se_0']:
#     for exponent in expers['Dual']['exponent']:
#         for window_size in expers['Dual']['window_size']:

#             experiments.append('Dual {} {} {} {}'.format(se_0, 0, window_size, exponent))


for se_0 in expers["Dual2"]["se_0"]:
    for exponent in expers["Dual2"]["exponent"]:
        for window_size in expers["Dual2"]["window_size"]:
            experiments.append(
                "Dual2 {} {} {} {}".format(se_0, 0, window_size, exponent)
            )

_experiments = []

for exper in experiments:
    _experiments.append("python grid_runner.py {}".format(exper))

experiments = _experiments


def _f(name):
    subprocess.call(name.split(" "))


with Pool(n_workers) as p:
    p.map(_f, experiments)
