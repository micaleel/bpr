import traceback
import sys
import pickle

try:
    from train_simple import *
except ImportError:
    from .train_simple import (
        RandomSampler,
        SimilaritySampler,
        RecencyBiasSampler,
        DualSampler,
        DualSampler2,
        wandb_logger,
        train_bpr,
    )


alg = sys.argv[1]
data = load_ml100k()
n_samples = len(data.train_df) * 10
folder = "bpr_grid"

if alg == "random":
    sampler = RandomSampler(data, n_samples)
    name = "random"

elif alg == "sim":
    exponent = float(sys.argv[2])
    sampler = SimilaritySampler(data, n_samples, epxonent=exponent)
    name = "sim_{}".format(exponent)

elif alg == "RB":
    se_0 = float(sys.argv[2])
    warm_up = int(sys.argv[3])
    window_size = int(sys.argv[4])
    sampler = RecencyBiasSampler(
        data, n_samples, warm_up=warm_up, se_0=se_0, window_size=window_size
    )
    name = "RB_{}_{}_{}".format(se_0, warm_up, window_size)

elif alg == "Dual":
    se_0 = float(sys.argv[2])
    warm_up = int(sys.argv[3])
    window_size = int(sys.argv[4])
    exponent = float(sys.argv[5])
    sampler = DualSampler(
        data,
        n_samples,
        warm_up=warm_up,
        se_0=se_0,
        window_size=window_size,
        exponent=exponent,
    )
    name = "Dual_{}_{}_{}_{}".format(se_0, warm_up, window_size, exponent)

elif alg == "Dual2":
    se_0 = float(sys.argv[2])
    warm_up = int(sys.argv[3])
    window_size = int(sys.argv[4])
    exponent = float(sys.argv[5])
    sampler = DualSampler2(
        data,
        n_samples,
        warm_up=warm_up,
        se_0=se_0,
        window_size=window_size,
        exponent=exponent,
    )
    name = "Dual2_{}_{}_{}_{}".format(se_0, warm_up, window_size, exponent)

wandb_config = {"sampler": name}

try:
    with wandb_logger(
        name="ml100k-bpr_grid-{}".format(name),
        wandb_project="ml100k_bpr_grid",
        wandb_config=wandb_config,
    ) as wlogger:
        result = train_bpr(data, sampler, wlogger=wlogger)
        f_name = "{}/{}.pkl".format(folder, name)
        with open(f_name, "wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
except:
    f_name = "{}/{}.txt".format(folder, name)
    with open(f_name, "w") as handle:
        print(traceback.format_exc())
        handle.write(traceback.format_exc())
