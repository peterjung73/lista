from utils.noise import GaussianNoise
from utils.train_utils import *
from utils.all_models import *

import torch

# try to use gpu
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Using: " + device)

# set randomness
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
Exemplary file to run experiments
=================================

The example runs some experiments that were evaluated for the paper.
Executing all runs required for complete reproduction takes quite some time. We provide results for seed=0 in the 
res/model folder.

You can define your own parameterizations of the models as in the utils.all_models file and use them here, 
e.g. changing the hidden layer size of NA-ALISTA.

All experiments are executed on a GPU if available.
Expected duration using a GPU for one run of NA_ALISTA, n=1000, k=16, noise=40db with the default settings below
will be approximately one hour (400 epochs).

"""

model_dir = 'res/models/'

# Default settings for reproducing our experiments.
M_DEFAULT = 250  # measurements
S_DEFAULT = 50  # sparsity
LR_DEFAULT = 0.2 * 10e-4  # learning rate

func_dict = {fn.__name__: fn for fn in
             (NA_ALISTA_UR_128, ALISTA_AT, ALISTA, FISTA, ISTA, AGLISTA, NA_ALISTA_U_128, NA_ALISTA_R_128)}
func_names = tuple(func_dict.keys())


def run(args):

    model_func = func_dict[args.model_func]
    m = args.measurements
    s = args.sparsity
    k = args.k
    n = args.n
    lr = args.learning_rate

    noisefn = GaussianNoise(args.noise)
    noisename = f"GaussianNoise{args.noise:.0f}"

    epoch = 100 + 20 * k

    # apply the p-trick
    p = (np.linspace((s * 1 * 1.2) // k, s * 1 * 1.2, k)).astype(int)

    params = {
        'model': args.model_func,
        'm': m,
        's': s,
        'k': k,
        'noise': noisename,
        'n': n,
    }

    # filename for saving, do not change if you intend to use plot_results.ipynb
    name = '__'.join([f"{k}={v}" for k, v in params.items()])

    print(f"Running: {name}")

    # trains and saves model along with some training metrics
    train_model(m=m, n=n, s=s, k=k, p=p,
                model_fn=model_func,
                noise_fn=noisefn,
                epochs=epoch,
                initial_lr=lr,
                name=name,
                model_dir=model_dir)

    print("Done.")


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Code to reproduce the results from Neurally Augmented ALISTA, '
                                                 'ICLR 2021 by Freya Behrens, Jonathan Sauder, Peter Jung.')

    parser.add_argument("--measurements", "-m", type=int, default=M_DEFAULT, help="Number of measurements")
    parser.add_argument("--sparsity", "-s", type=float, default=S_DEFAULT, help="Sparsity level")
    parser.add_argument("--learning-rate", "-l", type=float, default=LR_DEFAULT, help="Learning rate")
    parser.add_argument("-k", type=int, help="Number of iterations that the ISTA-style method is executed")
    parser.add_argument("-n", type=int, help="Input size")
    parser.add_argument("--noise", "-N", type=float, help="Reference SNR level to produce Gaussian Noise")
    parser.add_argument("--model-func", "-f", type=str, choices=func_names, metavar="MODEL_FN",
                        help=f"Model function to choose from {func_names}")

    run(parser.parse_args())
