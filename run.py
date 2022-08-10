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

synth_dir = 'res/models/'
com_dir = 'res/models-com/'


def FISTA_COM(m, n, s, k, p, forward_op, backward_op, L_):
    return algo.FISTA(m, n, k, forward_op, backward_op, 1, 0.003)


func_synth = {fn.__name__: fn for fn in (NA_ALISTA_UR_128, ALISTA_AT, ALISTA, FISTA,
                                         ISTA, AGLISTA, NA_ALISTA_U_128, NA_ALISTA_R_128)}
func_com = {fn.__name__: fn for fn in (ALISTA, FISTA, ISTA, AGLISTA)}
func_com['FISTA'] = FISTA_COM

# Default settings for reproducing our synthetic data experiments.
_synth_defaults = {'m': 250,                        # measurements
                   's': 50,                         # sparsity
                   'lr': 0.2 * 10e-4,               # learning rate
                   'fn': tuple(func_synth.keys())   # available functions
                   }

# Default settings for reproducing our communication experiments.
_com_defaults = {'m': 100,                          # measurements
                 's': 8,                            # sparsity
                 'lr': 0.05 * 10e-4,                # learning rate
                 'fn': tuple(func_com.keys())     # available functions
                 }


def run_synth(args):

    model_func = func_synth[args.model_func]
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
                model_dir=synth_dir)

    print("Done.")


def run_com(args):


    # for model_func in [ALISTA, FISTA, ISTA, AGLISTA]:
    #
    #     for k in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]: # number of iterations that the ISTA-style method is executed
    #
    #         epoch = 70 + 9 * k
    #
    #         for n in [256]: # input size
    #
    #             for noisename, noisefn in [["GaussianNoise10", GaussianNoise(10)]]:

    model_func = func_com[args.model_func]
    m = args.measurements
    s = args.sparsity
    k = args.k
    n = args.n
    lr = args.learning_rate

    noisefn = GaussianNoise(args.noise)
    noisename = f"GaussianNoise{args.noise:.0f}"

    epoch = 70 + 9 * k

    # apply the p-trick
    p = (np.linspace((s * 1 * 1.5) // k, s * 1 * 1.5, k)).astype(int)
    p = p.clip(3, 10)
    params = {
        'model': model_func.__name__,
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
    train_model_communication(m=m, n=n, s=s, k=k, p=p,
                              model_fn=params['model'],
                              noise_fn=noisefn,
                              epochs=epoch,
                              initial_lr=lr,
                              name=name,
                              model_dir=com_dir)

    print("Done.")


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Code to reproduce the results from Neurally Augmented ALISTA, '
                                                 'ICLR 2021 by Freya Behrens, Jonathan Sauder, Peter Jung.',
                                     epilog='For command-specific information, execute `run.py [command] --help`')
    subparsers = parser.add_subparsers(dest='command')
    synth = subparsers.add_parser('synthetic', description="Experiments with synthetic data")
    com = subparsers.add_parser('communication', description="Experiments for communication")

    synth.add_argument("--measurements", "-m", type=int, default=_synth_defaults['m'], help="Number of measurements")
    synth.add_argument("--sparsity", "-s", type=float, default=_synth_defaults['s'], help="Sparsity level")
    synth.add_argument("--learning-rate", "-l", type=float, default=_synth_defaults['lr'], help="Learning rate")
    synth.add_argument("-k", type=int, help="Number of iterations that the ISTA-style method is executed")
    synth.add_argument("-n", type=int, help="Input size")
    synth.add_argument("--noise", "-N", type=float, help="Reference SNR level to produce Gaussian Noise")
    synth.add_argument("--model-func", "-f", type=str, choices=_synth_defaults['fn'], metavar="MODEL_FN",
                       help=f"Model function to choose from {_synth_defaults['fn']}")

    com.add_argument("--measurements", "-m", type=int, default=_com_defaults['m'], help="Number of measurements")
    com.add_argument("--sparsity", "-s", type=float, default=_com_defaults['s'], help="Sparsity level")
    com.add_argument("--learning-rate", "-l", type=float, default=_com_defaults['lr'], help="Learning rate")
    com.add_argument("-k", type=int, help="Number of iterations that the ISTA-style method is executed")
    com.add_argument("-n", type=int, help="Input size")
    com.add_argument("--noise", "-N", type=float, help="Reference SNR level to produce Gaussian Noise")
    com.add_argument("--model-func", "-f", type=str, choices=_com_defaults['fn'], metavar="MODEL_FN",
                     help=f"Model function to choose from {_com_defaults['fn']}")

    parsed_args = parser.parse_args()
    if parsed_args.command == 'synthetic':
        run_synth(parsed_args)
    elif parsed_args.command == 'communication':
        run_com(parsed_args)
    else:
        print(f"Unrecognized command '{parsed_args.command}'. Exiting with code 1")
        exit(1)
