# Neurally-Augmented ALISTA

Code to reproduce the results from [Neurally Augmented ALISTA, ICLR 2021](https://openreview.net/forum?id=q_S44KLQ_Aa).
Freya Behrens, Jonathan Sauder, Peter Jung.

### Synthetic Data Experiments

Experiment demo:
```console
python run.py synthetic --param1 val1 --param2 val2 ...
```
for parameterization, refer to the command help:
```console
usage: run.py synthetic [-h] [--measurements MEASUREMENTS]                   
                        [--sparsity SPARSITY] [--learning-rate LEARNING_RATE]
                        [-k K] [-n N] [--noise NOISE] [--model-func MODEL_FN]

Experiments with synthetic data

optional arguments:
  -h, --help            show this help message and exit
  --measurements MEASUREMENTS, -m MEASUREMENTS
                        Number of measurements
  --sparsity SPARSITY, -s SPARSITY
                        Sparsity level
  --learning-rate LEARNING_RATE, -l LEARNING_RATE
                        Learning rate
  -k K                  Number of iterations that the ISTA-style method is
                        executed
  -n N                  Input size
  --noise NOISE, -N NOISE
                        Reference SNR level to produce Gaussian Noise
  --model-func MODEL_FN, -f MODEL_FN
                        Model function to choose from ('NA_ALISTA_UR_128',
                        'ALISTA_AT', 'ALISTA', 'FISTA', 'ISTA', 'AGLISTA',
                        'NA_ALISTA_U_128', 'NA_ALISTA_R_128')
```

Reproduce the figures from the paper:
```console
jupyter notebook plot_results.ipynb
```

### Communication Experiments

```console
python run.py communication --param1 val1 --param2 val2 ...
```
for parameterization, refer to the command help:
```console
usage: run.py communication [-h] [--measurements MEASUREMENTS]
                            [--sparsity SPARSITY]
                            [--learning-rate LEARNING_RATE] [-k K] [-n N]
                            [--noise NOISE] [--model-func MODEL_FN]

Experiments for communication

optional arguments:
  -h, --help            show this help message and exit
  --measurements MEASUREMENTS, -m MEASUREMENTS
                        Number of measurements
  --sparsity SPARSITY, -s SPARSITY
                        Sparsity level
  --learning-rate LEARNING_RATE, -l LEARNING_RATE
                        Learning rate
  -k K                  Number of iterations that the ISTA-style method is
                        executed
  -n N                  Input size
  --noise NOISE, -N NOISE
                        Reference SNR level to produce Gaussian Noise
  --model-func MODEL_FN, -f MODEL_FN
                        Model function to choose from ('ALISTA', 'FISTA',
                        'ISTA', 'AGLISTA')
```

Reproduce the figures from the paper:
```console
jupyter notebook plot_results_communication.ipynb
```


### Remarks

- Without a GPU the experiments take a lot of time.
- The directory [cluster](./cluster) includes some auxiliary files to run the experiments
as parallel jobs in an HPC cluster with
[Singularity](https://docs.sylabs.io/guides/3.6/user-guide/) and [Slurm](https://slurm.schedmd.com/).
