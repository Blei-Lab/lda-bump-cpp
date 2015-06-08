lda-bump-cpp
============

Latent Dirichlet allocation (LDA) with bumping variational inference.

[![](http://img.shields.io/badge/license-GPLv2-red.svg?style=flat)](https://github.com/akucukelbir/lda-bump-cpp/blob/master/LICENSE)

Implements three versions of LDA
* Coordinate ascent (mean-field)
* Stochastic variational inference (https://github.com/Blei-Lab/onlineldavb)
* Bumping variational inference [1]

[1] Alp Kucukelbir and David M Blei. _Population Empirical Bayes._
Uncertainty in Artificial Intelligence (UAI) 2015.


Requirements
------------

`lda-bump-cpp` is written in C++11. It requires a modern compiler. It also
depends on Eigen 3, Boost, and CMake. It uses docopt (provided).

* Eigen 3: http://eigen.tuxfamily.org/
* Boost: http://www.boost.org/
* CMake: http://www.cmake.org/

Refer to platform-specific instructions for installation.
(I recommend homebrew on Mac OS X.)


Instructions to Build and Run
-----------------------------

The driver (main) program runs all three algorithms.

```
cmake .
make driver
```

A toy dataset of arXiv abstracts is provided.

Example
```
./driver --topics=5
         --vocabulary=data/arxiv-vocab.dat
         --datatr=data/arxiv-train-5k.dat
         --datatest=data/arxiv-test-1k.dat
```

For more help, run
```
./driver -h

lda-bump-cpp LDA with bumping variational inference.

Usage:
  driver --topics=NUM_TOPICS --vocabulary=VOCAB
         --datatr=TRAIN --datatest=TEST
         [--bootstrap=NUM_BOOTSTRAP] [--minibatch=MINIBATCH]
         [--alpha=ALPHA] [--eta=ETA]
         [--tau0=TAU0] [--kappa=KAPPA]
         [--fixed_step_size=STEPSIZE]
         [--max_itr=MAX_ITR]
         [--compute_elbo]
  driver (-h | --help)
  driver --version

Options:
  --topics=NUM_TOPICS        Number of topics for LDA
  --vocabulary=VOCAB         Vocabulary, one word per line
  --datatr=TRAIN             Training data in LDA-C format
  --datatest=TEST            Testing  data in LDA-C format
  --bootstrap=NUM_BOOTSTRAP  Number of bootstraps for bumping [default: 10]
  --minibatch=MINIBATCH      Number of docs in minibatch [default: 500]
  --alpha=ALPHA              Hyperparameter on topic proportions [default: 1/K]
  --eta=ETA                  Hyperparameter on topics [default: 100/V]
  --tau0=TAU0                Learning rate delay [default: 10.0]
  --kappa=KAPPA              Learning rate forgetting rate [default: 0.75]
  --fixed_step_size=STEPSIZE Fixed stepsize instead RobMonro [default: 0.0]
  --max_itr=MAX_ITR          Max number of iterations for LDA [default: 100]
  --compute_elbo             Boolean flag for computing ELBO
  -h --help                  Show this screen
  --version                  Show version

```


Vocabulary Data Format
----------------------

A text file with each word (`[term_1]` through `[term_N]`) on a separate line.


Corpus Data Format
------------------

A text file where each line is of the form (the LDA-C format):

`[M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]`

where `[M]` is the number of unique terms in the document, and the
`[count]` associated with each term is how many times that term appeared
in the document.


Visualizing the Output
----------------------

A python script visualizes the topics.
(Modified from https://github.com/Blei-Lab/onlineldavb)

```
./printtopics.py data/arxiv-vocab.dat
                 results/Thu_Nov_27_10-45-09_2014/lambda_coord_ascent.dat

./printtopics.py data/arxiv-vocab.dat
                 results/Thu_Nov_27_10-45-09_2014/lambda_svi.dat

./printtopics.py data/arxiv-vocab.dat
                 results/Thu_Nov_27_10-45-09_2014/lambda_bumping.dat
```

























