/*
 *  Copyright (c) 2014 Alp Kucukelbir, alp@cs.columbia.edu
 *
 *  - Adapted from code by Cheng Zhang, Xavi Gratal, Chong Wang,
 *    Matthew D. Hoffman, and David M. Blei
 *  - Licensed under terms of GPLv2 license (see LICENSE)
 *
 */

#include <iostream>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <Eigen/Dense>

#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "docopt.h"
#include "corpus.hpp"
#include "util.hpp"

#include "lda.hpp"
#include "lda_coord_ascent.hpp"
#include "lda_svi.hpp"
#include "lda_bumping.hpp"

static const char USAGE[] =
R"(lda-bump-cpp LDA with bumping variational inference.

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
)";

int main(int argc, const char** argv)
{
  std::map< std::string, docopt::value > args =
    docopt::docopt(USAGE,
                  { argv + 1, argv + argc },
                  true,                   // show help if requested
                  util::VERSION_STRING);  // version string

  // Get time (std::put_time had problems on linux...)
  time_t rawtime;
  time (&rawtime);
  std::string time_string = ctime(&rawtime);
  boost::replace_all(time_string, ":",  "-");
  boost::replace_all(time_string, " ",  "_");
  boost::replace_all(time_string, "\n", "");

  // Create results subdirectory
  std::string results_dir = "results/" + time_string;
  std::system(("mkdir " + results_dir).c_str());

  // Set up TeeStream
  std::ofstream fileLOG( results_dir + "/LOG.txt" );
  Tee tee( std::cout, fileLOG );
  TeeStream both( tee );

  both << "== This is lda-bump-cpp/driver.cpp ==" << std::endl << std::endl;
  both << "The current local time is: " << time_string
       << std::endl << std::endl;

  /**
   * DOCOPT
   */
  both << "[ Arguments passed to driver.cpp ]" << std::endl;
  for(auto const& arg : args) {
    both << "  " << arg.first << ": " << arg.second << std::endl;
  }
  both << std::endl;

  int num_topics = 0;
  if (args["--topics"].isString())
  {
    num_topics = stoi(args["--topics"].asString());
  }
  if (num_topics<=0) return -1;

  int num_bootstrap = 0;
  if (args["--bootstrap"].isString())
  {
    num_bootstrap = stoi(args["--bootstrap"].asString());
  }
  if (num_bootstrap<=0) return -1;

  int minibatch_size = 0;
  if (args["--minibatch"].isString())
  {
    minibatch_size = stoi(args["--minibatch"].asString());
  }
  if (minibatch_size<=0) return -1;

  double alpha = 0.0;
  if (args["--alpha"].isString())
  {
    alpha = stod(args["--alpha"].asString());
  }
  if (alpha < 0.0) return -1;

  double eta = 0.0;
  if (args["--eta"].isString())
  {
    eta = stod(args["--eta"].asString());
  }
  if (eta < 0.0) return -1;

  double tau0 = 0.0;
  if (args["--tau0"].isString())
  {
    tau0 = stod(args["--tau0"].asString());
  }
  if (tau0<=0.0) return -1;

  double kappa = 0.0;
  if (args["--kappa"].isString())
  {
    kappa = stod(args["--kappa"].asString());
  }
  if (kappa<=0.0) return -1;

  int max_itr = 0;
  if (args["--max_itr"].isString())
  {
    max_itr = stoi(args["--max_itr"].asString());
  }
  if (max_itr<=0) return -1;

  double fixed_step_size = 0.0;
  if (args["--fixed_step_size"].isString())
  {
    fixed_step_size = stod(args["--fixed_step_size"].asString());
  }
  if (fixed_step_size < 0.0) return -1;

  bool compute_elbo = args["--compute_elbo"].asBool();

  /**
   * INITIALIZE
   */
  // Read in vocabulary
  util::vocabulary my_vocabulary;
  if (args["--vocabulary"].isString())
  {
    if (boost::filesystem::exists(args["--vocabulary"].asString()))
    {
      my_vocabulary.read_data(args["--vocabulary"].asString());
    }
    else
    {
      std::cout << "Vocabulary file : " << args["--vocabulary"].asString()
          << " does not exist! EXITING." << std::endl;
      return -1;
    }
  }

  // Create training corpus, read in training data
  util::corpus train_corpus;
  if (args["--datatr"].isString())
  {
    if (boost::filesystem::exists(args["--datatr"].asString()))
    {
      train_corpus.read_data(args["--datatr"].asString());
      train_corpus.init();
    }
    else
    {
      std::cout << "Training data : " << args["--datatr"].asString()
                << " does not exist! EXITING." << std::endl;
      return -1;
    }
  }

  // Create testing corpus, read in testing data
  util::corpus test_corpus;
  if (args["--datatest"].isString())
  {
    if (boost::filesystem::exists(args["--datatest"].asString()))
    {
      test_corpus.read_data(args["--datatest"].asString());
      test_corpus.init();
    }
    else
    {
      std::cout << "Testing data : " << args["--datatest"].asString()
                << " does not exist! EXITING." << std::endl;
      return -1;
    }
  }

  // Compute alpha and eta parameters (if necessary)
  both << "[ Resolving alpha, eta parameters ]" << std::endl;
  if (alpha == 1.0)
    alpha = 1.0/num_topics;
  if (eta == 100.0)
    eta   = 100.0/my_vocabulary.length_;
  both << "  alpha: " << alpha << std::endl;
  both << "  eta: " << eta << std::endl;
  both << std::endl;

  // Initialize lambda the same way for each algorithm
  std::mt19937_64 gen = util::mersenne_twister();
  std::gamma_distribution<double> gamma_dist_(1e2, 1e-2);  // Gamma distribution
  Eigen::MatrixXd lambda_init = MatrixXd::Zero(num_topics, my_vocabulary.length_);
  for (int j = 0, nRows = lambda_init.rows(),
                  nCols = lambda_init.cols(); j < nCols; ++j)
  {
    for (int i = 0; i < nRows; ++i)
    {
      lambda_init(i, j) = gamma_dist_(gen);
    }
  }

  both << "[ Begin LDA ]" << std::endl;

  // coordinate ascent
  lda::lda_coord_ascent lda_ca(
                              train_corpus, test_corpus,
                              max_itr, 1.0,
                              results_dir,
                              num_topics, my_vocabulary, alpha, eta, tau0, kappa
                              );
  int result_ca = lda_ca.run_lda_coord_ascent(compute_elbo, lambda_init);

  // stochastic variational inference
  lda::lda_svi lda_svi(
                      train_corpus, test_corpus,
                      minibatch_size,
                      max_itr, fixed_step_size,
                      results_dir,
                      num_topics, my_vocabulary, alpha, eta, tau0, kappa
                      );
  int result_svi = lda_svi.run_lda_svi(compute_elbo, lambda_init);

  // bumping variational inference
  lda::lda_bumping lda_bump(
                           train_corpus, test_corpus,
                           num_bootstrap,
                           max_itr, fixed_step_size,
                           results_dir,
                           num_topics, my_vocabulary, alpha, eta, tau0, kappa
                           );
  int result_bumping = lda_bump.run_lda_bumping(compute_elbo, lambda_init);

  both << std::endl;
  return 0;
}

















