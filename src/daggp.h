#ifndef DAGGP 
#define DAGGP

#include <RcppArmadillo.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <string>

//#include "nnsearch.h"
#include "covariance.h"

using namespace std;

class DagGP {
public:
  int nr;
  arma::mat coords;
  arma::vec theta;
  
  double precision_logdeterminant;
  double logdens(const arma::vec& x);
  void update_theta(const arma::vec& newtheta, bool update_H=true);
  arma::sp_mat H, Ci;
  void initialize_H();
  bool use_Ci;
  
  arma::field<arma::uvec> dag;
  
  // compute and store markov blanket
  arma::field<arma::uvec> mblanket;
  
  // storing just the nonzero elements of rows of H
  arma::field<arma::uvec> ax;
  arma::field<arma::vec> hrows; 
  arma::vec sqrtR;
  arma::field<arma::vec> h;
  void compute_comps(bool update_H=false);
  arma::mat H_times_A(const arma::mat& A, bool use_spmat=true);
  
  // info about covariance model:
  int matern; // 0: pexp; 1: matern; 2: wave
  double * bessel_ws;
  
  //double ldens;
  DagGP(){};
  
  DagGP(
    const arma::mat& coords_in, 
    const arma::vec& theta_in,
    const arma::field<arma::uvec>& custom_dag,
    int dag_opts=0,
    int covariance_matern=1,
    bool use_Ci_in=false,
    int num_threads_in=1);
  
  // utils
  arma::uvec oneuv;
  int n_threads;
  
  // find common parents, prune dag and cache objs
  int max_prune;
  arma::umat cache_map;
  arma::field<arma::uvec> dag_cache;
  arma::uvec child_cache;
  void prune_dag_cache();
  void build_grid_exemplars();
  
  arma::mat Corr_export(const arma::mat& cx, const arma::uvec& ix, const arma::uvec& jx, int matern, bool same);
};


static inline std::vector<int> set_unique_sorted(std::vector<int> v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

static inline std::string key_of_vec(const std::vector<int>& v) {
  if (v.empty()) return std::string();
  std::string s; s.reserve(v.size() * 3);
  for (size_t i = 0; i < v.size(); ++i) {
    s += std::to_string(v[i]);
    if (i + 1 < v.size()) s.push_back(',');
  }
  return s;
}

static inline std::vector<int> intersect_sorted(const std::vector<int>& a,
                                                const std::vector<int>& b) {
  std::vector<int> out;
  out.reserve(std::min(a.size(), b.size()));
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(out));
  return out;
}

// postings: parent id -> sorted vector (1-based) of nodes having that parent
using Postings = std::unordered_map<int, std::vector<int>>;

static inline int support_size_of_subset(const std::vector<int>& subset,
                                         const Postings& postings,
                                         int min_support) {
  std::vector<const std::vector<int>*> lists; lists.reserve(subset.size());
  for (int p : subset) {
    auto it = postings.find(p);
    if (it == postings.end()) return 0;
    lists.push_back(&it->second);
  }
  std::sort(lists.begin(), lists.end(),
            [](const std::vector<int>* A, const std::vector<int>* B) {
              return A->size() < B->size();
            });
  if ((int)lists.front()->size() < min_support) return 0;
  
  std::vector<int> acc = *lists[0];
  for (size_t k = 1; k < lists.size(); ++k) {
    std::vector<int> tmp;
    tmp.reserve(std::min(acc.size(), lists[k]->size()));
    std::set_intersection(acc.begin(), acc.end(),
                          lists[k]->begin(), lists[k]->end(),
                          std::back_inserter(tmp));
    acc.swap(tmp);
    if ((int)acc.size() < min_support) return 0;
  }
  return (int)acc.size();
}

static inline std::vector<int> nodes_having_subset(const std::vector<int>& subset,
                                                   const Postings& postings) {
  std::vector<const std::vector<int>*> lists; lists.reserve(subset.size());
  for (int p : subset) lists.push_back(&postings.at(p));
  std::sort(lists.begin(), lists.end(),
            [](const std::vector<int>* A, const std::vector<int>* B) {
              return A->size() < B->size();
            });
  std::vector<int> acc = *lists[0];
  for (size_t k = 1; k < lists.size(); ++k) {
    std::vector<int> tmp;
    tmp.reserve(std::min(acc.size(), lists[k]->size()));
    std::set_intersection(acc.begin(), acc.end(),
                          lists[k]->begin(), lists[k]->end(),
                          std::back_inserter(tmp));
    acc.swap(tmp);
    if (acc.empty()) break;
  }
  return acc;
}

static inline std::vector<double> sorted_unique_col(const arma::vec& x) {
  arma::vec u = arma::unique(x);
  std::vector<double> v(u.begin(), u.end());
  std::sort(v.begin(), v.end());
  return v;
}

static inline arma::uvec indexify_col_exact(const arma::vec& x,
                                            const std::vector<double>& uniq_sorted) {
  arma::uvec out(x.n_elem);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double val = x[i];
    auto it = std::lower_bound(uniq_sorted.begin(), uniq_sorted.end(), val);
    if (it == uniq_sorted.end() || *it != val) {
      Rcpp::stop("Grid indexing failed: non-gridded or jittered coordinate.");
    }
    out[i] = static_cast<arma::uword>(std::distance(uniq_sorted.begin(), it));
  }
  return out;
}

static inline std::string key_from_offsets(const arma::Mat<arma::uword>& idx,
                                           arma::uword node,
                                           const arma::uvec& parents) {
  if (parents.n_elem == 0) return std::string("[]");
  std::string s; s.reserve(parents.n_elem * (idx.n_cols * 4 + 1));
  for (arma::uword t = 0; t < parents.n_elem; ++t) {
    arma::uword p = parents[t];
    for (arma::uword k = 0; k < idx.n_cols; ++k) {
      long long off = (long long)idx(p, k) - (long long)idx(node, k);
      s += std::to_string(off);
      if (k + 1 < idx.n_cols) s.push_back(',');
    }
    if (t + 1 < parents.n_elem) s.push_back('|');
  }
  return s;
}

#endif