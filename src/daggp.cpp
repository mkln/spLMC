#include "daggp.h"
#include <thread>

DagGP::DagGP(
  const arma::mat& coords_in,
  const arma::vec& theta_in,
  const arma::field<arma::uvec>& custom_dag,
  int dag_opts,
  int covariance_matern,
  bool use_Ci_in,
  int num_threads_in){
  
  coords = coords_in;
  theta = theta_in;
  nr = coords.n_rows;
  
  use_Ci = use_Ci_in; 
  
  dag = custom_dag;
  
  oneuv = arma::ones<arma::uvec>(1);
  
  matern = covariance_matern; // 0 pexp or 1 matern or 2 wave
  //Rcpp::Rcout << "DagGP covariance choice " << matern << endl;
  //thread safe stuff
  n_threads = num_threads_in;
  
  int bessel_ws_inc = MAT_NU_MAX;//see bessel_k.c for working space needs
  bessel_ws = (double *) R_alloc(n_threads*bessel_ws_inc, sizeof(double));
  
  //
  mblanket = arma::field<arma::uvec>(nr);
  hrows = arma::field<arma::vec>(nr);
  ax = arma::field<arma::uvec>(nr);
  
  if(dag_opts != 0){
    if(dag_opts > 0){
      // pruning the dag for efficiency
      // = trim edges to see if there are ties in the parent sets
      max_prune = dag_opts;
      
      cache_map = arma::zeros<arma::umat>(nr, 2);
      prune_dag_cache();
    } else {
      // gridded coords so full cache
      max_prune = -1;
      
      cache_map = arma::zeros<arma::umat>(nr, 2);
      build_grid_exemplars();
    }
  } else {
    max_prune = 0;
    dag_cache = dag;                           
    cache_map.set_size(nr, 2);
    for (unsigned int i = 0; i < nr; i++) {
      cache_map(i, 0) = i;                     
      cache_map(i, 1) = i;                     
    }
  }
  
  compute_comps();
  initialize_H();
  
  
}


void DagGP::prune_dag_cache() {
  // this function simplifies the dag by removing at most max_prune edges from each node
  // to possibly find that multiple nodes have the same parent set
  // when parent sets are the same, we can reduce the number of matrix inversions
  
  const int n = dag.n_elem;
  
  int m = 0;
  for(int i=0; i<dag.n_elem; i++){
    m = dag(i).n_elem > m ? dag(i).n_elem : m;
  }
  
  int min_support = 1;
  // only prune for dags with more than 10 neighbors
  int min_size = m > 10? std::max(m - max_prune, 10) : m; 
  
  //Rcpp::Rcout << "min_size: " << min_size << endl;
  
#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
#endif
  
  // 1) Canonicalize parent sets P[i] (sorted & unique; as std::vector<int>)
  std::vector< std::vector<int> > P(n);
  long long edges_before = 0;
  for (int i = 0; i < n; ++i) {
    const arma::uvec& vi = dag(i);
    std::vector<int> tmp; tmp.reserve(vi.n_elem);
    for (size_t k = 0; k < vi.n_elem; ++k) {
      int val = (int)vi[k];
      //if (val > 0) 
      tmp.push_back(val);
    }
    P[i] = set_unique_sorted(tmp);
    edges_before += (long long)P[i].size();
  }
  
  // 2) Build postings (parent -> nodes), and prune rare parents (< min_support)
  Postings postings;
  postings.reserve(n * 2);
  for (int i = 0; i < n; ++i) {
    for (int p : P[i]) postings[p].push_back(i); // store nodes as 0-based
  }
  for (auto &kv : postings) {
    auto &vec = kv.second;
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
  }
  { // drop rare parents
    std::vector<int> to_drop; to_drop.reserve(postings.size());
    for (const auto &kv : postings)
      if ((int)kv.second.size() < min_support) to_drop.push_back(kv.first);
      for (int p : to_drop) postings.erase(p);
  }
  // Drop rare parents from each node's set
  for (int i = 0; i < n; ++i) {
    std::vector<int> kept; kept.reserve(P[i].size());
    for (int p : P[i]) if (postings.find(p) != postings.end()) kept.push_back(p);
    P[i].swap(kept);
  }
  
  // If nothing left after pruning, return identity mapping with originals
  bool any_left = false;
  for (int i = 0; i < n; ++i) { if (!P[i].empty()) { any_left = true; break; } }
  if (!any_left) {
    // unique sets = originals; exemplar id = node id
    dag_cache = arma::field<arma::uvec>(n);
    for (int i = 0; i < n; ++i) dag_cache(i) = arma::zeros<arma::uvec>(0); // empty after dropping all
    
    for (int i = 0; i < n; ++i) { cache_map(i,0) = i; cache_map(i,1) = i; }
  }
  
  // 3) Build candidate node pairs (i<j) by shared parents (parallelized)
  std::vector< std::pair<int,int> > pairs; 
  
  {
    // gather parent ids to iterate in parallel
    std::vector<int> parent_ids; parent_ids.reserve(postings.size());
    for (const auto &kv : postings) parent_ids.push_back(kv.first);
    
#ifdef _OPENMP
    int T = (n_threads > 0 ? n_threads : omp_get_max_threads());
#else
    int T = 1;
#endif
    std::vector< std::vector< std::pair<int,int> > > buffers(T);
    
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int idx = 0; idx < (int)parent_ids.size(); ++idx) {
#ifdef _OPENMP
      int tid = omp_get_thread_num();
#else
      int tid = 0;
#endif
      const int p = parent_ids[idx];
      const std::vector<int>& nodes = postings[p];
      const int s = (int)nodes.size();
      if (s < 2) continue;
      auto &buf = buffers[tid];
      for (int a = 0; a < s; ++a) {
        for (int b = a + 1; b < s; ++b) {
          buf.emplace_back(nodes[a], nodes[b]);
        }
      }
    }
    // merge & deduplicate
    size_t total = 0;
    for (auto &b : buffers) total += b.size();
    pairs.reserve(total);
    for (auto &b : buffers) {
      pairs.insert(pairs.end(), b.begin(), b.end());
      std::vector< std::pair<int,int> >().swap(b);
    }
  }
  
  if (pairs.empty()) {
    // No shared-parent pairs -> cores = originals
    dag_cache = arma::field<arma::uvec>(n);
    long long edges_after = 0;
    for (int i = 0; i < n; ++i) {
      dag_cache(i) = arma::conv_to<arma::uvec>::from(P[i]);
      edges_after += (long long)P[i].size();
    }
    for (int i = 0; i < n; ++i) { cache_map(i,0) = i; cache_map(i,1) = i; }
  }
  
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  
  // 4) Candidate subsets = intersections P[i] ∩ P[j]; keep |S| >= min_size (parallel)
#ifdef _OPENMP
  int T_c = (n_threads > 0 ? n_threads : omp_get_max_threads());
#else
  int T_c = 1;
#endif
  std::vector< std::vector< std::vector<int> > > cands_local(T_c);
  
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int k = 0; k < (int)pairs.size(); ++k) {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif
    const int i = pairs[k].first;
    const int j = pairs[k].second;
    const auto S = intersect_sorted(P[i], P[j]);
    if ((int)S.size() >= min_size) cands_local[tid].push_back(S);
  }
  
  // Merge & deduplicate candidates by key
  std::vector< std::vector<int> > Cands;
  {
    std::unordered_set<std::string> seen;
    for (int t = 0; t < T_c; ++t) {
      for (auto &S : cands_local[t]) {
        std::string key = key_of_vec(S);
        if (!key.empty() && seen.insert(key).second) {
          Cands.push_back(std::move(S));
        }
      }
      std::vector< std::vector<int> >().swap(cands_local[t]);
    }
  }
  
  if (Cands.empty()) {
    dag_cache = arma::field<arma::uvec>(n);
    long long edges_after = 0;
    for (int i = 0; i < n; ++i) {
      dag_cache(i) = arma::conv_to<arma::uvec>::from(P[i]);
      edges_after += (long long)P[i].size();
    }
    for (int i = 0; i < n; ++i) { cache_map(i,0) = i; cache_map(i,1) = i; }
  }
  
  // 5) Compute supports for candidates (parallel), keep those >= min_support
  std::vector<int> support(Cands.size(), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int c = 0; c < (int)Cands.size(); ++c) {
    support[c] = support_size_of_subset(Cands[c], postings, min_support);
  }
  
  std::vector< std::vector<int> > C_keep;
  std::vector<int> supp_keep;
  std::vector<std::string> keys_keep;
  C_keep.reserve(Cands.size());
  supp_keep.reserve(Cands.size());
  keys_keep.reserve(Cands.size());
  
  for (size_t c = 0; c < Cands.size(); ++c) {
    if (support[c] >= min_support) {
      C_keep.push_back(std::move(Cands[c]));
      supp_keep.push_back(support[c]);
      keys_keep.push_back(key_of_vec(C_keep.back()));
    }
  }
  
  if (C_keep.empty()) {
    dag_cache = arma::field<arma::uvec>(n);
    long long edges_after = 0;
    for (int i = 0; i < n; ++i) {
      dag_cache(i) = arma::conv_to<arma::uvec>::from(P[i]);
      edges_after += (long long)P[i].size();
    }
    
    for (int i = 0; i < n; ++i) { cache_map(i,0) = i; cache_map(i,1) = i; }

  }
  
  // 6) For each frequent candidate, get the nodes that contain it (parallel)
  std::vector< std::vector<int> > cand_nodes(C_keep.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int c = 0; c < (int)C_keep.size(); ++c) {
    cand_nodes[c] = nodes_having_subset(C_keep[c], postings); 
  }
  
  // 7) Per-node best core (size -> support -> lexicographic key)
  struct BestCore {
    int size = -1;
    int support = -1;
    std::string key;
    int cand_index = -1;
  };
  std::vector<BestCore> best(n);
  
  for (int c = 0; c < (int)C_keep.size(); ++c) {
    const int sz = (int)C_keep[c].size();
    const int sp = supp_keep[c];
    const std::string &ky = keys_keep[c];
    const std::vector<int> &nodes = cand_nodes[c];
    for (int node1b : nodes) {
      const int i = node1b;
      auto &cur = best[i];
      bool better = false;
      if (sz > cur.size) better = true;
      else if (sz == cur.size && sp > cur.support) better = true;
      else if (sz == cur.size && sp == cur.support && (cur.key.empty() || ky < cur.key)) better = true;
      if (better) {
        cur.size = sz; cur.support = sp; cur.key = ky; cur.cand_index = c;
      }
    }
  }
  
  // Chosen cores (fallback to original P[i] if none assigned)
  std::vector< std::vector<int> > cores(n);
  for (int i = 0; i < n; ++i) {
    if (best[i].cand_index >= 0) cores[i] = C_keep[best[i].cand_index];
    else cores[i] = P[i];
  }
  
  // 8) Deduplicate cores to exemplar ids
  std::unordered_map<std::string, int> core_to_id;
  core_to_id.reserve(n * 2);
  std::vector< std::vector<int> > exemplars; exemplars.reserve(n);
  
  for (int i = 0; i < n; ++i) {
    std::string k = key_of_vec(cores[i]);
    auto it = core_to_id.find(k);
    int gid;
    if (it == core_to_id.end()) {
      gid = (int)exemplars.size(); // 0-based group id
      core_to_id.emplace(k, gid);
      exemplars.push_back(cores[i]);
    } else {
      gid = it->second;
    }
    cache_map(i, 0) = i;
    cache_map(i, 1) = gid;
  }
  
  // Convert exemplars to R list and compute edges_after as total size

  dag_cache = arma::field<arma::uvec>(exemplars.size());
  long long edges_after = 0;
  for (size_t k = 0; k < exemplars.size(); ++k) {
    dag_cache(k) = arma::conv_to<arma::uvec>::from(exemplars[k]);
    edges_after += (long long)exemplars[k].size();
  }
}

void DagGP::build_grid_exemplars() {
  // this function computes exemplar sets (parents, i) for gridded data
  // and assuming isotropic covariance
  // in this case the relationships between locations replicates across the domain
  // so we can massively reduce the number of matrix operations
  // this function is best paired with a neighbor search on an ordering that
  // makes it more likely to find many ties.
  
  const arma::uword N = coords.n_rows;
  const arma::uword d = coords.n_cols;
  if (N != dag.n_elem) Rcpp::stop("coords / dag size mismatch");
  
  arma::field<arma::uvec> dagplus = dag;
  if(max_prune < 0){
    // gridded coords, so cache the child too
    for(unsigned int i=0; i<nr; i++){
      dagplus(i) = arma::join_vert(dag(i), oneuv*i);
    }
  }
  
  
  // 1) grid index per dimension (0-based)
  arma::Mat<arma::uword> idx(N, d);
  for (arma::uword k = 0; k < d; ++k) {
    auto levels = sorted_unique_col(coords.col(k));
    idx.col(k) = indexify_col_exact(coords.col(k), levels);
  }
  
  // 2) hash by ordered offset pattern
  std::unordered_map<std::string, int> key2gid;
  key2gid.reserve(N * 2);
  
  cache_map.set_size(N, 2);
  std::vector< arma::uvec > exemplar_sets; exemplar_sets.reserve(N);
  
  for (arma::uword i = 0; i < N; ++i) {
    const arma::uvec& Pi = dagplus(i);           // 0-based parents for node i
    std::string key = key_from_offsets(idx, i, Pi);
    auto it = key2gid.find(key);
    int gid;
    if (it == key2gid.end()) {
      gid = (int)exemplar_sets.size();
      key2gid.emplace(std::move(key), gid);
      exemplar_sets.push_back(Pi);            // store this node's parents as exemplar
    } else {
      gid = it->second;
    }
    cache_map(i, 0) = i;                      // node id (0-based)
    cache_map(i, 1) = gid;                    // exemplar id (0-based)
  }
  
  // 3) materialize dag_cache from exemplars
  dag_cache.set_size(exemplar_sets.size());
  for (size_t g = 0; g < exemplar_sets.size(); ++g) {
    dag_cache(g) = exemplar_sets[g];
  }
  

  // 3) materialize dag_cache from exemplars
  const arma::uword G = exemplar_sets.size();
  dag_cache.set_size(G);
  child_cache = arma::zeros<arma::uvec>(G);  // arma::uvec
  
  for (arma::uword g = 0; g < G; ++g) {
    if(max_prune < 0){
      const arma::uvec& v = exemplar_sets[g];      // expects [parents..., child]
      const arma::uword n = v.n_elem;
      
      if (n == 0) {
        dag_cache(g)   = arma::uvec();             // empty parents
        child_cache(g) = arma::uword(-1);          // sentinel (or choose 0)
        continue;
      }
      
      // parents = first n-1 elements (copy); handles n==1 by making empty uvec
      if (n > 1) dag_cache(g) = v.head(n - 1);
      else       dag_cache(g) = arma::uvec();      // no parents
      // child = last element
      child_cache(g) = v[n - 1];                   // or v.back()
    } else {
      dag_cache(g) = exemplar_sets[g];
    }
  }
}

void DagGP::compute_comps(bool update_H){
  // this function avoids building H since H is always used to multiply a matrix A
  sqrtR = arma::zeros(nr);
  h = arma::field<arma::vec>(nr);
  arma::vec logdetvec(nr);
  
  arma::uvec errors = arma::zeros<arma::uvec>(nr);
  
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed;
  
  tstart = std::chrono::steady_clock::now();
  const int G = static_cast<int>(dag_cache.n_elem);
  arma::field<arma::mat> Pinv_cache(G);
  arma::field<arma::mat> h_cache(G);
  arma::vec CPC_cache(G);
  
  bool cache_error = false;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for (int g = 0; g < G; ++g) {
    try {
      const arma::uvec& pxg = dag_cache(g);          // 0-based indices
      
      if (pxg.n_elem == 0u) { Pinv_cache(g).reset(); continue; }
      
      arma::mat Pg  = Correlationf(coords, pxg, pxg, theta, bessel_ws, matern, /*same=*/true);
      Pinv_cache(g) = arma::inv_sympd(Pg);
      
      if(max_prune < 0){
        const arma::uvec& ixg = oneuv*child_cache(g);
        arma::mat CPt = Correlationf(coords, pxg, ixg, theta, bessel_ws, matern, false);
        h_cache(g) = Pinv_cache(g) * CPt;
        CPC_cache(g) = arma::conv_to<double>::from(CPt.t() * h_cache(g));
      }
    } catch (...) {
      cache_error = true;
    }
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  
  if(cache_error){
    Rcpp::stop("Failed to compute cached elements.\n");
  }
  
  //Rcpp::Rcout << "First loop: " << timed << endl; 
  
  tstart = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for(int i=0; i<nr; i++){
    try {
      arma::uvec ix = oneuv * i;                 
      const int gid = static_cast<int>(cache_map(i,1));   
      
      arma::uvec px;
      if(max_prune < 0){
        // gridded, look up original dag
        // because dag_cache has the exemplar parents
        px = dag(i);
      } else {
        px = dag_cache(gid);   
      }   
      
      ax(i) = arma::join_vert(ix, px);
      
      arma::mat CC  = Correlationf(coords, ix, ix, theta, bessel_ws, matern, true);
      
      arma::mat CPC;
      arma::vec hi;
      
      if (px.n_elem == 0u) {
        hi.set_size(0);
        sqrtR(i) = std::sqrt( arma::conv_to<double>::from( CC ) );
      } else {
        //const arma::mat& Pinv = Pinv_cache(gid);   
        // gridded data
        if(max_prune < 0){
          hi = h_cache(gid);   
          CPC = CPC_cache(gid);
        } else {
          arma::mat CPt = Correlationf(coords, px, ix, theta, bessel_ws, matern, false);
          hi = Pinv_cache(gid) * CPt;             
          CPC = CPt.t() * hi;
        }
        sqrtR(i) = std::sqrt( arma::conv_to<double>::from( CC - CPC ) );
      }
      
      h(i) = hi;
      
      arma::vec mhR = (px.n_elem ? -hi / sqrtR(i) : arma::vec());
      arma::vec rowoneR = arma::ones<arma::vec>(1) / sqrtR(i);
      
      hrows(i) = (px.n_elem ? arma::join_vert(rowoneR, mhR) : rowoneR);
      logdetvec(i) = -std::log(sqrtR(i));
      
      if(update_H){
        H(i,i) = 1.0/sqrtR(i);
        for (arma::uword j = 0; j < px.n_elem; ++j) {
          H(i, px(j)) = -hi(j) / sqrtR(i);        
        }
      }
    } catch (...) {
      errors(i) = 1;
    }
  }
  tend = std::chrono::steady_clock::now();
  
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  
  //Rcpp::Rcout << "Second loop: " << timed << endl; 
  
  if(arma::any(errors)){
    Rcpp::stop("Failure in building sparse DAG GP. Check coordinates/values of theta.");
  }
  
  if(use_Ci){ Ci = H.t() * H; }
  precision_logdeterminant = 2 * arma::accu(logdetvec);
}


double DagGP::logdens(const arma::vec& x){
  double loggausscore = arma::conv_to<double>::from( x.t() * H.t() * H * x );
  return 0.5 * ( precision_logdeterminant - loggausscore );
}

void DagGP::update_theta(const arma::vec& newtheta, bool update_H){
  theta = newtheta;
  compute_comps(update_H);
}

void DagGP::initialize_H(){
  // building sparse H and Ci
  unsigned int H_n_elem = nr;
  for(int i=0; i<nr; i++){
    if(max_prune < 0){
      // gridded, look up original dag
      // because dag_cache has the exemplar parents
      H_n_elem += dag(i).n_elem;
    } else {
      int u = cache_map(i, 1);                 
      H_n_elem += dag_cache(u).n_elem;      
    }
    
       
  }
  arma::umat H_locs(2, H_n_elem);
  arma::vec H_values(H_n_elem);
  unsigned int ix=0;
  for(int i=0; i<nr; i++){
    H_locs(0, ix) = i;
    H_locs(1, ix) = i;
    H_values(ix) = 1.0/sqrtR(i);
    ix ++;
    int u;
    arma::uvec daghere;
    if(max_prune < 0){
      u = i;
      daghere = dag(u);
    } else {
      u = cache_map(i,1);
      daghere = dag_cache(u);
    }
    
    for(unsigned int j=0; j<daghere.n_elem; j++){
      H_locs(0, ix) = i;
      H_locs(1, ix) = daghere(j);
      H_values(ix) =  -h(i)(j)/sqrtR(i);
      ix ++;
    }
  }
  H = arma::sp_mat(H_locs, H_values, nr, nr);
  
  // compute precision just to get the markov blanket
  arma::sp_mat Ci_temp = H.t() * H;
  
  // comp markov blanket
  for(int i=0; i<nr; i++){
    int nonzeros_in_col = 0;
    for (arma::sp_mat::const_iterator it = Ci_temp.begin_col(i); it != Ci_temp.end_col(i); ++it) {
      nonzeros_in_col ++;
    }
    mblanket(i) = arma::zeros<arma::uvec>(nonzeros_in_col-1); // not the diagonal
    int ix = 0;
    for (arma::sp_mat::const_iterator it = Ci_temp.begin_col(i); it != Ci_temp.end_col(i); ++it) {
      if(it.row() != i){ 
        mblanket(i)(ix) = it.row();
        ix ++;
      }
    }
  }
  
  if(use_Ci){
    Ci = Ci_temp;
  }
}

arma::mat DagGP::H_times_A(const arma::mat& A, bool use_spmat){
  // this function avoids building H since H is always used to multiply a matrix A
  if(!use_spmat){
    arma::mat result = arma::zeros(nr, A.n_cols);
    // calculate components for H and Ci
    for(unsigned int j=0; j<A.n_cols; j++){
      arma::vec Aj = A.col(j);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
      for(int i=0; i<nr; i++){
        result(i, j) = arma::accu(hrows(i).t() * Aj(ax(i)));
      }
    }
    return result;
  } else {
    return H * A;
  }
}


arma::mat DagGP::Corr_export(const arma::mat& these_coords, 
                             const arma::uvec& ix, const arma::uvec& jx, 
                             int matern, bool same){
  return Correlationf(these_coords, ix, jx, theta, bessel_ws, matern, same);
}

//[[Rcpp::export]]
Rcpp::List daggp_build(const arma::mat& coords, const arma::field<arma::uvec>& dag,
                       double phi, double sigmasq, double nu, double tausq,
                       int matern=1, int num_threads=1, bool prune_dag=false){
  
  arma::vec theta(4);
  theta(0) = phi;
  theta(1) = sigmasq;
  theta(2) = nu;
  theta(3) = tausq;
  
  Rcpp::Rcout << "Building DAG-GP model\n";
  DagGP adag(coords, theta, dag, matern, false, num_threads, prune_dag);
  
  Rcpp::Rcout << "Calculating Ci\n";
  arma::sp_mat Ci = adag.H.t() * adag.H;
  Rcpp::Rcout << "Done. Returning. \n";
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("dag_cache") = adag.dag_cache,
    Rcpp::Named("cache_map") = adag.cache_map,
    Rcpp::Named("H") = adag.H,
    Rcpp::Named("Cinv") = Ci,
    Rcpp::Named("Cinv_logdet") = adag.precision_logdeterminant
  );
}



/*

using EigenFromArma = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;
 
arma::sp_mat sparse_convert(const Eigen::SparseMatrix<double>& XE){
 unsigned int num_cols = XE.cols();
 unsigned int num_rows = XE.rows();
 
 std::vector<unsigned int> rowind_vect(
 XE.innerIndexPtr(), XE.innerIndexPtr() + XE.nonZeros());
 std::vector<unsigned int> colptr_vect(
 XE.outerIndexPtr(), XE.outerIndexPtr() + XE.outerSize() + 1);
 std::vector<double> values_vect(
 XE.valuePtr(), XE.valuePtr() + XE.nonZeros());
 
 arma::vec values(values_vect.data(), values_vect.size(), false);
 arma::uvec rowind(rowind_vect.data(), rowind_vect.size(), false);
 arma::uvec colptr(colptr_vect.data(), colptr_vect.size(), false);
 
 arma::sp_mat result(rowind, colptr, values, num_rows, num_cols);
 
 return result; 
} */