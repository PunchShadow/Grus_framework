
#include "common.cuh"
#include "frontier.cuh"
#include "graph.cuh"
#include "graph_loader.cuh"
#include "kernel.cuh"
#include "worklist.cuh"
#include <gflags/gflags.h>
using namespace mgg;

DECLARE_int32(device);
DECLARE_string(input);
DECLARE_string(output);
DECLARE_int32(src);
DECLARE_bool(pull);
namespace sssp {

__global__ void SSSPInit(weight_t *label, vtx_t nnodes, vtx_t source) {
  size_t tid = TID_1D;
  if (tid < nnodes) {
    label[tid] = tid == source ? 0 : INFINIT;
  }
}
class job_t {
public:
  vtx_t src;
  weight_t *label;
  vtx_t itr = 0;
  vtx_t numNode;
  weight_t *adjwgt = nullptr;
  void operator()(vtx_t _numNode, vtx_t _src, weight_t *_adjwgt) {
    numNode = _numNode;
    src = _src;
    adjwgt = _adjwgt;
    init();
  }
  void init() {
    H_ERR(cudaMallocManaged(&label, numNode * sizeof(weight_t)));
    SSSPInit<<<numNode / BLOCK_SIZE + 1, BLOCK_SIZE>>>(label, numNode, src);
  }
  void prepare(){}
    void clean() {
  // __host__ __device__ ~job_t() {
#if !defined(__CUDA_ARCH__)
    if (!gflags::GetCommandLineFlagInfoOrDie("output").is_default)
      print::SaveResults(FLAGS_output, label, numNode);
#endif
  }
};

struct updater {
  __forceinline__ __device__ bool operator()(vtx_t src, vtx_t dst,
                                             edge_t edge_id, job_t job) {
    // Relax edge (src → dst). Before the fix this was a plain check-then-write,
    // which races when multiple srcs in the same iteration push to the same
    // dst with different src.label + weight values — the last writer could
    // overwrite the smaller value, leaving label[dst] at a non-minimal cost
    // that subsequent iterations may not recover from (a source that already
    // pushed and left the frontier will not re-push).
    // atomicMin returns the old value, so we know whether *this* thread
    // actually shortened the distance.
    weight_t new_cost = job.label[src] + job.adjwgt[edge_id];
    // Skip relaxations from unreachable sources — src.label = INFINIT would
    // otherwise wrap around to a small value and poison label[dst].
    if (job.label[src] >= INFINIT) return false;
    weight_t old = atomicMin(&job.label[dst], new_cost);
    return old > new_cost;
  }
};
struct generator {
  __forceinline__ __device__ void operator()(bool updated,
                                             worklist::Worklist wl, vtx_t dst) {
    if (updated)
      wl.append(dst);
  }
  __forceinline__ __device__ void operator()(bool updated, char *flag,
                                             vtx_t dst) {
    if (updated)
      flag[dst] = true;
  }
  __forceinline__ __device__ void operator()(bool updated, char *flag,
                                             vtx_t dst, char *finished) {
    if (updated) {
      flag[dst] = true;
      *finished = false;
    }
  }
};
struct pull_selector {
  __forceinline__ __device__ bool operator()(vtx_t id, job_t job) {
    // if (job.label[id] == INFINIT) {
    return true;
    // }
    // return false;
  }
};

} // namespace sssp
bool SSSP_multi_gpu() {}
bool SSSP_pull_single_gpu() {
  cudaSetDevice(FLAGS_device);
  H_ERR(cudaDeviceReset());
  graph_t<CSR> G_csr;
  graph_loader loader;
  loader.Load(G_csr, false);
  graph_t<CSC> G;
  G.CSR2CSC(G_csr);
  LOG("SSSP pull single\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  sssp::job_t job;
  job(G.numNode, FLAGS_src, G.adjwgt);
  frontier::Frontier<BITMAP> F; // BDF  BDF_AUTO BITMAP
  F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, false);
  G.Set_Mem_Policy(&stream); // stream
  cudaDeviceSynchronize();
  Timer t;
  t.Start();
  kernel_pull<sssp::updater, sssp::generator, sssp::pull_selector, sssp::job_t>
      K;
  while (!F.finish()) {
    // cout << "itr " << job.itr << " wl_sz " << F.wl_sz << endl;
    K(G, F, job);
    cudaDeviceSynchronize();
    // H_ERR(cudaStreamSynchronize(stream));
    F.Next();
    job.itr++;
  }
  cout << "itr " << job.itr << " in " << t.Finish() << endl;
  return 0;
}
bool SSSP_single_gpu() {
  if (FLAGS_pull) {
    return SSSP_pull_single_gpu();
  }
  cudaSetDevice(FLAGS_device);
  H_ERR(cudaDeviceReset());
  graph_t<CSR> G(true);
  graph_loader loader;
  loader.Load(G, true);
  // LOG("make g1 chunks\n");
  // G.make_chunks(4);
  // for (size_t i = 0; i < 4; i++) {
  //   cout << "G " << i << G.chunks[i] << endl;
  // }
  // graph_t<CSC> G2;
  // G2.CSR2CSC(G);
  // G2.make_chunks(4);
  // for (size_t i = 0; i < 4; i++) {
  //   cout << "G2 " << i << G2.chunks[i] << endl;
  // }

  LOG("SSSP single\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // G.Init(false);
  sssp::job_t job;
  job(G.numNode, FLAGS_src, G.adjwgt);
  frontier::Frontier<BDF> F; // BDF  BDF_AUTO BITMAP
  F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, false);
  G.Set_Mem_Policy(&stream); // stream
  cudaDeviceSynchronize();
  Timer t;
  t.Start();
  kernel<graph_t<CSR>, frontier::Frontier<BDF>, sssp::updater,
         sssp::generator, sssp::job_t>
      K;
  while (!F.finish()) {
    // cout << "itr " << job.itr << " wl_sz " << F.wl_sz << endl;
    K(G, F, job);
    cudaDeviceSynchronize();
    // H_ERR(cudaStreamSynchronize(stream));
    F.Next();
    job.itr++;
  }
  cout << "itr " << job.itr << " in " << t.Finish() << endl;
  job.clean();
  return 0;
}
