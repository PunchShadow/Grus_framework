
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
namespace cc {
__device__ char char_atomicCAS(char *addr, char cmp, char val) {
  unsigned *al_addr = reinterpret_cast<unsigned *>(((unsigned long long)addr) &
                                                   (0xFFFFFFFFFFFFFFFCULL));
  unsigned al_offset = ((unsigned)(((unsigned long long)addr) & 3)) * 8;
  unsigned mask = 0xFFU;
  mask <<= al_offset;
  mask = ~mask;
  unsigned sval = val;
  sval <<= al_offset;
  unsigned old = *al_addr, assumed, setval;
  do {
    assumed = old;
    setval = assumed & mask;
    setval |= sval;
    old = atomicCAS(al_addr, assumed, setval);
  } while (assumed != old);
  return (char)((assumed >> al_offset) & 0xFFU);
}
__global__ void CCInit(vtx_t *label, vtx_t nnodes, vtx_t source) {
  size_t tid = TID_1D;
  if (tid < nnodes) {
    label[tid] = tid;
  }
}
// template<typename graph_t>
class job_t {

public:
  vtx_t src;
  vtx_t *label;
  vtx_t itr = 0;
  vtx_t numNode;
  weight_t *adjwgt = nullptr;
  void operator()(vtx_t _numNode, vtx_t _src) {
    numNode = _numNode;
    src = _src;
    init();
  }
  void init() {
    H_ERR(cudaMallocManaged(&label, numNode * sizeof(vtx_t)));
    CCInit<<<numNode / BLOCK_SIZE + 1, BLOCK_SIZE>>>(label, numNode, src);
  }
  void prepare() {}
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
    if (job.label[dst] > job.label[src]) {
      atomicMin(&job.label[dst], job.label[src]);
      // if (!char_atomicCAS(&flag[dstId], 0, 1)) //need F.flag
      return true;
    }
    return false;
  }
};

// Bidirectional CC relaxation used by CC_single_gpu. The generic push_kernel
// + updater pattern can only flag `dst`, which loses half of the union
// opportunities on directed graphs (edges that only exist in the "larger-ID
// → smaller-ID" direction). This kernel does both sides explicitly:
//   * if label[src] < label[dst]: propagate into dst, flag dst for next iter
//   * if label[dst] < label[src]: propagate into src, flag src for next iter
// Both cases use atomicMin so concurrent threads in the warp (different
// edges of the same src) can safely reduce either endpoint.
//
// NB: on a directed graph this pass alone is still insufficient. A vertex
// with no in-edges (e.g. sk-2005 #784982) is never chosen as `dst` when its
// out-neighbours' labels drop later, so its own label stays frozen at the
// initial value. The driver in CC_single_gpu combines this pass with the
// CSC variant below, which iterates in-edges of frontier vertices and
// covers the reverse direction.
template <typename graph_t>
__global__ void cc_bidir_push_kernel(graph_t G, worklist::Worklist wl_c,
                                     char *flag2, vtx_t *label) {
  size_t tid = TID_1D;
  uint laneid = threadIdx.x % 32;
  vtx_t wpid = static_cast<vtx_t>(tid / 32);
  if (wpid < *wl_c.count) {
    vtx_t src = wl_c.data[wpid];
    for (edge_t edge_id = G.xadj[src] + laneid; edge_id < G.xadj[src + 1];
         edge_id += 32) {
      // Re-read label[src] each iteration: another thread in this warp
      // (or elsewhere) may have atomic-min'd it down already, and we want
      // the current value for the comparison so we don't propagate a stale
      // (larger) label to dst.
      vtx_t ls = label[src];
      vtx_t dst = G.adjncy[edge_id];
      vtx_t ld = label[dst];
      if (ls < ld) {
        atomicMin(&label[dst], ls);
        flag2[dst] = 1;
      } else if (ld < ls) {
        atomicMin(&label[src], ld);
        flag2[src] = 1;
      }
    }
  }
}

// Companion CSC pass: each frontier vertex v iterates its *in-edges*, so for
// every directed edge (u, v) we also get a chance to reconcile label[u] and
// label[v] when u is unreachable from the frontier via out-edges alone.
//   * if label[u] < label[v]: propagate into v, flag v (v is already in the
//     frontier but re-flagging is harmless).
//   * if label[v] < label[u]: propagate into u, flag u so u re-iterates its
//     out-edges next round.
template <typename graph_t>
__global__ void cc_bidir_pull_kernel(graph_t G, worklist::Worklist wl_c,
                                     char *flag2, vtx_t *label) {
  size_t tid = TID_1D;
  uint laneid = threadIdx.x % 32;
  vtx_t wpid = static_cast<vtx_t>(tid / 32);
  if (wpid < *wl_c.count) {
    vtx_t dst = wl_c.data[wpid];
    for (edge_t edge_id = G.xadj[dst] + laneid; edge_id < G.xadj[dst + 1];
         edge_id += 32) {
      vtx_t src = G.adjncy[edge_id];
      vtx_t ls = label[src];
      vtx_t ld = label[dst];
      if (ls < ld) {
        atomicMin(&label[dst], ls);
        flag2[dst] = 1;
      } else if (ld < ls) {
        atomicMin(&label[src], ld);
        flag2[src] = 1;
      }
    }
  }
}
struct generator {
  __forceinline__ __device__ void operator()(bool updated,
                                             worklist::Worklist wl, vtx_t dst) {
    if (updated)
      wl.append(dst);
  }
  __forceinline__ __device__ void operator()(bool updated, char *flag,
                                             vtx_t dst) {
    if (updated)
      if (!char_atomicCAS(&flag[dst], 0, 1))
        flag[dst] = true;
  }
  __forceinline__ __device__ void operator()(bool updated, char *flag,
                                             vtx_t dst, char *finished) {
    if (updated) {
      flag[dst] = 1;
      *finished = false;
    }
  }
};
struct pull_selector {
  __forceinline__ __device__ bool operator()(vtx_t id, job_t job) {
    if (job.label[id] == id) {
      return true;
    }
    return false;
  }
};
} // namespace cc
bool CC_multi_gpu() {
  graph_t<CSR> G_csr;
  graph_loader loader;
  loader.Load(G_csr, false);
  graph_t<CSC> G;
  G.CSR2CSC(G_csr);
  G.make_chunks(4);
  // for (size_t i = 0; i < 4; i++) {
  //   cout << "G " << i << G.chunks[i] << endl;
  // }
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  LOG("distributing\n");
  G.distribute_chunks(&stream);
}
bool CC_pull_single_gpu() {
  cudaSetDevice(FLAGS_device);
  H_ERR(cudaDeviceReset());
  graph_t<CSR> G_csr;
  graph_loader loader;
  loader.Load(G_csr, false);
  graph_t<CSC> G;
  G.CSR2CSC(G_csr);
  LOG("CC pull single\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cc::job_t job;
  job(G.numNode, FLAGS_src);
  frontier::Frontier<BITMAP> F; // BDF  BDF_AUTO BITMAP
  F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, true);
  G.Set_Mem_Policy(&stream); // stream
  cudaDeviceSynchronize();
  Timer t;
  t.Start();
  kernel_pull<cc::updater, cc::generator, cc::pull_selector, cc::job_t> K;
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
bool CC_single_gpu() {
  if (FLAGS_pull) {
    return CC_pull_single_gpu();
  }
  cudaSetDevice(FLAGS_device);
  H_ERR(cudaDeviceReset());
  // Load CSR (out-edges) and build CSC (in-edges) up-front. The CC kernel
  // makes two passes per iteration — push on CSR and pull on CSC — which
  // together exercise every logical undirected edge and fixes the
  // over-fragmentation seen on directed inputs (sk-2005, uk-2007-05).
  graph_t<CSR> G;
  graph_loader loader;
  loader.Load(G, false);
  graph_t<CSC> Gc;
  Gc.CSR2CSC(G);
  LOG("CC single\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cc::job_t job;
  job(G.numNode, FLAGS_src);
  frontier::Frontier<BDF> F; // BDF  BDF_AUTO BITMAP
  F.Init(G.numNode, FLAGS_src, FLAGS_device, 1.0, true);
  G.Set_Mem_Policy(&stream);
  Gc.Set_Mem_Policy(&stream);
  cudaDeviceSynchronize();
  Timer t;
  t.Start();
  while (!F.finish()) {
    uint grid = F.get_work_size_h() / (BLOCK_SIZE >> 5) + 1;
    // Pass 1: CSR out-edges. Each frontier vertex pushes its label to its
    // out-neighbours (and pulls from them when they're smaller).
    cc::cc_bidir_push_kernel<graph_t<CSR>>
        <<<grid, BLOCK_SIZE>>>(G, *F.wl_c, F.flag2, job.label);
    // Pass 2: CSC in-edges. Each frontier vertex reconciles with its
    // in-neighbours — specifically this is what propagates labels to
    // zero-in-degree vertices once their out-neighbours drop.
    cc::cc_bidir_pull_kernel<graph_t<CSC>>
        <<<grid, BLOCK_SIZE>>>(Gc, *F.wl_c, F.flag2, job.label);
    cudaDeviceSynchronize();
    F.Next();
    job.itr++;
  }
  cout << "itr " << job.itr << " in " << t.Finish() << endl;
  job.clean();
  return 0;
}
