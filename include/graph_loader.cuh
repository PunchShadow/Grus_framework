#ifndef _GRAPH_LOADER_CUH
#define _GRAPH_LOADER_CUH

#include "common.cuh"
#include "graph.cuh"
#include "intrinsics.cuh"
#include "print.cuh"
#include "timer.cuh"

#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <gflags/gflags.h>
#include <limits>
#include <string>

// using namespace intrinsics;
// using namespace grus;
// using namespace frontier;

DECLARE_string(input);
DECLARE_bool(pf);
DECLARE_bool(ab);
DECLARE_bool(rm);
DECLARE_bool(pl);
DECLARE_bool(opt);
DECLARE_int32(device);
namespace mgg {
class graph_loader {
public:
  string graphFilePath;

  bool hasZeroID;
  uint64_t numNode;
  uint64_t numEdge;
  uint64_t sizeEdgeTy;

  // graph
  edge_t *xadj = nullptr;
  vtx_t *vwgt = nullptr, *adjncy = nullptr;
  // vtx_t *xadj_d, *vwgt_d, *adjncy_d;
  weight_t *adjwgt = nullptr, *adjwgt_d = nullptr;
  uint *inDegree = nullptr;
  uint *outDegree = nullptr;
  bool weighted;
  bool needWeight;

  uint64_t mem_used = 0;

  graph_loader() {}
  ~graph_loader() {}
  void gk_fclose(FILE *fp) { fclose(fp); }
  FILE *gk_fopen(const char *fname, const char *mode, const char *msg) {
    FILE *fp;
    char errmsg[8192];
    fp = fopen(fname, mode);
    if (fp != NULL)
      return fp;
    sprintf(errmsg, "file: %s, mode: %s, [%s]", fname, mode, msg);
    perror(errmsg);
    printf("Failed on gk_fopen()\n");
    return NULL;
  }
  static string GetExt(const string &path) {
    size_t dot = path.find_last_of('.');
    if (dot == string::npos)
      return "";
    return path.substr(dot + 1);
  }
  static bool IsBinaryCsr(const string &ext) {
    return ext == "bcsr" || ext == "bwcsr" || ext == "bcsr64" ||
           ext == "bwcsr64";
  }
  static bool BinaryIsWeighted(const string &ext) {
    return ext == "bwcsr" || ext == "bwcsr64";
  }
  static bool BinaryIs64Bit(const string &ext) {
    return ext == "bcsr64" || ext == "bwcsr64";
  }
  // Streams an array of SrcT values into a 64-bit edge_t destination buffer,
  // widening if the file type is 32-bit and a direct copy otherwise. Chunked
  // to avoid allocating a temporary host buffer as large as the full graph.
  template <typename SrcT>
  static void ReadOffsetsInto(FILE *fpin, edge_t *dst, uint64_t num_Node) {
    constexpr size_t CHUNK = 1 << 20; // 1M entries per chunk
    if (sizeof(SrcT) == sizeof(edge_t)) {
      // Direct read into destination buffer.
      size_t got = fread(dst, sizeof(edge_t), num_Node, fpin);
      if (got != num_Node) {
        fprintf(stderr,
                "Error: partial read of %llu 64-bit offsets (got %zu)\n",
                (unsigned long long)num_Node, got);
        exit(-1);
      }
      return;
    }
    vector<SrcT> buf(CHUNK);
    uint64_t remaining = num_Node;
    uint64_t idx = 0;
    while (remaining > 0) {
      size_t n = remaining < CHUNK ? remaining : CHUNK;
      size_t got = fread(buf.data(), sizeof(SrcT), n, fpin);
      if (got != n) {
        fprintf(stderr, "Error: partial read of %zu offsets (got %zu)\n", n,
                got);
        exit(-1);
      }
      for (size_t i = 0; i < got; i++)
        dst[idx + i] = static_cast<edge_t>(buf[i]);
      idx += got;
      remaining -= got;
    }
  }
  // Unweighted edge destinations. vtx_t is now 64-bit; widen from 32-bit files
  // or read 64-bit files directly.
  template <typename SrcT>
  static void ReadEdgesInto(FILE *fpin, vtx_t *dst, uint64_t num_Edge) {
    constexpr size_t CHUNK = 1 << 20;
    if (sizeof(SrcT) == sizeof(vtx_t)) {
      size_t got = fread(dst, sizeof(vtx_t), num_Edge, fpin);
      if (got != num_Edge) {
        fprintf(stderr,
                "Error: partial read of %llu 64-bit edge dests (got %zu)\n",
                (unsigned long long)num_Edge, got);
        exit(-1);
      }
      return;
    }
    vector<SrcT> buf(CHUNK);
    uint64_t remaining = num_Edge;
    uint64_t idx = 0;
    while (remaining > 0) {
      size_t n = remaining < CHUNK ? remaining : CHUNK;
      size_t got = fread(buf.data(), sizeof(SrcT), n, fpin);
      if (got != n) {
        fprintf(stderr, "Error: partial read of %zu edge dests (got %zu)\n", n,
                got);
        exit(-1);
      }
      for (size_t i = 0; i < got; i++)
        dst[idx + i] = static_cast<vtx_t>(buf[i]);
      idx += got;
      remaining -= got;
    }
  }
  // Weighted edges stored as {SrcT end; SrcT w8;} pairs. Split into separate
  // (adjncy, adjwgt) arrays of vtx_t/weight_t (both 64-bit now).
  template <typename SrcT>
  static void ReadWeightedEdgesInto(FILE *fpin, vtx_t *dst, weight_t *wgt,
                                    uint64_t num_Edge) {
    struct Pair {
      SrcT end;
      SrcT w8;
    };
    constexpr size_t CHUNK = 1 << 19;
    vector<Pair> buf(CHUNK);
    uint64_t remaining = num_Edge;
    uint64_t idx = 0;
    while (remaining > 0) {
      size_t n = remaining < CHUNK ? remaining : CHUNK;
      size_t got = fread(buf.data(), sizeof(Pair), n, fpin);
      if (got != n) {
        fprintf(stderr,
                "Error: partial read of %zu weighted edges (got %zu)\n", n,
                got);
        exit(-1);
      }
      for (size_t i = 0; i < got; i++) {
        dst[idx + i] = static_cast<vtx_t>(buf[i].end);
        wgt[idx + i] = static_cast<weight_t>(buf[i].w8);
      }
      idx += got;
      remaining -= got;
    }
  }
  // Reads Subway-style binary CSR (bcsr/bwcsr/bcsr64/bwcsr64). The file layout
  // is:
  //   header:  num_nodes num_edges    (uint32 for bcsr/bwcsr, uint64 for *64)
  //   offsets: nodePointer[num_nodes] (same width as header)
  //   edges:   dest[num_edges]        for unweighted
  //            {dest, w8}[num_edges]  for weighted
  // Note: the file only stores num_nodes offsets; the sentinel xadj[num_nodes]
  // = num_edges is synthesized here.
  void ReadGraphBCSR(graph_t<CSR> &G, const string &ext) {
    FILE *fpin = gk_fopen(graphFilePath.data(), "rb", "ReadGraphBCSR: Graph");
    if (!fpin)
      exit(-1);
    bool fileHasWeight = BinaryIsWeighted(ext);
    bool is64 = BinaryIs64Bit(ext);
    uint64_t num_Node = 0, num_Edge = 0;
    if (is64) {
      if (fread(&num_Node, sizeof(uint64_t), 1, fpin) != 1 ||
          fread(&num_Edge, sizeof(uint64_t), 1, fpin) != 1) {
        fprintf(stderr, "Failed to read 64-bit header from %s\n",
                graphFilePath.c_str());
        exit(-1);
      }
    } else {
      uint32_t n32 = 0, e32 = 0;
      if (fread(&n32, sizeof(uint32_t), 1, fpin) != 1 ||
          fread(&e32, sizeof(uint32_t), 1, fpin) != 1) {
        fprintf(stderr, "Failed to read 32-bit header from %s\n",
                graphFilePath.c_str());
        exit(-1);
      }
      num_Node = n32;
      num_Edge = e32;
    }
    cout << graphFilePath + " has " << num_Node << " nodes and " << num_Edge
         << " edges\n";

    H_ERR(cudaMallocManaged(&G.xadj, (num_Node + 1) * sizeof(edge_t)));
    H_ERR(cudaMallocManaged(&G.adjncy, num_Edge * sizeof(vtx_t)));
    mem_used += (num_Node + 1) * sizeof(edge_t) + num_Edge * sizeof(vtx_t);
    // adjwgt is always allocated (like the .gr reader) so unweighted users
    // still get a uniform-1 weight array for SSSP/PR.
    H_ERR(cudaMallocManaged(&G.adjwgt, num_Edge * sizeof(weight_t)));
    mem_used += num_Edge * sizeof(weight_t);

    // offsets
    if (is64)
      ReadOffsetsInto<uint64_t>(fpin, G.xadj, num_Node);
    else
      ReadOffsetsInto<uint32_t>(fpin, G.xadj, num_Node);
    G.xadj[num_Node] = num_Edge; // sentinel

    // edges
    weighted = fileHasWeight;
    if (fileHasWeight) {
      if (is64)
        ReadWeightedEdgesInto<uint64_t>(fpin, G.adjncy, G.adjwgt, num_Edge);
      else
        ReadWeightedEdgesInto<uint32_t>(fpin, G.adjncy, G.adjwgt, num_Edge);
    } else {
      if (is64)
        ReadEdgesInto<uint64_t>(fpin, G.adjncy, num_Edge);
      else
        ReadEdgesInto<uint32_t>(fpin, G.adjncy, num_Edge);
      for (uint64_t i = 0; i < num_Edge; i++)
        G.adjwgt[i] = 1;
    }

    G.outDegree = new uint[num_Node];
    for (uint64_t i = 0; i < num_Node; i++)
      G.outDegree[i] = static_cast<uint>(G.xadj[i + 1] - G.xadj[i]);
    uint64_t maxD = static_cast<uint64_t>(std::distance(
        G.outDegree, std::max_element(G.outDegree, G.outDegree + num_Node)));
    printf("vtx %llu has max out degree %u\n", (unsigned long long)maxD,
           G.outDegree[maxD]);
    G.mem_used = mem_used;
    G.numNode = num_Node;
    G.numEdge = num_Edge;
    gk_fclose(fpin);
  }
  void ReadGraphGR(graph_t<CSR> &G) {
    // uint *vsize;
    FILE *fpin;
    fpin = gk_fopen(graphFilePath.data(), "r", "ReadGraphGR: Graph");
    size_t read = 0;
    uint64_t x[4];
    if (fread(x, sizeof(uint64_t), 4, fpin) != 4) {
      printf("Unable to read header\n");
    }
    if (x[0] != 1) /* version */
      printf("Unknown file version\n");
    sizeEdgeTy = x[1];
    // uint64_t sizeEdgeTy = le64toh(x[1]);
    uint64_t num_Node = x[2];
    uint64_t num_Edge = x[3];
    cout << graphFilePath + " has " << num_Node << " nodes and " << num_Edge
         << "  edges\n";

    // H_ERR(cudaMallocHost(&xadj, (num_Node + 1) * sizeof(uint)));
    // H_ERR(cudaMallocHost(&adjncy, num_Edge * sizeof(uint)));

    H_ERR(cudaMallocManaged(&G.xadj, (num_Node + 1) * sizeof(edge_t)));
    H_ERR(cudaMallocManaged(&G.adjncy, num_Edge * sizeof(vtx_t)));
    G.xadj[0] = 0; // CSR origin; Galois .gr stores endOfList[0..n-1], so xadj[0]
                   // is synthesized.
    mem_used += (num_Node + 1) * sizeof(edge_t) + num_Edge * sizeof(vtx_t);

    // adjwgt = nullptr;
    H_ERR(cudaMallocManaged(&G.adjwgt, num_Edge * sizeof(weight_t)));
    // um_used += num_Edge * sizeof(uint);
    weighted = true;
    if (!sizeEdgeTy) {
      // adjwgt = new uint[num_Edge];
      for (size_t i = 0; i < num_Edge; i++) {
        G.adjwgt[i] = 1;
      }
      weighted = false;
    }
    G.outDegree = new uint[num_Node];
    assert(G.xadj != NULL);
    assert(G.adjncy != NULL);
    // assert(vwgt != NULL);
    // assert(adjwgt != NULL);
    // Node offsets are stored as uint64_t in .gr — copy straight into edge_t.
    for (uint64_t i = 0; i < num_Node; i++) {
      uint64_t rs;
      if (fread(&rs, sizeof(uint64_t), 1, fpin) != 1)
        printf("Error: Unable to read node data\n");
      G.xadj[i + 1] = static_cast<edge_t>(rs);
    }
    // Edge destinations in Galois .gr are 32-bit. Widen into the 64-bit vtx_t
    // adjncy array via a streaming buffer so we don't need a giant temporary.
    {
      const size_t CHUNK = 1 << 20;
      vector<uint32_t> buf(CHUNK);
      uint64_t remaining = num_Edge;
      uint64_t idx = 0;
      while (remaining > 0) {
        size_t n = remaining < CHUNK ? remaining : CHUNK;
        size_t got = fread(buf.data(), sizeof(uint32_t), n, fpin);
        if (got != n) {
          printf("Error: Partial read of edge destinations\n");
          break;
        }
        for (size_t i = 0; i < got; i++)
          G.adjncy[idx + i] = static_cast<vtx_t>(buf[i]);
        idx += got;
        remaining -= got;
      }
    }
    for (size_t i = 0; i < num_Node; i++) {
      G.outDegree[i] = static_cast<uint>(G.xadj[i + 1] - G.xadj[i]);
    }
    uint64_t maxD = static_cast<uint64_t>(std::distance(
        G.outDegree, std::max_element(G.outDegree, G.outDegree + num_Node)));
    printf("vtx %llu has max out degree %u\n", (unsigned long long)maxD,
           G.outDegree[maxD]);
    if (sizeEdgeTy) {
      if (num_Edge % 2)
        if (fseek(fpin, 4, SEEK_CUR) != 0) // skip
          printf("Error when seeking\n");
      // Edge weights in .gr are 32-bit. Same widening path as edge dests.
      const size_t CHUNK = 1 << 20;
      vector<uint32_t> buf(CHUNK);
      uint64_t remaining = num_Edge;
      uint64_t idx = 0;
      while (remaining > 0) {
        size_t n = remaining < CHUNK ? remaining : CHUNK;
        size_t got = fread(buf.data(), sizeof(uint32_t), n, fpin);
        if (got != n) {
          printf("Error: Partial read of edge data\n");
          break;
        }
        for (size_t i = 0; i < got; i++)
          G.adjwgt[idx + i] = static_cast<weight_t>(buf[i]);
        idx += got;
        remaining -= got;
      }
    }
    G.mem_used = mem_used;
    G.numNode = num_Node;
    G.numEdge = num_Edge;
    gk_fclose(fpin);
  }
  void Load(graph_t<CSR> &G, bool _needweight = true) {
    this->graphFilePath = FLAGS_input;
    // this->weighted = FLAGS_weight||false;
    this->hasZeroID = false;
    this->needWeight = _needweight;
    G.needWeight = _needweight;
    string ext = GetExt(this->graphFilePath);
    if (IsBinaryCsr(ext))
      ReadGraphBCSR(G, ext);
    else
      ReadGraphGR(G);
    G.weighted = this->weighted;
  }
};

} // namespace mgg
#endif
