#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <thread>
#include <functional>
#include <vector>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <limits>

#include <cblas.h>

#ifdef __cplusplus
extern "C"{
#endif 

#include <yael/binheap.h>
#include <yael/vector.h>
#include <yael/matrix.c>

int fvecs_read(const char *fname, int d, int n, float *v);
int ivecs_new_read (const char *fname, int *d_out, int **vi);
void fmat_mul_full(const float *left, const float *right,
                   int m, int n, int k,
                   const char *transp,
                   float *result);
int fvec_read (const char *fname, int d, float *a, int o_f);
int * ivec_new_read(const char *fname, int *d_out);
void fmat_rev_subtract_from_columns(int d,int n,float *m,const float *avg);
void fvec_sub (float * v1, const float * v2, long n);
struct fbinheap_s * fbinheap_new (int maxk);
int b2fvecs_read (const char *fname, int d, int n, float *v);
void fvec_add (float * v1, const float * v2, long n);
float* fmat_new_transp (const float *a, int ncol, int nrow);

#ifdef __cplusplus
}
#endif

using std::cout;
using std::ios;
using std::string;
using std::vector;

const int D = 96;
const int K = 16384;
const int M = 8;
const int rerankK = 256;
const int N = 1000000000;
const int queriesCount = 10000;
const int L = 32;
const int neighborsCount = 5000;
const int threadsCount = 29;

string coarseCodebookFilename = "./deep1B_coarse.fvecs";
string fineCodebookFilename = "./deep1B_fine.fvecs";
string alphaFilename = "./deep1B_alpha.fvecs";
string indexFilename = "./deep1B_index.dat";
string rerankCodesFilename = "./codes_deep1B_8.dat";
string rerankRotationFilename = "./deep1B_rerankVocabsRotation8.fvecs";
string rerankVocabsFilename = "./deep1B_rerankVocabs_8.fvecs";
string cellEdgesFilename = "./deep1B_cellEdges.dat";
string queryFilename = "./deep1B_queries.fvecs";
string groundFilename = "./deep1B_groundtruth.ivecs";

struct Record {
  int pointId;
  unsigned char bytes[M];
};

struct Searcher {
  float* coarseVocab;
  float* coarseNorms;
  float* fineVocab;
  float* fineNorms;
  float* alpha;
  float* coarseFineProducts;
  Record* index;
  int* cellEdges;
  float* coarseResiduals;
  float* rerankRotation;
  float* rerankVocabs;
};

void LoadCellEdgesPart(const string& cellEdgesFilename,
                       int startId, int count, int* cellEdges) {
  std::ifstream inputCellEdges(cellEdgesFilename.c_str(), ios::binary | ios::in);
  inputCellEdges.seekg(startId * sizeof(int));
  for(int i = 0; i < count; ++i) {
    inputCellEdges.read((char*)&(cellEdges[startId + i]), sizeof(int));
  }
  inputCellEdges.close();
}

void LoadCellEdges(const string& cellEdgesFilename, int N, int* cellEdges) {
  int perThreadCount = N / threadsCount;
  std::vector<std::thread> threads;
  for (int threadId = 0; threadId < threadsCount; ++ threadId) {
    int startId = threadId * perThreadCount;
    int count = (startId + perThreadCount > N) ? (N - startId) : perThreadCount;
    threads.push_back(std::thread(std::bind(LoadCellEdgesPart, cellEdgesFilename, startId, count, cellEdges)));
  }
  for (int threadId = 0; threadId < threadsCount; ++threadId) {
    threads[threadId].join();
  }
}

void LoadIndexPart(const string& indexFilename, const string& rerankFilename,
                   int startId, int count, Record* index) {
  std::ifstream inputIndex(indexFilename.c_str(), ios::binary | ios::in);
  inputIndex.seekg(startId * sizeof(int));
  std::ifstream inputRerank(rerankFilename.c_str(), ios::binary | ios::in);
  inputRerank.seekg(startId * sizeof(unsigned char) * M);
  for(int i = 0; i < count; ++i) {
    inputIndex.read((char*)&(index[startId + i].pointId), sizeof(int));
    for(int m = 0; m < M; ++m) {
      inputRerank.read((char*)&(index[startId + i].bytes[m]), sizeof(unsigned char));
    }
  }
  inputIndex.close();
  inputRerank.close();
}

void LoadIndex(const string& indexFilename, const string& rerankFilename, int N, Record* index) {
  int perThreadCount = N / threadsCount;
  std::vector<std::thread> threads;
  for (int threadId = 0; threadId < threadsCount; ++ threadId) {
    int startId = threadId * perThreadCount;
    int count = (startId + perThreadCount > N) ? (N - startId) : perThreadCount;
    threads.push_back(std::thread(std::bind(LoadIndexPart, indexFilename, rerankFilename, startId, count, index)));
  }
  for (int threadId = 0; threadId < threadsCount; ++ threadId) {
    threads[threadId].join();
  }  
}

void ReadAndPrecomputeVocabsData(Searcher& searcher) {
  searcher.coarseVocab = (float*) malloc(K * D * sizeof(float));
  fvecs_read(coarseCodebookFilename.c_str(), D, K, searcher.coarseVocab);
  searcher.fineVocab = (float*) malloc(K * D * sizeof(float));
  fvecs_read(fineCodebookFilename.c_str(), D, K, searcher.fineVocab);
  searcher.alpha = (float*) malloc(K * K * sizeof(float));
  fvecs_read(alphaFilename.c_str(), K, K, searcher.alpha);
  searcher.rerankRotation = (float*) malloc(D * D * sizeof(float));
  fvecs_read(rerankRotationFilename.c_str(), D, D, searcher.rerankRotation);
  float* temp = (float*) malloc(K * D * sizeof(float));
  fmat_mul_full(searcher.rerankRotation, searcher.coarseVocab,
                D, K, D, "TN", temp);
  memcpy(searcher.coarseVocab, temp, K * D * sizeof(float));
  free(temp);
  temp = (float*) malloc(K * D * sizeof(float));
  fmat_mul_full(searcher.rerankRotation, searcher.fineVocab,
                D, K, D, "TN", temp);
  memcpy(searcher.fineVocab, temp, K * D * sizeof(float));
  free(temp);
  searcher.coarseNorms = (float*) malloc(K * sizeof(float));
  searcher.fineNorms = (float*) malloc(K * sizeof(float));
  for(int i = 0; i < K; ++i) {
    searcher.coarseNorms[i] = fvec_norm2sqr(searcher.coarseVocab + D * i, D) / 2;
    searcher.fineNorms[i] = fvec_norm2sqr(searcher.fineVocab + D * i, D) / 2;
  }
  temp = (float*) malloc(K * K * sizeof(float));
  fmat_mul_full(searcher.coarseVocab, searcher.fineVocab,
                K, K, D, "TN", temp);
  searcher.coarseFineProducts = fmat_new_transp(temp, K, K);
  free(temp);
  int Dread;
  cout << "Before allocation...\n";
  searcher.index = (Record*) malloc(N * sizeof(Record));
  LoadIndex(indexFilename, rerankCodesFilename, N, searcher.index);
  searcher.cellEdges = (int*) malloc(K * K * sizeof(int));
  LoadCellEdges(cellEdgesFilename, N, searcher.cellEdges);
  searcher.coarseResiduals = (float*)malloc(L * D * sizeof(float));
  cout << "Before local vocabs reading...\n"; 
  searcher.rerankVocabs = (float*)malloc(rerankK * D * K * sizeof(float));
  fvecs_read(rerankVocabsFilename.c_str(), D / M, K * M * rerankK, searcher.rerankVocabs);
  cout << "Search data is prepared...\n";  
}

void SearchNearestNeighbors(const Searcher& searcher,
                            const float* queries,
                            int neighborsCount,
                            vector<vector<std::pair<float, int> > >& result) {
  result.resize(queriesCount);
  vector<float> queryCoarseDistance(K);
  vector<float> queryFineDistance(K);
  vector<std::pair<float, int> > coarseList(K);
  for(int qid = 0; qid < queriesCount; ++qid) {
    result[qid].resize(neighborsCount, std::make_pair(std::numeric_limits<float>::max(), 0));
  }
  struct fbinheap_s* heap = fbinheap_new(L);
  vector<int> topPointers(L);
  float* residual = (float*)malloc(D * sizeof(float));
  float* preallocatedVocabs = (float*)malloc(L * rerankK * D * sizeof(float));
  int subDim = D / M;
  vector<std::pair<float, int> > scores(L * K);
  vector<int> coarseIdToTopId(K);
  std::clock_t c_start = std::clock();
  for(int qid = 0; qid < queriesCount; ++qid) {
    cout << qid << "\n";
    int found = 0;
    fmat_mul_full(searcher.coarseVocab, queries + qid * D,
                  K, 1, D, "TN", &(queryCoarseDistance[0]));
    fmat_mul_full(searcher.fineVocab, queries + qid * D,
                  K, 1, D, "TN", &(queryFineDistance[0]));
    for(int c = 0; c < K; ++c) {
      coarseList[c].first = searcher.coarseNorms[c] - queryCoarseDistance[c];
      coarseList[c].second = c;
    }
    std::sort(coarseList.begin(), coarseList.end());
    for(int l = 0; l < L; ++l) {
      int coarseId = coarseList[l].second;
      coarseIdToTopId[coarseId] = l;
      for(int k = 0; k < K; ++k) {
        int cellId = coarseId * K + k;
        float alphaFactor = searcher.alpha[cellId];
        scores[l*K+k].first = coarseList[l].first + searcher.fineNorms[k] * alphaFactor * alphaFactor
                              - queryFineDistance[k] * alphaFactor + searcher.coarseFineProducts[cellId] * alphaFactor;
        scores[l*K+k].second = cellId;
      }
      memcpy(searcher.coarseResiduals + l * D, searcher.coarseVocab + D * coarseId, D * sizeof(float));
      fvec_rev_sub(searcher.coarseResiduals + l * D, queries + qid * D, D);
      memcpy(preallocatedVocabs + l * rerankK * D, searcher.rerankVocabs + coarseId * rerankK * D, rerankK * D * sizeof(float));
    }
    int cellsCount = neighborsCount * ((float)K * K / N);
    std::nth_element(scores.begin(), scores.begin() + cellsCount, scores.end());
    std::sort(scores.begin(), scores.begin() + cellsCount);
    int currentPointer = 0;
    int cellTraversed = 0;
    while(found < neighborsCount) {
      cellTraversed += 1;
      int cellId = scores[currentPointer].second;
      int topListId = coarseIdToTopId[cellId / K];
      ++currentPointer;
      int cellStart = (cellId == 0) ? 0 : searcher.cellEdges[cellId - 1];
      int cellFinish = searcher.cellEdges[cellId];
      if(cellStart == cellFinish) {
        continue;
      }
      memcpy(residual, searcher.coarseResiduals + topListId * D, D * sizeof(float));
      cblas_saxpy(D, -1.0 * searcher.alpha[cellId], searcher.fineVocab + (cellId % K) * D, 1, residual, 1);
      float* cellVocab = preallocatedVocabs + topListId * rerankK * D;
      for(int id = cellStart; id < cellFinish && found < neighborsCount; ++id) {
        result[qid][found].second = searcher.index[id].pointId;
        result[qid][found].first = 0.0;
        float diff = 0.0;
        for(int m = 0; m < M; ++m) {
          float* codeword = cellVocab + m * rerankK * subDim + searcher.index[id].bytes[m] * subDim;
          float* residualSubvector = residual + m * subDim;
          for(int d = 0; d < subDim; ++d) {
            diff = residualSubvector[d] - codeword[d];
            result[qid][found].first += diff * diff;
          }
        }
        ++found;
      }
    }
    std::sort(result[qid].begin(), result[qid].end());
  }
  std::clock_t c_end = std::clock();
  std::cout << std::fixed << std::setprecision(2) << "CPU time used: "
              << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / queriesCount << " ms\n";
  fbinheap_delete(heap);
  free(residual);
  free(searcher.coarseVocab);
  free(searcher.coarseNorms);
  free(searcher.fineVocab);
  free(searcher.fineNorms);
  free(searcher.alpha);
  free(searcher.index);
  free(searcher.cellEdges);
  free(searcher.coarseFineProducts);
}

float computeRecallAt(const vector<vector<std::pair<float, int> > >& result,
                      const int* groundtruth, int R) {
  int limit = (R < result[0].size()) ? R : result[0].size();
  int positive = 0;
  for(int i = 0; i < result.size(); ++i) {
    for(int j = 0; j < limit; ++j) {
      if(result[i][j].second == groundtruth[i]) {
        ++positive;
      }
    }
  }
  return (float(positive) / result.size());
}
void ComputeRecall(const vector<vector<std::pair<float, int> > >& result,
                   const int* groundtruth) {
  for(int i = 0; i < 19; ++i) {
    int R = std::pow(2.0, i);
    std::cout << std::fixed << std::setprecision(5) << "Recall@ " << R << ": " << 
                 computeRecallAt(result, groundtruth, R) << "\n";
  }
}

int main() {
  Searcher searcher;
  ReadAndPrecomputeVocabsData(searcher);
  float* queries = (float*) malloc(queriesCount * D * sizeof(float));
  fvecs_read(queryFilename.c_str(), D, queriesCount, queries);
  float* temp = (float*) malloc(queriesCount * D * sizeof(float));
  fmat_mul_full(searcher.rerankRotation, queries,
                D, queriesCount, D, "TN", temp);
  memcpy(queries, temp, queriesCount * D * sizeof(float));
  free(temp);  
  vector<vector<std::pair<float, int> > > result;
  SearchNearestNeighbors(searcher, queries, neighborsCount, result);
  cout << "Before reading groundtruth...\n";
  int* groundtruth = (int*) malloc(queriesCount * 1 * sizeof(int));
  int d;
  ivecs_new_read(groundFilename.c_str(), &d, &groundtruth);
  cout << "Groundtruth is read\n";
  ComputeRecall(result, groundtruth); 
  return 0;
}
