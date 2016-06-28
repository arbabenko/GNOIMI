#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <functional>
#include <vector>
#include <ctime>
#include <chrono>

#include <cblas.h>

#ifdef __cplusplus
extern "C"{
#endif

#include <yael/kmeans.h>
#include <yael/vector.h>
#include <yael/matrix.c>

int fvecs_read(const char *fname, int d, int n, float *v);
int ivecs_new_read(const char *fname, int *d_out, int **vi);
void fmat_mul_full(const float *left, const float *right,
                   int m, int n, int k,
                   const char *transp,
                   float *result);
int fvec_read(const char *fname, int d, float *a, int o_f);
int* ivec_new_read(const char *fname, int *d_out);
void fmat_rev_subtract_from_columns(int d,int n,float *m,const float *avg);
void fvec_sub(float * v1, const float * v2, long n);
int b2fvecs_read(const char *fname, int d, int n, float *v);
void fvec_add(float * v1, const float * v2, long n);
float* fmat_new_transp(const float *a, int ncol, int nrow);

#ifdef __cplusplus
}
#endif

using std::cout;
using std::ios;
using std::string;
using std::vector;

int D = 96;
int K = 256;
int totalLearnCount = 1000000;
int learnIterationsCount = 10;
int L = 8;
string learnFilename = "./deep10M.fvecs";
string initCoarseFilename = "./coarse.fvecs";
string initFineFilename = "./fine.fvecs";
string outputFilesPrefix = "./test_";
int trainThreadChunkSize = 10000;
int threadsCount = 25;
int* coarseAssigns = (int*)malloc(totalLearnCount * sizeof(int));
int* fineAssigns = (int*)malloc(totalLearnCount * sizeof(int));
float* alphaNum = (float*)malloc(K * K * sizeof(float));
float* alphaDen = (float*)malloc(K * K * sizeof(float));
float* alpha = (float*)malloc(K * K * sizeof(float));

vector<float*> alphaNumerators(threadsCount);
vector<float*> alphaDenominators(threadsCount);

float* coarseVocab = (float*)malloc(D * K * sizeof(float));
float* fineVocab = (float*)malloc(D * K * sizeof(float));
float* fineVocabNum = (float*)malloc(D * K * sizeof(float));
float* fineVocabDen = (float*)malloc(K * sizeof(float));
float* coarseVocabNum = (float*)malloc(D * K * sizeof(float));
float* coarseVocabDen = (float*)malloc(K * sizeof(float));

vector<float*> fineVocabNumerators(threadsCount);
vector<float*> fineVocabDenominators(threadsCount);
vector<float*> coarseVocabNumerators(threadsCount);
vector<float*> coarseVocabDenominators(threadsCount);

float* coarseNorms = (float*)malloc(K * sizeof(float));
float* fineNorms = (float*)malloc(K * sizeof(float));
float* coarseFineProducts = (float*)malloc(K * K * sizeof(float));

float* errors = (float*)malloc(threadsCount * sizeof(float));

///////////////////////////
void computeOptimalAssignsSubset(int threadId) {
  long long startId = (totalLearnCount / threadsCount) * threadId;
  int pointsCount = totalLearnCount / threadsCount;
  int chunksCount = pointsCount / trainThreadChunkSize;
  float* pointsCoarseTerms = (float*)malloc(trainThreadChunkSize * K * sizeof(float));
  float* pointsFineTerms = (float*)malloc(trainThreadChunkSize * K * sizeof(float));
  errors[threadId] = 0.0;
  FILE* learnStream = fopen(learnFilename.c_str(), "r");
  fseek(learnStream, startId * (D + 1) * sizeof(float), SEEK_SET);
  float* chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));
  std::vector<std::pair<float, int> > coarseScores(K);
  for(int chunkId = 0; chunkId < chunksCount; ++chunkId) {
    std::cout << "[Assigns][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << chunksCount << "\n";
    fvecs_fread(learnStream, chunkPoints, trainThreadChunkSize, D);
    fmat_mul_full(coarseVocab, chunkPoints, K, trainThreadChunkSize, D, "TN", pointsCoarseTerms);
    fmat_mul_full(fineVocab, chunkPoints, K, trainThreadChunkSize, D, "TN", pointsFineTerms);
    for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {
      cblas_saxpy(K, -1.0, coarseNorms, 1, pointsCoarseTerms + pointId * K, 1);
      for(int k = 0; k < K; ++k) {
        coarseScores[k].first = (-1.0) * pointsCoarseTerms[pointId * K + k];
        coarseScores[k].second = k;
      }
      std::sort(coarseScores.begin(), coarseScores.end());
      float currentMinScore = 999999999.0;
      int currentMinCoarseId = -1;
      int currentMinFineId = -1;
      for(int l = 0; l < L; ++l) {
        //examine cluster l
        int currentCoarseId = coarseScores[l].second;
        float currentCoarseTerm = coarseScores[l].first;
        for(int currentFineId = 0; currentFineId < K; ++currentFineId) {
          float alphaFactor = alpha[currentCoarseId * K + currentFineId];
          float score = currentCoarseTerm + alphaFactor * coarseFineProducts[currentCoarseId * K + currentFineId] + 
                        (-1.0) * alphaFactor * pointsFineTerms[pointId * K + currentFineId] + 
                        alphaFactor * alphaFactor * fineNorms[currentFineId];
          if(score < currentMinScore) {
            currentMinScore = score;
            currentMinCoarseId = currentCoarseId;
            currentMinFineId = currentFineId;
          }
        }
      }
      coarseAssigns[startId + chunkId * trainThreadChunkSize + pointId] = currentMinCoarseId;
      fineAssigns[startId + chunkId * trainThreadChunkSize + pointId] = currentMinFineId;
      errors[threadId] += currentMinScore * 2 + 1.0; // point has a norm equals 1.0
    }
  }
  fclose(learnStream);
  free(chunkPoints);
  free(pointsCoarseTerms);
  free(pointsFineTerms);
}

void computeOptimalAlphaSubset(int threadId) {
  memset(alphaNumerators[threadId], 0, K * K * sizeof(float));
  memset(alphaDenominators[threadId], 0, K * K * sizeof(float));
  long long startId = (totalLearnCount / threadsCount) * threadId;
  int pointsCount = totalLearnCount / threadsCount;
  int chunksCount = pointsCount / trainThreadChunkSize;
  FILE* learnStream = fopen(learnFilename.c_str(), "r");
  fseek(learnStream, startId * (D + 1) * sizeof(float), SEEK_SET);
  float* residual = (float*)malloc(D * sizeof(float));
  float* chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));
  for(int chunkId = 0; chunkId < chunksCount; ++chunkId) {
    std::cout << "[Alpha][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << chunksCount << "\n";
    fvecs_fread(learnStream, chunkPoints, trainThreadChunkSize, D);
    for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {
      int coarseAssign = coarseAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      int fineAssign = fineAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      memcpy(residual, chunkPoints + pointId * D, D * sizeof(float));
      cblas_saxpy(D, -1.0, coarseVocab + coarseAssign * D, 1, residual, 1);
      alphaNumerators[threadId][coarseAssign * K + fineAssign] += 
           cblas_sdot(D, residual, 1, fineVocab + fineAssign * D, 1);
      alphaDenominators[threadId][coarseAssign * K + fineAssign] += fineNorms[fineAssign] * 2; // we keep halves of norms 
    }
  }
  fclose(learnStream);
  free(chunkPoints);
  free(residual);
}

void computeOptimalFineVocabSubset(int threadId) {
  memset(fineVocabNumerators[threadId], 0, K * D * sizeof(float));
  memset(fineVocabDenominators[threadId], 0, K * sizeof(float));
  long long startId = (totalLearnCount / threadsCount) * threadId;
  int pointsCount = totalLearnCount / threadsCount;
  int chunksCount = pointsCount / trainThreadChunkSize;
  FILE* learnStream = fopen(learnFilename.c_str(), "r");
  fseek(learnStream, startId * (D + 1) * sizeof(float), SEEK_SET);
  float* residual = (float*)malloc(D * sizeof(float));
  float* chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));
  for(int chunkId = 0; chunkId < chunksCount; ++chunkId) {
    std::cout << "[Fine vocabs][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << chunksCount << "\n";
    fvecs_fread(learnStream, chunkPoints, trainThreadChunkSize, D);
    for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {
      int coarseAssign = coarseAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      int fineAssign = fineAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      float alphaFactor = alpha[coarseAssign * K + fineAssign];
      memcpy(residual, chunkPoints + pointId * D, D * sizeof(float));
      cblas_saxpy(D, -1.0, coarseVocab + coarseAssign * D, 1, residual, 1);
      cblas_saxpy(D, alphaFactor, residual, 1, fineVocabNumerators[threadId] + fineAssign * D, 1);
      fineVocabDenominators[threadId][fineAssign] += alphaFactor * alphaFactor;
    }
  }
  fclose(learnStream);
  free(chunkPoints);
  free(residual);
}

void computeOptimalCoarseVocabSubset(int threadId) {
  memset(coarseVocabNumerators[threadId], 0, K * D * sizeof(float));
  memset(coarseVocabDenominators[threadId], 0, K * sizeof(float));
  long long startId = (totalLearnCount / threadsCount) * threadId;
  int pointsCount = totalLearnCount / threadsCount;
  int chunksCount = pointsCount / trainThreadChunkSize;
  FILE* learnStream = fopen(learnFilename.c_str(), "r");
  fseek(learnStream, startId * (D + 1) * sizeof(float), SEEK_SET);
  float* residual = (float*)malloc(D * sizeof(float));
  float* chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));
  for(int chunkId = 0; chunkId < chunksCount; ++chunkId) {
    std::cout << "[Coarse vocabs][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << chunksCount << "\n";
    fvecs_fread(learnStream, chunkPoints, trainThreadChunkSize, D);
    for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {
      int coarseAssign = coarseAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      int fineAssign = fineAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      float alphaFactor = alpha[coarseAssign * K + fineAssign];
      memcpy(residual, chunkPoints + pointId * D, D * sizeof(float));
      cblas_saxpy(D, -1.0 * alphaFactor, fineVocab + fineAssign * D, 1, residual, 1);
      cblas_saxpy(D, 1, residual, 1, coarseVocabNumerators[threadId] + coarseAssign * D, 1);
      coarseVocabDenominators[threadId][coarseAssign] += 1.0;
    }
  }
  fclose(learnStream);
  free(chunkPoints);
  free(residual);
}

int main() {
  for(int threadId = 0; threadId < threadsCount; ++threadId) {
    alphaNumerators[threadId] = (float*)malloc(K * K * sizeof(float*));
    alphaDenominators[threadId] = (float*)malloc(K * K * sizeof(float*));
  }
  for(int threadId = 0; threadId < threadsCount; ++threadId) {
    fineVocabNumerators[threadId] = (float*)malloc(K * D * sizeof(float*));
    fineVocabDenominators[threadId] = (float*)malloc(K * sizeof(float*));
  }  
  for(int threadId = 0; threadId < threadsCount; ++threadId) {
    coarseVocabNumerators[threadId] = (float*)malloc(K * D * sizeof(float));
    coarseVocabDenominators[threadId] = (float*)malloc(K * sizeof(float));
  }
  // init vocabs
  fvecs_read(initCoarseFilename.c_str(), D, K, coarseVocab);
  fvecs_read(initFineFilename.c_str(), D, K, fineVocab);
  // init alpha
  for(int i = 0; i < K * K; ++i) {
    alpha[i] = 1.0;
  }
  // learn iterations
  std::cout << "Start learning iterations...\n";
  for(int it = 0; it < learnIterationsCount; ++it) {
    for(int k = 0; k < K; ++k) {
      coarseNorms[k] = cblas_sdot(D, coarseVocab + k * D, 1, coarseVocab + k * D, 1) / 2;
      fineNorms[k] = cblas_sdot(D, fineVocab + k * D, 1, fineVocab + k * D, 1) / 2;
    }
    fmat_mul_full(fineVocab, coarseVocab, K, K, D, "TN", coarseFineProducts);
    // update Assigns
    vector<std::thread> workers;
    memset(errors, 0, threadsCount * sizeof(float));
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      workers.push_back(std::thread(computeOptimalAssignsSubset, threadId));
    }
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      workers[threadId].join();
    }
    float totalError = 0.0;
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      totalError += errors[threadId];
    }
    std::cout << "Current reconstruction error... " << totalError / totalLearnCount << "\n";
    workers.clear();
    // update alpha
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      workers.push_back(std::thread(computeOptimalAlphaSubset, threadId));
    }
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      workers[threadId].join();
    }
    workers.clear();
    memset(alphaNum, 0, K * K * sizeof(float));
    memset(alphaDen, 0, K * K * sizeof(float));
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      cblas_saxpy(K * K, 1, alphaNumerators[threadId], 1, alphaNum, 1);
      cblas_saxpy(K * K, 1, alphaDenominators[threadId], 1, alphaDen, 1);
    }
    for(int i = 0; i < K * K; ++i) {
      alpha[i] = (alphaDen[i] == 0) ? 1.0 : alphaNum[i] / alphaDen[i];
    }
    // update fine Vocabs
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      workers.push_back(std::thread(computeOptimalFineVocabSubset, threadId));
    }
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      workers[threadId].join();
    }
    workers.clear();
    memset(fineVocabNum, 0, K * D * sizeof(float));
    memset(fineVocabDen, 0, K * sizeof(float));
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      cblas_saxpy(K * D, 1, fineVocabNumerators[threadId], 1, fineVocabNum, 1);
      cblas_saxpy(K, 1, fineVocabDenominators[threadId], 1, fineVocabDen, 1);
    }
    for(int i = 0; i < K * D; ++i) {
      fineVocab[i] = (fineVocabDen[i / D] == 0) ? 0 : fineVocabNum[i] / fineVocabDen[i / D];
    }
    // update coarse Vocabs
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      workers.push_back(std::thread(computeOptimalCoarseVocabSubset, threadId));
    }
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      workers[threadId].join();
    }
    workers.clear();
    memset(coarseVocabNum, 0, K * D * sizeof(float));
    memset(coarseVocabDen, 0, K * sizeof(float));
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      cblas_saxpy(K * D, 1, coarseVocabNumerators[threadId], 1, coarseVocabNum, 1);
      cblas_saxpy(K, 1, coarseVocabDenominators[threadId], 1, coarseVocabDen, 1);
    }
    for(int i = 0; i < K * D; ++i) {
      coarseVocab[i] = (coarseVocabDen[i / D] == 0) ? 0 : coarseVocabNum[i] / coarseVocabDen[i / D];
    }
    // save current alpha and vocabs
    std::stringstream alphaFilename;
    alphaFilename << outputFilesPrefix << "alpha_" << it << ".dat";
    std::ofstream outAlpha(alphaFilename.str().c_str(), ios::binary | ios::out);
    outAlpha.write((char*)alpha, K * K * sizeof(float));
    outAlpha.close();
    std::stringstream fineVocabFilename;
    fineVocabFilename << outputFilesPrefix << "fine_" << it << ".dat";
    std::ofstream outFine(fineVocabFilename.str().c_str(), ios::binary | ios::out);
    outFine.write((char*)fineVocab, K * D * sizeof(float));
    outFine.close();
    std::stringstream coarseVocabFilename;
    coarseVocabFilename << outputFilesPrefix << "coarse_" << it << ".dat";
    std::ofstream outCoarse(coarseVocabFilename.str().c_str(), ios::binary | ios::out);
    outCoarse.write((char*)coarseVocab, K * D * sizeof(float));
    outCoarse.close();
  }
  free(coarseAssigns);
  free(fineAssigns);
  free(alphaNum);
  free(alphaDen);
  free(alpha);
  free(coarseVocab);
  free(coarseVocabNum);
  free(coarseVocabDen);
  free(fineVocab);
  free(fineVocabNum);
  free(fineVocabDen);
  free(coarseNorms);
  free(fineNorms);
  free(coarseFineProducts);
  free(errors);
  return 0;
}











