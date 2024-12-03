#include "DiscreteVerification/Alloc/CudaPetriNetAllocator.cuh"
#include "DiscreteVerification/Alloc/RealMarkingAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaAST.cuh"
#include "DiscreteVerification/Cuda/CudaQueryVisitor.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"
#include "DiscreteVerification/Cuda/CudaSMCQueryConverter.cuh"
#include "DiscreteVerification/Cuda/CudaTAPNConverter.cuh"
#include "DiscreteVerification/VerificationTypes/AtlerProbabilityEstimation.hpp"

#include <cuda_runtime.h>

namespace VerifyTAPN::DiscreteVerification {
using namespace VerifyTAPN::Cuda;

__global__ void runSimulationKernel(Cuda::CudaTimedArcPetriNet *ctapn, Cuda::CudaRealMarking *initialMarking,
                                    Cuda::AST::CudaSMCQuery *query, Cuda::CudaRunResult *runner, int *timeBound,
                                    int *stepBound, int *successCount, int *runsNeeded, curandState *states) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int runNeed = *runsNeeded;
  if (tid >= runNeed) return;

  curand_init(clock64(), tid, 0, &states[tid]);
  if (tid % 1000 == 0) {
    printf("Thread %d initialized\n", tid);
  }

  int tBound = *timeBound;
  int sBound = *stepBound;

  Cuda::CudaTimedArcPetriNet tapn = *ctapn;

  // TODO prepare per thread
  // runner.prepare(initialMarking);
  Cuda::CudaRealMarking *newMarking = runner->parent;

  while (!runner->maximal && !(runner->totalTime >= tBound || runner->totalSteps >= sBound)) {

    Cuda::CudaRealMarking *child = newMarking->clone();
    Cuda::CudaQueryVisitor checker(*child, tapn);
    Cuda::AST::BoolResult result;

    query->accept(checker, result);

    if (result.value) {
      atomicAdd(successCount, 1);
      break;
    }
    newMarking = runner->next(tid);
  }
}

__global__ void testAllocationKernel(VerifyTAPN::Cuda::CudaTimedArcPetriNet *pn,
                                     VerifyTAPN::Cuda::CudaRealMarking *marking, u_int *runNeed) {
  printf("Kernel executed\n");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > 0) return;

  printf("Thread id: %d\n", tid);
  // printf("Petri net maxConstant: %s\n", pn->maxConstant);
  printf("Petri net placesLength: %d\n", pn->placesLength);

  printf("__Run_Transitions_Test__:\n\n");
  printf("Num of transitions is: %llu\n", (unsigned long long)pn->transitionsLength);

  for (int i = 0; i < pn->transitionsLength; i++) {
    printf("TimedTransition with index: %d, UntimedPostSet: %d, Urgent: %d, "
           "Controllable: %d, Position: < %f , %f >, Weight: %f, Name: %s, Id: "
           "%s\n",
           pn->transitions[i]->index, pn->transitions[i]->untimedPostset, pn->transitions[i]->urgent,
           pn->transitions[i]->controllable, pn->transitions[i]->_position.first, pn->transitions[i]->_position.second,
           pn->transitions[i]->_weight, pn->transitions[i]->name, pn->transitions[i]->id);
    printf("Transition has pointer: %p to preset, Name place in preset double "
           "pointer preset[0]: %s with pointer: %p\n",
           pn->transitions[i]->preset, pn->transitions[i]->preset[0]->inputPlace->name,
           pn->transitions[i]->preset[0]->inputPlace);
    printf("Transition has pointer: %p to postset, Name place in postset "
           "double pointer postset[0]: %s, with pointer: %p\n",
           pn->transitions[i]->postset, pn->transitions[i]->postset[0]->outputPlace->name,
           pn->transitions[i]->postset[0]->outputPlace);
    if (pn->transitions[i]->transportArcsLength > 0) {
      printf("Transition has pointer: %p to transportArcs, Source name in "
             "transportarcs double pointer transportArcs[0]: %s, with pointer: %p\n",
             pn->transitions[i]->transportArcs, pn->transitions[i]->transportArcs[0]->source->name,
             pn->transitions[i]->transportArcs[0]->source);
    }
    if (pn->transitions[i]->inhibitorArcsLength > 0) {
      printf("Transition has pointer: %p to inhibitorArcs, Place name in "
             "inhibitorArc double pointer inhibitor[0]: %s, with pointer: %p\n",
             pn->transitions[i]->inhibitorArcs, pn->transitions[i]->inhibitorArcs[0]->inputPlace->name,
             pn->transitions[i]->inhibitorArcs[0]->inputPlace);
    }
    printf("\nTimed Transition %d Has FiringMode: %d\n\n", i, pn->transitions[i]->_firingMode);
  }

  printf("Logging all places and their token counts:\n");
  for (size_t i = 0; i < pn->placesLength; i++) {
    const char *place_name = pn->places[i]->name;
    // Assume h_marking->places[i] corresponds to pn->places[i] // Adjust based on your marking structure
    printf("Place %zu: Name: %s, Token Count: %d\n", i, place_name);

    // Print inputArcs
    if (pn->places[i]->inputArcsLength > 0) {
      printf("Place has inputArcs: %p\n", pn->places[i]->inputArcs);
      printf("First inputArc Weight: %s, Pointer: %p\n", pn->places[i]->inputArcs[0]->weight,
             pn->places[i]->inputArcs[0]);
    }

    // Print transportArcs
    if (pn->places[i]->transportArcsLength > 0) {
      printf("Place has transportArcs: %p\n", pn->places[i]->transportArcs);
      printf("First transportArc Source Name: %s, Pointer: %p\n", pn->places[i]->transportArcs[0]->source->name,
             pn->places[i]->transportArcs[0]->source);
    }

    // Print prodTransportArcs
    if (pn->places[i]->prodTransportArcsLength > 0) {
      printf("Place has prodTransportArcs: %p\n", pn->places[i]->prodTransportArcs);
      printf("First prodTransportArc Source Name: %s, Pointer: %p\n", pn->places[i]->prodTransportArcs[0]->source->name,
             pn->places[i]->prodTransportArcs[0]->source);
    }

    // Print inhibitorArcs
    if (pn->places[i]->inhibitorArcsLength > 0) {
      printf("Place has inhibitorArcs: %p\n", pn->places[i]->inhibitorArcs);
      printf("First inhibitorArc InputPlace Name: %s, Pointer: %p\n", pn->places[i]->inhibitorArcs[0]->inputPlace->name,
             pn->places[i]->inhibitorArcs[0]->inputPlace);
    }

    // Print outputArcs
    if (pn->places[i]->outputArcsLength > 0) {
      printf("Place has outputArcs: %p\n", pn->places[i]->outputArcs);
      printf("First outputArc Weight: %s, Pointer: %p\n", pn->places[i]->outputArcs[0]->weight,
             pn->places[i]->outputArcs[0]);
    }

    printf("\n");
  }
};

bool AtlerProbabilityEstimation::runCuda() {
  std::cout << "Converting TAPN and marking..." << std::endl;
  auto result = VerifyTAPN::Cuda::CudaTAPNConverter::convert(tapn, initialMarking);
  VerifyTAPN::Cuda::CudaTimedArcPetriNet ctapn = result->first;
  VerifyTAPN::Cuda::CudaRealMarking ciMarking = result->second;

  std::cout << "Converting Query..." << std::endl;
  SMCQuery *currentSMCQuery = static_cast<SMCQuery *>(query);
  VerifyTAPN::Cuda::AST::CudaSMCQuery *cudaSMCQuery = VerifyTAPN::Cuda::CudaSMCQueryConverter::convert(currentSMCQuery);

  // std::cout << "Converting Options..." << std::endl;
  // VerifyTAPN::Cuda::CudaVerificationOptions cudaOptions = Cuda::CudaOptionsConverter::convert(options);

  // TODO: Convert the PlaceVisitor to a simple representation
  // NOTE: Also find a way to simplify the representation of the PlaceVisitor

  std::cout << "Creating run generator..." << std::endl;

  const unsigned int threadsPerBlock = 256;

  // Calculate the number of blocks needed
  unsigned int blocks = (this->runsNeeded + threadsPerBlock - 1) / threadsPerBlock;

  std::cout << "Runs needed..." << this->runsNeeded << std::endl;
  std::cout << "Threads per block..." << threadsPerBlock << std::endl;
  std::cout << "Blocks..." << blocks << std::endl;

  // Allocate the petry net

  VerifyTAPN::Alloc::CudaPetriNetAllocator pnAllocator = VerifyTAPN::Alloc::CudaPetriNetAllocator();
  VerifyTAPN::Alloc::RealMarkingAllocator markingAllocator;

  VerifyTAPN::Cuda::CudaTimedArcPetriNet *pn = pnAllocator.cuda_allocator(&ctapn);
  VerifyTAPN::Cuda::CudaRealMarking *marking =
      markingAllocator.allocate_real_marking(&ciMarking, pnAllocator.transition_map, pnAllocator.place_map);

  testAllocationKernel<<<1, 1>>>(pn, marking, &this->runsNeeded);
  // Allocate the initial marking

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  err = cudaDeviceSynchronize();
  // // Allocate the query

  // cudaError_t allocStatus = cudaGetLastError();
  // if (allocStatus != cudaSuccess) {
  //   std::cerr << "cudaMalloc failed: " << cudaGetErrorString(allocStatus) << std::endl;
  // } else {
  //   std::cout << "Device memory for curand allocated successfully." << std::endl;
  // }

  auto runres = new VerifyTAPN::Cuda::CudaRunResult(ctapn);

  std::cout << "Run prepare" << std::endl;

  // VerifyTAPN::DiscreteVerification::runSimulationKernel<<<blocks, threads>>>(
  //     stapn, ciMarking, cudaSMCQuery, runres, smcSettings.timeBound, smcSettings.stepBound, 0, runsNeeded);

  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //   std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  //   return false;
  // }

  // err = cudaDeviceSynchronize();

  // if (err != cudaSuccess) {
  //   std::cerr << "CUDA device synchronization failed: " << cudaGetErrorString(err) << std::endl;
  //   return false;
  // }

  std::cout << "Kernel execution completed successfully." << std::endl;

  return false;
}
} // namespace VerifyTAPN::DiscreteVerification