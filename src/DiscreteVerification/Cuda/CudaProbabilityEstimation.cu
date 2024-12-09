#include "DiscreteVerification/Alloc/CudaPetriNetAllocator.cuh"
#include "DiscreteVerification/Alloc/RealMarkingAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaAST.cuh"
#include "DiscreteVerification/Cuda/CudaQueryVisitor.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"
#include "DiscreteVerification/Cuda/CudaSMCQueryConverter.cuh"
#include "DiscreteVerification/Cuda/CudaTAPNConverter.cuh"
#include "DiscreteVerification/VerificationTypes/AtlerProbabilityEstimation.hpp"
#include "DiscreteVerification/Alloc/RunResultAllocator.cuh"
#include "DiscreteVerification/Alloc/SMCQueryAllocator.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::DiscreteVerification {
using namespace VerifyTAPN::Cuda;
using namespace VerifyTAPN::Alloc;
//For now single kernel execution per run needed
//Since every run can have different execution time could be nice to try running multiple runs per kernel to improve warp utilization

__global__ void runSimulationKernel(Cuda::CudaTimedArcPetriNet *ctapn, Cuda::CudaRealMarking *initialMarking,
                                    Cuda::AST::CudaSMCQuery *query, Cuda::CudaRunResult *runner, int *timeBound,
                                    int *stepBound, int *successCount, int *runsNeeded, curandState *states, int *rand_seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int runNeed = *runsNeeded;
  if (tid >= runNeed) return;

  curand_init(*rand_seed, tid, 0, &states[tid]);

  // Copy global state to local memory for faster access
  curandState local_r_state = states[tid];

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
    newMarking = runner->next(&local_r_state);
  }
}

__global__ void testAllocationKernel(CudaRunResult *runner,
                                     CudaRealMarking *marking, u_int *runNeed) {
  printf("Kernel executed\n");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > 0) return;

  printf("Thread id: %d\n", tid);
  // printf("Petri net maxConstant: %s\n", pn->maxConstant);
  printf("Petri net placesLength: %d\n", runner->tapn->placesLength);

  printf("__Run_Transitions_Test__:\n\n");
  printf("Num of transitions is: %llu\n", (unsigned long long)runner->tapn->transitionsLength);

  for (int i = 0; i < runner->tapn->transitionsLength; i++) {
    printf("TimedTransition with index: %d, UntimedPostSet: %d, Urgent: %d, "
           "Controllable: %d, Position: < %f , %f >, Weight: %f, Name: %s, Id: "
           "%s\n",
           runner->tapn->transitions[i]->index, runner->tapn->transitions[i]->untimedPostset, runner->tapn->transitions[i]->urgent,
           runner->tapn->transitions[i]->controllable, runner->tapn->transitions[i]->_position.first, runner->tapn->transitions[i]->_position.second,
           runner->tapn->transitions[i]->_weight, runner->tapn->transitions[i]->name, runner->tapn->transitions[i]->id);
    printf("Transition has pointer: %p to preset, Name place in preset double "
           "pointer preset[0]: %s with pointer: %p\n",
           runner->tapn->transitions[i]->preset, runner->tapn->transitions[i]->preset[0]->inputPlace->name,
           runner->tapn->transitions[i]->preset[0]->inputPlace);
    printf("Transition has pointer: %p to postset, Name place in postset "
           "double pointer postset[0]: %s, with pointer: %p\n",
           runner->tapn->transitions[i]->postset, runner->tapn->transitions[i]->postset[0]->outputPlace->name,
           runner->tapn->transitions[i]->postset[0]->outputPlace);
    if (runner->tapn->transitions[i]->transportArcsLength > 0) {
      printf("Transition has pointer: %p to transportArcs, Source name in "
             "transportarcs double pointer transportArcs[0]: %s, with pointer: %p\n",
             runner->tapn->transitions[i]->transportArcs, runner->tapn->transitions[i]->transportArcs[0]->source->name,
             runner->tapn->transitions[i]->transportArcs[0]->source);
    }
    if (runner->tapn->transitions[i]->inhibitorArcsLength > 0) {
      printf("Transition has pointer: %p to inhibitorArcs, Place name in "
             "inhibitorArc double pointer inhibitor[0]: %s, with pointer: %p\n",
             runner->tapn->transitions[i]->inhibitorArcs, runner->tapn->transitions[i]->inhibitorArcs[0]->inputPlace->name,
             runner->tapn->transitions[i]->inhibitorArcs[0]->inputPlace);
    }
    printf("\nTimed Transition %d Has FiringMode: %d\n\n", i, runner->tapn->transitions[i]->_firingMode);
  }

  printf("Logging all places and their token counts:\n");
  for (size_t i = 0; i < runner->tapn->placesLength; i++) {
    const char *place_name = runner->tapn->places[i]->name;
    // Assume h_marking->places[i] corresponds to runner->tapn->places[i] // Adjust based on your marking structure
    printf("Place %d:, Token Count: %s\n", i, place_name);

    // Print inputArcs
    if (runner->tapn->places[i]->inputArcsLength > 0) {
      printf("Place has inputArcs: %p\n", runner->tapn->places[i]->inputArcs);
      printf("First inputArc Weight: %s, Pointer: %p\n", runner->tapn->places[i]->inputArcs[0]->weight,
             runner->tapn->places[i]->inputArcs[0]);
    }

    // Print transportArcs
    if (runner->tapn->places[i]->transportArcsLength > 0) {
      printf("Place has transportArcs: %p\n", runner->tapn->places[i]->transportArcs);
      printf("First transportArc Source Name: %s, Pointer: %p\n", runner->tapn->places[i]->transportArcs[0]->source->name,
             runner->tapn->places[i]->transportArcs[0]->source);
    }

    // Print prodTransportArcs
    if (runner->tapn->places[i]->prodTransportArcsLength > 0) {
      printf("Place has prodTransportArcs: %p\n", runner->tapn->places[i]->prodTransportArcs);
      printf("First prodTransportArc Source Name: %s, Pointer: %p\n", runner->tapn->places[i]->prodTransportArcs[0]->source->name,
             runner->tapn->places[i]->prodTransportArcs[0]->source);
    }

    // Print inhibitorArcs
    if (runner->tapn->places[i]->inhibitorArcsLength > 0) {
      printf("Place has inhibitorArcs: %p\n", runner->tapn->places[i]->inhibitorArcs);
      printf("First inhibitorArc InputPlace Name: %s, Pointer: %p\n", runner->tapn->places[i]->inhibitorArcs[0]->inputPlace->name,
             runner->tapn->places[i]->inhibitorArcs[0]->inputPlace);
    }

    // Print outputArcs
    if (runner->tapn->places[i]->outputArcsLength > 0) {
      printf("Place has outputArcs: %p\n", runner->tapn->places[i]->outputArcs);
      printf("First outputArc Weight: %s, Pointer: %p\n", runner->tapn->places[i]->outputArcs[0]->weight,
             runner->tapn->places[i]->outputArcs[0]);
    }

    printf("\n");

    printf("\nLogging Real Marking details:\n");
    printf("Marking deadlocked: %d\n", marking->deadlocked);
    printf("Marking fromDelay: %f\n", marking->fromDelay);
    printf("Marking generatedBy: %p\n", marking->generatedBy);
    printf("Marking places length: %zu\n", marking->placesLength);

    for (size_t i = 0; i < marking->placesLength; i++) {
      CudaRealPlace *place = marking->places[i];
      printf("\nReal Place %u:\n", i);
      printf("Place Name: %s\n", place->place->name);
      printf("Total Token Count: %d\n", place->totalTokenCount());
      printf("Available Delay: %f\n", place->availableDelay());

      // Print tokens
      printf("Tokens in place:\n");
      for (size_t j = 0; j < place->tokens->size; j++) {
        CudaRealToken *token = place->tokens->get(j);
        printf("Token %zu - Age: %f, Count: %d\n", j, token->age, token->count);
      }

      printf("Max Token Age: %f\n", place->maxTokenAge());
      printf("Is Empty: %d\n\n", place->isEmpty());
    }
  }
};

// __global__ void testCudaSMCQueryAllocationKernel(CudaSMCQuery *query) {
//     printf("Kernel executed\n");
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid > 0) return;

//     printf("Thread id: %d\n", tid);
//     printf("Quantifier: %d\n", query->getQuantifier());
//     printf("Time Bound: %d\n", query->getSmcSettings().timeBound);
//     printf("Step Bound: %d\n", query->getSmcSettings().stepBound);
//     printf("False Positives: %f\n", query->getSmcSettings().falsePositives);
//     printf("False Negatives: %f\n", query->getSmcSettings().falseNegatives);
//     printf("Indifference Region Up: %f\n", query->getSmcSettings().indifferenceRegionUp);
//     printf("Indifference Region Down: %f\n", query->getSmcSettings().indifferenceRegionDown);
//     printf("Confidence: %f\n", query->getSmcSettings().confidence);
//     printf("Estimation Interval Width: %f\n", query->getSmcSettings().estimationIntervalWidth);
//     printf("Compare To Float: %d\n", query->getSmcSettings().compareToFloat);
//     printf("Geq Than: %f\n", query->getSmcSettings().geqThan);

//     // Print details of the expression
//     CudaExpression *expr = query->getChild();
//     switch (expr->getType()) {
//         case BOOL_EXPRESSION: {
//             Cuda::AST::BoolExpression *boolExpr = static_cast<Cuda::AST::BoolExpression*>(expr);
//             printf("Expression Type: BoolExpression\n");
//             printf("Value: %d\n", boolExpr->getValue());
//             break;
//         }
//         case NOT_EXPRESSION: {
//             Cuda::AST::NotExpression *notExpr = static_cast<Cuda::AST::NotExpression*>(expr);
//             printf("Expression Type: NotExpression\n");
//             // Recursively print child expression details
//             CudaExpression *childExpr = &notExpr->getChild();
//             if (childExpr->getType() == BOOL_EXPRESSION) {
//                 Cuda::AST::BoolExpression *boolChildExpr = static_cast<Cuda::AST::BoolExpression*>(childExpr);
//                 printf("Child Expression Type: BoolExpression\n");
//                 printf("Child Value: %d\n", boolChildExpr->getValue());
//             }
//             break;
//         }
//         case DEADLOCK_EXPRESSION: {
//             printf("Expression Type: DeadlockExpression\n");
//             break;
//         }
//         case ATOMIC_PROPOSITION: {
//             Cuda::AST::AtomicProposition *atomicExpr = static_cast<Cuda::AST::AtomicProposition*>(expr);
//             printf("Expression Type: AtomicProposition\n");
//             // Recursively print left and right child expression details
//             Cuda::AST::ArithmeticExpression *leftExpr = &atomicExpr->getLeft();
//             Cuda::AST::ArithmeticExpression *rightExpr = &atomicExpr->getRight();
//             printf("Left Child Expression Type: %d\n", leftExpr->getType());
//             printf("Right Child Expression Type: %d\n", rightExpr->getType());
//         }
//         case AND_EXPRESSION: {
//             Cuda::AST::AndExpression *andExpr = static_cast<Cuda::AST::AndExpression*>(expr);
//             printf("Expression Type: AndExpression\n");
//             // Recursively print left and right child expression details
//             CudaExpression *leftExpr = &andExpr->getLeft();
//             if (leftExpr->getType() == BOOL_EXPRESSION) {
//                 Cuda::AST::BoolExpression *boolLeftExpr = static_cast<Cuda::AST::BoolExpression*>(leftExpr);
//                 printf("Left Child Expression Type: BoolExpression\n");
//                 printf("Left Child Value: %d\n", boolLeftExpr->getValue());
//             }
//             CudaExpression *rightExpr = &andExpr->getRight();
//             if (rightExpr->getType() == BOOL_EXPRESSION) {
//                 Cuda::AST::BoolExpression *boolRightExpr = static_cast<Cuda::AST::BoolExpression*>(rightExpr);
//                 printf("Right Child Expression Type: BoolExpression\n");
//                 printf("Right Child Value: %d\n", boolRightExpr->getValue());
//             }
//             break;
//         }
//         case OR_EXPRESSION: {
//             Cuda::AST::OrExpression *orExpr = static_cast<Cuda::AST::OrExpression*>(expr);
//             printf("Expression Type: OrExpression\n");
//             // Recursively print left and right child expression details
//             CudaExpression *leftExpr = &orExpr->getLeft();
//             if (leftExpr->getType() == BOOL_EXPRESSION) {
//                 Cuda::AST::BoolExpression *boolLeftExpr = static_cast<Cuda::AST::BoolExpression*>(leftExpr);
//                 printf("Left Child Expression Type: BoolExpression\n");
//                 printf("Left Child Value: %d\n", boolLeftExpr->getValue());
//             }
//             CudaExpression *rightExpr = &orExpr->getRight();
//             if (rightExpr->getType() == BOOL_EXPRESSION) {
//                 Cuda::AST::BoolExpression *boolRightExpr = static_cast<Cuda::AST::BoolExpression*>(rightExpr);
//                 printf("Right Child Expression Type: BoolExpression\n");
//                 printf("Right Child Value: %d\n", boolRightExpr->getValue());
//             }
//             break;
//         }
//         default: {
//             printf("Unknown Expression Type: %d\n", expr->getType());
//             break;
//         }
//     }
// }

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

  auto runres = CudaRunResult(&ctapn);

  auto runner = new CudaRunResult(&ctapn);

  // Allocate the run result
  
  RunResultAllocator allocator;

  auto allocResult = allocator.allocate(runner, &ciMarking, blocks, threadsPerBlock);

  CudaRunResult *runResultDevice = allocResult->first;
  CudaRealMarking *realMarkingDevice = allocResult->second;

  //testAllocationKernel<<<1, 1>>>(runResultDevice, realMarkingDevice, &this->runsNeeded);
  // Allocate the initial marking

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // Allocate the query
  SMCQueryAllocator queryAllocator;
  std::cout << "Allocating query" << std::endl;
  CudaSMCQuery *d_cudaSMCQuery = queryAllocator.allocate(cudaSMCQuery);
  std::cout << "Query Allocation done" << std::endl;


  //testCudaSMCQueryAllocationKernel<<<1, 1>>>(d_cudaSMCQuery);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
      return false;
  }


  // cudaError_t allocStatus = cudaGetLastError();
  // if (allocStatus != cudaSuccess) {
  //   std::cerr << "cudaMalloc failed: " << cudaGetErrorString(allocStatus) << std::endl;
  // } else {
  //   std::cout << "Device memory for curand allocated successfully." << std::endl;
  // }

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