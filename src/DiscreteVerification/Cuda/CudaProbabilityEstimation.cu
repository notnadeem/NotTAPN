#include "DiscreteVerification/Alloc/CudaPetriNetAllocator.cuh"
#include "DiscreteVerification/Alloc/RealMarkingAllocator.cuh"
#include "DiscreteVerification/Alloc/RunResultAllocator.cuh"
#include "DiscreteVerification/Alloc/SMCQueryAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaAST.cuh"
#include "DiscreteVerification/Cuda/CudaQueryVisitor.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"
#include "DiscreteVerification/Cuda/CudaSMCQueryConverter.cuh"
#include "DiscreteVerification/Cuda/CudaTAPNConverter.cuh"
#include "DiscreteVerification/VerificationTypes/AtlerProbabilityEstimation.hpp"

#include <cuda_runtime.h>

namespace VerifyTAPN::DiscreteVerification {
using namespace VerifyTAPN::Cuda;
using namespace VerifyTAPN::Alloc;
// For now single kernel execution per run needed
// Since every run can have different execution time could be nice to try running multiple runs per kernel to improve
// warp utilization

__global__ void runSimulationKernel(Cuda::CudaRunResult *runner, Cuda::AST::CudaSMCQuery *query, int *successCount,
                                    int *runsNeeded, curandState *states, int *rand_seed, int *timeBound,
                                    int *stepBound) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int runNeed = *runsNeeded;
  if (tid >= runNeed) return;

  curand_init(*rand_seed, tid, 0, &states[tid]);

  // Copy global state to local memory for faster access
  curandState local_r_state = states[tid];

  if (tid % 1000 == 0) {
    printf("Thread %d initialized\n", tid);
  }

  // Cuda::CudaTimedArcPetriNet tapn = *ctapn;

  // TODO prepare per thread
  runner->prepare(&local_r_state);

  CudaSMCQuery lQuery = *query;

  int lTimeBound = *timeBound;
  int lStepBound = *stepBound;

  while (!runner->maximal && !(runner->totalTime >= lTimeBound || runner->totalSteps >= lStepBound)) {
    Cuda::CudaRealMarking *child = runner->realMarking;
    Cuda::CudaQueryVisitor checker(*child, *runner->tapn);
    Cuda::AST::BoolResult result;

    lQuery.accept(checker, result);

    if (result.value) {
      atomicAdd(successCount, 1);
      break;
    }

    runner->next(&local_r_state);
  }
}

__global__ void testAllocationKernel(CudaRunResult *runner, CudaRealMarking *marking, u_int *runNeed) {
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
           runner->tapn->transitions[i]->index, runner->tapn->transitions[i]->untimedPostset,
           runner->tapn->transitions[i]->urgent, runner->tapn->transitions[i]->controllable,
           runner->tapn->transitions[i]->_position.first, runner->tapn->transitions[i]->_position.second,
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
             runner->tapn->transitions[i]->inhibitorArcs,
             runner->tapn->transitions[i]->inhibitorArcs[0]->inputPlace->name,
             runner->tapn->transitions[i]->inhibitorArcs[0]->inputPlace);
    }
    printf("\nTimed Transition %d Has FiringMode: %d\n\n", i, runner->tapn->transitions[i]->_firingMode);
    for (size_t i = 0; i < runner->tapn->transitions[i]->presetLength; i++) {
      printf("Preset %d: %s\n", i, runner->tapn->transitions[i]->preset[i]->inputPlace->name);
    }
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
      printf("First transportArc Source Name: %s, Pointer: %p\n",
             runner->tapn->places[i]->transportArcs[0]->source->name,
             runner->tapn->places[i]->transportArcs[0]->source);
    }

    // Print prodTransportArcs
    if (runner->tapn->places[i]->prodTransportArcsLength > 0) {
      printf("Place has prodTransportArcs: %p\n", runner->tapn->places[i]->prodTransportArcs);
      printf("First prodTransportArc Source Name: %s, Pointer: %p\n",
             runner->tapn->places[i]->prodTransportArcs[0]->source->name,
             runner->tapn->places[i]->prodTransportArcs[0]->source);
    }

    // Print inhibitorArcs
    if (runner->tapn->places[i]->inhibitorArcsLength > 0) {
      printf("Place has inhibitorArcs: %p\n", runner->tapn->places[i]->inhibitorArcs);
      printf("First inhibitorArc InputPlace Name: %s, Pointer: %p\n",
             runner->tapn->places[i]->inhibitorArcs[0]->inputPlace->name,
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
    printf("Marking fromDelay: %2f\n", marking->fromDelay);
    printf("Marking generatedBy: %p\n", marking->generatedBy);
    printf("Marking places length: %d\n", marking->placesLength);

    for (size_t i = 0; i < marking->placesLength; i++) {
      CudaRealPlace *place = marking->places[i];
      printf("\nReal Place %u:\n", i);
      printf("Place Name: %s\n", place->place->name);
      printf("Total Token Count: %u\n", place->tokens->size);
      printf("Available Delay: %f\n", place->availableDelay() == HUGE_VAL ? 0.0f : place->availableDelay());

      // Print tokens
      printf("Tokens in place:\n");
      if (place->tokens->size == 0) {
        printf("No tokens in place\n");
      }
      for (size_t j = 0; j < place->tokens->size; j++) {
        CudaRealToken *token = place->tokens->get(j);
        printf("Token %d - Age: %f, Count: %d\n", j, token->age, token->count);
      }

      printf("Max Token Age: %f\n", place->maxTokenAge());
      printf("Is Empty: %d\n\n", place->isEmpty());
    }

    printf("\nInterval test\n");
    printf("d_array pointer is: %p\n", runner->transitionIntervals);
    printf("d_array size is: %llu\n", (unsigned long long)runner->transitionIntervals->size);

    for (int i = 0; i < runner->transitionIntervals->size; i++) {
      printf("%d. CudaDynamicArray<CudaInterval> inside dobbel aray\n", i);
      printf("Has Size: %llu, and Capacity: %llu\n", (unsigned long long)runner->transitionIntervals->arr[i]->size,
             (unsigned long long)runner->transitionIntervals->arr[i]->capacity);
      for (int j = 0; j < runner->transitionIntervals->arr[i]->size; j++) {
        printf("Intervals contained in %d CudaDynamicArray<CudaIntervals> are: low: %f high: %f\n", i,
               runner->transitionIntervals->arr[i]->arr->low, runner->transitionIntervals->arr[i]->arr->high);
      }
    }
  }
};

__global__ void testCudaSMCQueryAllocationKernel(CudaSMCQuery *query) {
  printf("Kernel executed\n");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > 0) return;

  printf("Thread id: %d\n", tid);
  printf("Quantifier: %d\n", query->getQuantifier());
  printf("Time Bound: %d\n", query->getSmcSettings().timeBound);
  printf("Step Bound: %d\n", query->getSmcSettings().stepBound);
  printf("False Positives: %f\n", query->getSmcSettings().falsePositives);
  printf("False Negatives: %f\n", query->getSmcSettings().falseNegatives);
  printf("Indifference Region Up: %f\n", query->getSmcSettings().indifferenceRegionUp);
  printf("Indifference Region Down: %f\n", query->getSmcSettings().indifferenceRegionDown);
  printf("Confidence: %f\n", query->getSmcSettings().confidence);
  printf("Estimation Interval Width: %f\n", query->getSmcSettings().estimationIntervalWidth);
  printf("Compare To Float: %d\n", query->getSmcSettings().compareToFloat);
  printf("Geq Than: %f\n", query->getSmcSettings().geqThan);

  // // Print details of the expression
  // CudaExpression *expr = query->getChild();
  // switch (expr->type) {
  //     case BOOL_EXPRESSION: {
  //         Cuda::AST::BoolExpression *boolExpr = static_cast<Cuda::AST::BoolExpression*>(expr->boolExpr);
  //         printf("Expression Type: BoolExpression\n");
  //         printf("Value: %d\n", boolExpr->value);
  //         break;
  //     }
  //     case NOT_EXPRESSION: {
  //         Cuda::AST::NotExpression *notExpr = static_cast<Cuda::AST::NotExpression*>(expr->notExpr);
  //         printf("Expression Type: NotExpression\n");
  //         CudaExpression *childExpr = notExpr->expr;
  //         if (childExpr->type == BOOL_EXPRESSION) {
  //             Cuda::AST::BoolExpression *boolChildExpr =
  //             static_cast<Cuda::AST::BoolExpression*>(childExpr->boolExpr); printf("Child Expression Type:
  //             BoolExpression\n"); printf("Child Value: %d\n", boolChildExpr->value);
  //         }
  //         break;
  //     }
  //     case DEADLOCK_EXPRESSION: {
  //         printf("Expression Type: DeadlockExpression\n");
  //         break;
  //     }
  //     case ATOMIC_PROPOSITION: {
  //         Cuda::AST::AtomicProposition *atomicExpr = static_cast<Cuda::AST::AtomicProposition*>(expr->atomicProp);
  //         printf("Expression Type: AtomicProposition\n");
  //         Cuda::AST::ArithmeticExpression *leftExpr = atomicExpr->left;
  //         Cuda::AST::ArithmeticExpression *rightExpr = atomicExpr->right;
  //         printf("Left Child Expression Type: %d\n", leftExpr->type);
  //         printf("Right Child Expression Type: %d\n", rightExpr->type);
  //     }
  //     case AND_EXPRESSION: {
  //         Cuda::AST::AndExpression *andExpr = static_cast<Cuda::AST::AndExpression*>(expr->andExpr);
  //         printf("Expression Type: AndExpression\n");
  //         CudaExpression *leftExpr = andExpr->left;
  //         if (leftExpr->type == BOOL_EXPRESSION) {
  //             Cuda::AST::BoolExpression *boolLeftExpr = static_cast<Cuda::AST::BoolExpression*>(leftExpr->boolExpr);
  //             printf("Left Child Expression Type: BoolExpression\n");
  //             printf("Left Child Value: %d\n", boolLeftExpr->value);
  //         }
  //         CudaExpression *rightExpr = andExpr->right;
  //         if (rightExpr->type == BOOL_EXPRESSION) {
  //             Cuda::AST::BoolExpression *boolRightExpr =
  //             static_cast<Cuda::AST::BoolExpression*>(rightExpr->boolExpr); printf("Right Child Expression Type:
  //             BoolExpression\n"); printf("Right Child Value: %d\n", boolRightExpr->value);
  //         }
  //         break;
  //     }
  //     case OR_EXPRESSION: {
  //         Cuda::AST::OrExpression *orExpr = static_cast<Cuda::AST::OrExpression*>(expr->orExpr);
  //         printf("Expression Type: OrExpression\n");
  //         CudaExpression *leftExpr = orExpr->left;
  //         if (leftExpr->type == BOOL_EXPRESSION) {
  //             Cuda::AST::BoolExpression *boolLeftExpr = static_cast<Cuda::AST::BoolExpression*>(leftExpr->boolExpr);
  //             printf("Left Child Expression Type: BoolExpression\n");
  //             printf("Left Child Value: %d\n", boolLeftExpr->value);
  //         }
  //         CudaExpression *rightExpr = orExpr->right;
  //         if (rightExpr->type == BOOL_EXPRESSION) {
  //             Cuda::AST::BoolExpression *boolRightExpr =
  //             static_cast<Cuda::AST::BoolExpression*>(rightExpr->boolExpr); printf("Right Child Expression Type:
  //             BoolExpression\n"); printf("Right Child Value: %d\n", boolRightExpr->value);
  //         }
  //         break;
  //     }
  //     default: {
  //         printf("Unknown Expression Type: %d\n", expr->type);
  //         break;
  //     }
  // }
}

bool AtlerProbabilityEstimation::runCuda() {
  std::cout << "Converting TAPN and marking..." << std::endl;
  auto result = VerifyTAPN::Cuda::CudaTAPNConverter::convert(tapn, initialMarking);
  VerifyTAPN::Cuda::CudaTimedArcPetriNet ctapn = result->first;
  VerifyTAPN::Cuda::CudaRealMarking *cipMarking = result->second;

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

  CudaTimedArcPetriNet *cptapn = &ctapn;

  auto runres = CudaRunResult(cptapn, cipMarking);

  auto runner = new CudaRunResult(cptapn, cipMarking);

  // Allocate the run result

  RunResultAllocator allocator;

  auto allocResult = allocator.allocate(runner, cipMarking, blocks, threadsPerBlock);

  // Allocate the query
  SMCQueryAllocator queryAllocator;
  std::cout << "Allocating query" << std::endl;
  CudaSMCQuery *d_cudaSMCQuery = queryAllocator.allocate(cudaSMCQuery);
  std::cout << "Query Allocation done" << std::endl;

  int successCountVal = 0;
  int *successCount;
  cudaMalloc(&successCount, sizeof(int));
  cudaMemcpy(successCount, &successCountVal, sizeof(int), cudaMemcpyHostToDevice);

  int *runsNeeded;
  cudaMalloc(&runsNeeded, sizeof(int));
  cudaMemcpy(runsNeeded, &this->runsNeeded, sizeof(int), cudaMemcpyHostToDevice);

  int *timeBound;
  cudaMalloc(&timeBound, sizeof(int));
  cudaMemcpy(timeBound, &cudaSMCQuery->smcSettings.timeBound, sizeof(int), cudaMemcpyHostToDevice);

  int *stepBound;
  cudaMalloc(&stepBound, sizeof(int));
  cudaMemcpy(stepBound, &cudaSMCQuery->smcSettings.stepBound, sizeof(int), cudaMemcpyHostToDevice);

  int rand_seed_val = 1234;
  int *rand_seed;
  cudaMalloc(&rand_seed, sizeof(int));
  cudaMemcpy(rand_seed, &rand_seed_val, sizeof(int), cudaMemcpyHostToDevice);

  CudaRunResult *runResultDevice = allocResult->first;
  CudaRealMarking *realMarkingDevice = allocResult->second;

  cudaDeviceSetLimit(cudaLimitStackSize, 256 * 1024);

  // testAllocationKernel<<<1, 1>>>(runResultDevice, realMarkingDevice, &this->runsNeeded);

  VerifyTAPN::DiscreteVerification::runSimulationKernel<<<100, threadsPerBlock>>>(
      runResultDevice, d_cudaSMCQuery, successCount, runsNeeded, runner->rngStates, rand_seed, timeBound, stepBound);

  cudaDeviceSynchronize();

  int successCountHost;
  cudaMemcpy(&successCountHost, successCount, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Success count: %d\n", successCountHost);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // testCudaSMCQueryAllocationKernel<<<1, 1>>>(d_cudaSMCQuery);

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

  std::cout << "Kernel execution completed successfully." << std::endl;

  return false;
}
} // namespace VerifyTAPN::DiscreteVerification