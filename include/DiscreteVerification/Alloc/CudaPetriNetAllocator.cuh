#ifndef CUDAPETRINETALLOCATOR_CUH
#define CUDAPETRINETALLOCATOR_CUH

#include "DiscreteVerification/Cuda/CudaTimedArcPetriNet.cuh"
#include "DiscreteVerification/Cuda/CudaTimedPlace.cuh"
#include <cuda_runtime.h>
#include <unordered_map>

namespace VerifyTAPN::Alloc {
using namespace Cuda;

struct CudaPetriNetAllocator {
  std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map;
  std::unordered_map<CudaTimedTransition *, CudaTimedTransition *> transition_map;
  std::unordered_map<CudaTimedInputArc *, CudaTimedInputArc *> inputArc_map;
  std::unordered_map<CudaTimedOutputArc *, CudaTimedOutputArc *> outputArc_map;
  std::unordered_map<CudaTimedTransportArc *, CudaTimedTransportArc *> transportArc_map;
  std::unordered_map<CudaTimedInhibitorArc *, CudaTimedInhibitorArc *> inhibitorArc_map;

  static CudaTimedPlace **cuda_allocate_places(CudaTimedArcPetriNet *h_petrinet,
                                               std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> &place_map) {

    CudaTimedPlace **d_places;
    cudaMalloc(&d_places, sizeof(CudaTimedPlace *) * h_petrinet->placesLength);

    for (int i = 0; i < h_petrinet->placesLength; i++) {
      CudaTimedPlace *d_place;
      cudaMalloc(&d_place, sizeof(CudaTimedPlace));

      char *d_name, *d_id;
      cudaMalloc(&d_name, sizeof(char) * h_petrinet->places[i]->nameLength);
      cudaMalloc(&d_id, sizeof(char) * h_petrinet->places[i]->idLength);

      cudaMemcpy(d_name, h_petrinet->places[i]->name, sizeof(char) * h_petrinet->places[i]->nameLength,
                 cudaMemcpyHostToDevice);
      cudaMemcpy(d_id, h_petrinet->places[i]->id, sizeof(char) * h_petrinet->places[i]->idLength,
                 cudaMemcpyHostToDevice);

      CudaTimedPlace *temp_place = h_petrinet->places[i];
      temp_place->name = d_name;
      temp_place->id = d_id;

      cudaMemcpy(d_place, temp_place, sizeof(CudaTimedPlace), cudaMemcpyHostToDevice);

      cudaMemcpy(&d_places[i], &d_place, sizeof(CudaTimedPlace *), cudaMemcpyHostToDevice);

      place_map[h_petrinet->places[i]] = d_place;
    }

    return d_places;
  }

  static CudaTimedTransition **
  cuda_allocate_timedTransitions(CudaTimedArcPetriNet *h_petrinet,
                                 std::unordered_map<CudaTimedTransition *, CudaTimedTransition *> &transition_map) {

    CudaTimedTransition **d_transitions;
    cudaMalloc(&d_transitions, sizeof(CudaTimedTransition *) * h_petrinet->transitionsLength);

    CudaTimedTransition **temp_transitions =
        (CudaTimedTransition **)malloc(sizeof(CudaTimedTransition *) * h_petrinet->transitionsLength);

    for (int i = 0; i < h_petrinet->transitionsLength; i++) {

      printf("TransAllocator: %d Name: %s, Id: %s\n", i, h_petrinet->transitions[i]->name,
             h_petrinet->transitions[i]->id);

      CudaTimedTransition *d_transition;
      cudaMalloc(&d_transition, sizeof(CudaTimedTransition));

      char *d_name, *d_id;
      cudaMalloc(&d_name, sizeof(char) * h_petrinet->transitions[i]->nameLength);
      printf("TransAllocator: %d Name: %s, Id: %s\n", i, h_petrinet->transitions[i]->name,
             h_petrinet->transitions[i]->id);
      cudaMalloc(&d_id, sizeof(char) * h_petrinet->transitions[i]->idLength);
      printf("TransAllocator: %d Name: %s, Id: %s\n", i, h_petrinet->transitions[i]->name,
             h_petrinet->transitions[i]->id);

      cudaMemcpy(d_name, h_petrinet->transitions[i]->name, sizeof(char) * h_petrinet->transitions[i]->nameLength,
                 cudaMemcpyHostToDevice);
      cudaMemcpy(d_id, h_petrinet->transitions[i]->id, sizeof(char) * h_petrinet->transitions[i]->idLength,
                 cudaMemcpyHostToDevice);

      h_petrinet->transitions[i]->id = d_id;
      h_petrinet->transitions[i]->name = d_name;

      cudaMemcpy(d_transition, h_petrinet->transitions[i], sizeof(CudaTimedTransition), cudaMemcpyHostToDevice);

      temp_transitions[i] = d_transition;

      transition_map[h_petrinet->transitions[i]] = d_transition;
    }

    cudaMemcpy(d_transitions, temp_transitions, sizeof(CudaTimedTransition *) * h_petrinet->transitionsLength,
               cudaMemcpyHostToDevice);

    free(temp_transitions);

    return d_transitions;
  }

  static CudaTimedInputArc **
  cuda_allocate_inputArcs(CudaTimedArcPetriNet *h_petrinet,
                          std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map,
                          std::unordered_map<CudaTimedTransition *, CudaTimedTransition *> transition_map,
                          std::unordered_map<CudaTimedInputArc *, CudaTimedInputArc *> &inputArc_map) {

    CudaTimedInputArc **d_inputArcs;
    cudaMalloc(&d_inputArcs, sizeof(CudaTimedInputArc *) * h_petrinet->inputArcsLength);

    CudaTimedInputArc **temp_inputArcs =
        (CudaTimedInputArc **)malloc(sizeof(CudaTimedInputArc *) * h_petrinet->inputArcsLength);

    for (int i = 0; i < h_petrinet->inputArcsLength; i++) {

      CudaTimedInputArc *d_inputArc;
      cudaMalloc(&d_inputArc, sizeof(CudaTimedInputArc));

      CudaTimedInputArc *temp_inputArc = (CudaTimedInputArc *)malloc(sizeof(CudaTimedInputArc));

      new (temp_inputArc) CudaTimedInputArc{.weight = h_petrinet->inputArcs[i]->weight};

      temp_inputArc->interval = h_petrinet->inputArcs[i]->interval;
      temp_inputArc->outputTransition = transition_map[h_petrinet->inputArcs[i]->outputTransition];
      temp_inputArc->inputPlace = place_map[h_petrinet->inputArcs[i]->inputPlace];

      cudaMemcpy(d_inputArc, temp_inputArc, sizeof(CudaTimedInputArc), cudaMemcpyHostToDevice);

      temp_inputArcs[i] = d_inputArc;
      inputArc_map[h_petrinet->inputArcs[i]] = d_inputArc;
    }

    cudaMemcpy(d_inputArcs, temp_inputArcs, sizeof(CudaTimedInputArc *) * h_petrinet->inputArcsLength,
               cudaMemcpyHostToDevice);

    free(temp_inputArcs);

    return d_inputArcs;
  }

  static CudaTimedOutputArc **
  cuda_allocate_outputArcs(CudaTimedArcPetriNet *h_petrinet,
                           std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map,
                           std::unordered_map<CudaTimedTransition *, CudaTimedTransition *> transition_map,
                           std::unordered_map<CudaTimedOutputArc *, CudaTimedOutputArc *> &outputArc_map) {
    CudaTimedOutputArc **d_outputArcs;

    cudaMalloc(&d_outputArcs, sizeof(CudaTimedOutputArc *) * h_petrinet->outputArcsLength);

    CudaTimedOutputArc **temp_outputArcs =
        (CudaTimedOutputArc **)malloc(sizeof(CudaTimedOutputArc *) * h_petrinet->outputArcsLength);

    for (int i = 0; i < h_petrinet->outputArcsLength; i++) {

      CudaTimedOutputArc *temp_outputArc = (CudaTimedOutputArc *)malloc(sizeof(CudaTimedOutputArc));

      CudaTimedOutputArc *d_outputArc;
      cudaMalloc(&d_outputArc, sizeof(CudaTimedOutputArc));

      new (temp_outputArc) CudaTimedOutputArc{.weight = h_petrinet->outputArcs[i]->weight};

      temp_outputArc->inputTransition = transition_map[h_petrinet->outputArcs[i]->inputTransition];

      temp_outputArc->outputPlace = place_map[h_petrinet->outputArcs[i]->outputPlace];

      cudaMemcpy(d_outputArc, temp_outputArc, sizeof(CudaTimedOutputArc), cudaMemcpyHostToDevice);

      temp_outputArcs[i] = d_outputArc;
      outputArc_map[h_petrinet->outputArcs[i]] = d_outputArc;
    }

    cudaMemcpy(d_outputArcs, temp_outputArcs, sizeof(CudaTimedOutputArc *) * h_petrinet->outputArcsLength,
               cudaMemcpyHostToDevice);

    free(temp_outputArcs);

    return d_outputArcs;
  }

  static CudaTimedTransportArc **
  cuda_allocate_transportArcs(CudaTimedArcPetriNet *h_petrinet,
                              std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map,
                              std::unordered_map<CudaTimedTransition *, CudaTimedTransition *> transition_map,
                              std::unordered_map<CudaTimedTransportArc *, CudaTimedTransportArc *> &transportArc_map) {

    CudaTimedTransportArc **d_transportArcs;
    cudaMalloc(&d_transportArcs, sizeof(CudaTimedTransportArc *) * h_petrinet->transportArcsLength);

    CudaTimedTransportArc **temp_transportArcs =
        (CudaTimedTransportArc **)malloc(sizeof(CudaTimedTransportArc *) * h_petrinet->transportArcsLength);

    for (int i = 0; i < h_petrinet->transportArcsLength; i++) {

      CudaTimedTransportArc *temp_transportArc = (CudaTimedTransportArc *)malloc(sizeof(CudaTimedTransportArc));

      CudaTimedTransportArc *d_transportArc;
      cudaMalloc(&d_transportArc, sizeof(CudaTimedTransportArc));

      new (temp_transportArc) CudaTimedTransportArc{.weight = h_petrinet->transportArcs[i]->weight};

      temp_transportArc->transition = transition_map[h_petrinet->transportArcs[i]->transition];

      temp_transportArc->source = place_map[h_petrinet->transportArcs[i]->source];

      temp_transportArc->destination = place_map[h_petrinet->transportArcs[i]->destination];

      temp_transportArc->interval = h_petrinet->transportArcs[i]->interval;

      cudaMemcpy(d_transportArc, temp_transportArc, sizeof(CudaTimedTransportArc), cudaMemcpyHostToDevice);

      transportArc_map[h_petrinet->transportArcs[i]] = d_transportArc;
      temp_transportArcs[i] = d_transportArc;
    }

    cudaMemcpy(d_transportArcs, temp_transportArcs, sizeof(CudaTimedTransportArc *) * h_petrinet->transportArcsLength,
               cudaMemcpyHostToDevice);

    free(temp_transportArcs);

    return d_transportArcs;
  }

  CudaTimedInhibitorArc **
  cuda_allocate_inhibitorArcs(CudaTimedArcPetriNet *h_petrinet,
                              std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map,
                              std::unordered_map<CudaTimedTransition *, CudaTimedTransition *> transition_map,
                              std::unordered_map<CudaTimedInhibitorArc *, CudaTimedInhibitorArc *> &inhibitorArc_map) {

    CudaTimedInhibitorArc **d_inhibitorArcs;
    cudaMalloc(&d_inhibitorArcs, sizeof(CudaTimedInhibitorArc *) * h_petrinet->inhibitorArcsLength);

    CudaTimedInhibitorArc **temp_inhibitorArcs =
        (CudaTimedInhibitorArc **)malloc(sizeof(CudaTimedInhibitorArc *) * h_petrinet->inhibitorArcsLength);

    for (int i = 0; i < h_petrinet->inhibitorArcsLength; i++) {

      CudaTimedInhibitorArc *temp_inhibitorArc = (CudaTimedInhibitorArc *)malloc(sizeof(CudaTimedInhibitorArc));

      CudaTimedInhibitorArc *d_inhibitorArc;
      cudaMalloc(&d_inhibitorArc, sizeof(CudaTimedInhibitorArc));

      temp_inhibitorArc->inputPlace = place_map[h_petrinet->inhibitorArcs[i]->inputPlace];

      temp_inhibitorArc->outputTransition = transition_map[h_petrinet->inhibitorArcs[i]->outputTransition];

      temp_inhibitorArc->weight = h_petrinet->inhibitorArcs[i]->weight;

      cudaMemcpy(d_inhibitorArc, temp_inhibitorArc, sizeof(CudaTimedInhibitorArc), cudaMemcpyHostToDevice);

      inhibitorArc_map[h_petrinet->inhibitorArcs[i]] = d_inhibitorArc;

      temp_inhibitorArcs[i] = d_inhibitorArc;
    }

    cudaMemcpy(d_inhibitorArcs, temp_inhibitorArcs, sizeof(CudaTimedInhibitorArc *) * h_petrinet->inhibitorArcsLength,
               cudaMemcpyHostToDevice);

    free(temp_inhibitorArcs);

    return d_inhibitorArcs;
  }

  void allocate_circular_dependencies_places(
      CudaTimedPlace **d_places, std::unordered_map<CudaTimedInputArc *, CudaTimedInputArc *> inputArc_map,
      std::unordered_map<CudaTimedTransportArc *, CudaTimedTransportArc *> transportArc_map,
      std::unordered_map<CudaTimedInhibitorArc *, CudaTimedInhibitorArc *> inhibitorArc_map,
      std::unordered_map<CudaTimedOutputArc *, CudaTimedOutputArc *> outputArc_map, CudaTimedArcPetriNet *h_petrinet) {

    CudaTimedPlace **temp_places = (CudaTimedPlace **)malloc(sizeof(CudaTimedPlace *) * h_petrinet->placesLength);

    cudaMemcpy(temp_places, d_places, sizeof(CudaTimedPlace *) * h_petrinet->placesLength, cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_petrinet->placesLength; i++) {
      CudaTimedInputArc **temp_place_inputArcs =
          (CudaTimedInputArc **)malloc(sizeof(CudaTimedInputArc *) * h_petrinet->places[i]->inputArcsLength);

      CudaTimedTransportArc **temp_place_transportArcs = (CudaTimedTransportArc **)malloc(
          sizeof(CudaTimedTransportArc *) * h_petrinet->places[i]->transportArcsLength);

      CudaTimedInhibitorArc **temp_place_inhibitorArcs = (CudaTimedInhibitorArc **)malloc(
          sizeof(CudaTimedInhibitorArc *) * h_petrinet->places[i]->inhibitorArcsLength);

      CudaTimedOutputArc **temp_place_outputArcs =
          (CudaTimedOutputArc **)malloc(sizeof(CudaTimedOutputArc *) * h_petrinet->places[i]->outputArcsLength);

      CudaTimedTransportArc **temp_place_prodTransportArc = (CudaTimedTransportArc **)malloc(
          sizeof(CudaTimedTransportArc *) * h_petrinet->places[i]->prodTransportArcsLength);

      for (int j = 0; j < h_petrinet->places[i]->prodTransportArcsLength; j++) {
        temp_place_prodTransportArc[j] = transportArc_map[h_petrinet->places[i]->prodTransportArcs[j]];
      }

      for (int j = 0; j < h_petrinet->places[i]->outputArcsLength; j++) {
        temp_place_outputArcs[j] = outputArc_map[h_petrinet->places[i]->outputArcs[j]];
      }

      for (int j = 0; j < h_petrinet->places[i]->inputArcsLength; j++) {
        temp_place_inputArcs[j] = inputArc_map[h_petrinet->places[i]->inputArcs[j]];
      }

      for (int j = 0; j < h_petrinet->places[i]->transportArcsLength; j++) {
        temp_place_transportArcs[j] = transportArc_map[h_petrinet->places[i]->transportArcs[j]];
      }

      for (int j = 0; j < h_petrinet->places[i]->inhibitorArcsLength; j++) {
        temp_place_inhibitorArcs[j] = inhibitorArc_map[h_petrinet->places[i]->inhibitorArcs[j]];
      }

      CudaTimedTransportArc **d_place_prodTransportArcs;
      cudaMalloc(&d_place_prodTransportArcs,
                 sizeof(CudaTimedTransportArc *) * h_petrinet->places[i]->prodTransportArcsLength);

      CudaTimedOutputArc **d_place_outputArcs;
      cudaMalloc(&d_place_outputArcs, sizeof(CudaTimedOutputArc *) * h_petrinet->places[i]->outputArcsLength);

      CudaTimedInhibitorArc **d_place_inhibitorArcs;
      cudaMalloc(&d_place_inhibitorArcs, sizeof(CudaTimedInhibitorArc *) * h_petrinet->places[i]->inhibitorArcsLength);

      CudaTimedTransportArc **d_place_transportArcs;
      cudaMalloc(&d_place_transportArcs, sizeof(CudaTimedTransportArc *) * h_petrinet->places[i]->transportArcsLength);

      CudaTimedInputArc **d_place_inputArcs;
      cudaMalloc(&d_place_inputArcs, sizeof(CudaTimedInputArc *) * h_petrinet->places[i]->inputArcsLength);

      cudaMemcpy(d_place_prodTransportArcs, temp_place_prodTransportArc,
                 sizeof(CudaTimedTransportArc *) * h_petrinet->places[i]->prodTransportArcsLength,
                 cudaMemcpyHostToDevice);

      cudaMemcpy(d_place_outputArcs, temp_place_outputArcs,
                 sizeof(CudaTimedOutputArc *) * h_petrinet->places[i]->outputArcsLength, cudaMemcpyHostToDevice);

      cudaMemcpy(d_place_inhibitorArcs, temp_place_inhibitorArcs,
                 sizeof(CudaTimedInhibitorArc *) * h_petrinet->places[i]->inhibitorArcsLength, cudaMemcpyHostToDevice);

      cudaMemcpy(d_place_transportArcs, temp_place_transportArcs,
                 sizeof(CudaTimedTransportArc *) * h_petrinet->places[i]->transportArcsLength, cudaMemcpyHostToDevice);

      cudaMemcpy(d_place_inputArcs, temp_place_inputArcs,
                 sizeof(CudaTimedInputArc *) * h_petrinet->places[i]->inputArcsLength, cudaMemcpyHostToDevice);

      cudaMemcpy(&(temp_places[i]->prodTransportArcs), &d_place_prodTransportArcs, sizeof(CudaTimedTransportArc **),
                 cudaMemcpyHostToDevice);

      cudaMemcpy(&(temp_places[i]->inputArcs), &d_place_inputArcs, sizeof(CudaTimedInputArc **),
                 cudaMemcpyHostToDevice);

      cudaMemcpy(&(temp_places[i]->outputArcs), &d_place_outputArcs, sizeof(CudaTimedOutputArc **),
                 cudaMemcpyHostToDevice);

      cudaMemcpy(&(temp_places[i]->inhibitorArcs), &d_place_inhibitorArcs, sizeof(CudaTimedInhibitorArc **),
                 cudaMemcpyHostToDevice);

      cudaMemcpy(&(temp_places[i]->transportArcs), &d_place_transportArcs, sizeof(CudaTimedTransportArc **),
                 cudaMemcpyHostToDevice);

      free(temp_place_prodTransportArc);
      free(temp_place_outputArcs);
      free(temp_place_inhibitorArcs);
      free(temp_place_inputArcs);
      free(temp_place_transportArcs);
    }

    cudaMemcpy(d_places, temp_places, sizeof(CudaTimedPlace *) * h_petrinet->placesLength, cudaMemcpyHostToDevice);

    free(temp_places);
  }

  void allocate_circular_dependencies_transitions(
      CudaTimedTransition **d_transitions, std::unordered_map<CudaTimedInputArc *, CudaTimedInputArc *> inputArc_map,
      std::unordered_map<CudaTimedOutputArc *, CudaTimedOutputArc *> outputArc_map,
      std::unordered_map<CudaTimedTransportArc *, CudaTimedTransportArc *> transportArc_map,
      std::unordered_map<CudaTimedInhibitorArc *, CudaTimedInhibitorArc *> inhibitorArc_map,
      CudaTimedArcPetriNet *h_petrinet) {

    CudaTimedTransition **temp_transitions =
        (CudaTimedTransition **)malloc(sizeof(CudaTimedTransition *) * h_petrinet->transitionsLength);

    cudaMemcpy(temp_transitions, d_transitions, sizeof(CudaTimedTransition *) * h_petrinet->transitionsLength,
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_petrinet->transitionsLength; i++) {

      CudaTimedInputArc **temp_transition_preset =
          (CudaTimedInputArc **)malloc(sizeof(CudaTimedInputArc *) * h_petrinet->transitions[i]->presetLength);

      CudaTimedTransportArc **temp_transition_transportArcs = (CudaTimedTransportArc **)malloc(
          sizeof(CudaTimedTransportArc *) * h_petrinet->transitions[i]->transportArcsLength);

      CudaTimedInhibitorArc **temp_transition_inhibitorArcs = (CudaTimedInhibitorArc **)malloc(
          sizeof(CudaTimedInhibitorArc *) * h_petrinet->transitions[i]->inhibitorArcsLength);

      CudaTimedOutputArc **temp_transition_postset =
          (CudaTimedOutputArc **)malloc(sizeof(CudaTimedOutputArc *) * h_petrinet->transitions[i]->postsetLength);

      for (int j = 0; j < h_petrinet->transitions[i]->postsetLength; j++) {
        temp_transition_postset[j] = outputArc_map[h_petrinet->transitions[i]->postset[j]];
      }

      for (int j = 0; j < h_petrinet->transitions[i]->presetLength; j++) {
        temp_transition_preset[j] = inputArc_map[h_petrinet->transitions[i]->preset[j]];
      }

      for (int j = 0; j < h_petrinet->transitions[i]->transportArcsLength; j++) {
        temp_transition_transportArcs[j] = transportArc_map[h_petrinet->transitions[i]->transportArcs[j]];
      }

      for (int j = 0; j < h_petrinet->transitions[i]->inhibitorArcsLength; j++) {
        temp_transition_inhibitorArcs[j] = inhibitorArc_map[h_petrinet->transitions[i]->inhibitorArcs[j]];
      }

      CudaTimedOutputArc **d_transition_postset;
      cudaMalloc(&d_transition_postset, sizeof(CudaTimedOutputArc *) * h_petrinet->transitions[i]->postsetLength);

      CudaTimedInhibitorArc **d_transition_inhibitorArcs;
      cudaMalloc(&d_transition_inhibitorArcs,
                 sizeof(CudaTimedInhibitorArc *) * h_petrinet->transitions[i]->inhibitorArcsLength);

      CudaTimedTransportArc **d_transition_transportArcs;
      cudaMalloc(&d_transition_transportArcs,
                 sizeof(CudaTimedTransportArc *) * h_petrinet->transitions[i]->transportArcsLength);

      CudaTimedInputArc **d_transition_preset;
      cudaMalloc(&d_transition_preset, sizeof(CudaTimedInputArc *) * h_petrinet->transitions[i]->presetLength);

      cudaMemcpy(d_transition_postset, temp_transition_postset,
                 sizeof(CudaTimedOutputArc *) * h_petrinet->transitions[i]->postsetLength, cudaMemcpyHostToDevice);

      cudaMemcpy(d_transition_inhibitorArcs, temp_transition_inhibitorArcs,
                 sizeof(CudaTimedInhibitorArc *) * h_petrinet->transitions[i]->inhibitorArcsLength,
                 cudaMemcpyHostToDevice);

      cudaMemcpy(d_transition_transportArcs, temp_transition_transportArcs,
                 sizeof(CudaTimedTransportArc *) * h_petrinet->transitions[i]->transportArcsLength,
                 cudaMemcpyHostToDevice);

      cudaMemcpy(d_transition_preset, temp_transition_preset,
                 sizeof(CudaTimedInputArc *) * h_petrinet->transitions[i]->presetLength, cudaMemcpyHostToDevice);

      cudaMemcpy(&(temp_transitions[i]->preset), &d_transition_preset, sizeof(CudaTimedInputArc **),
                 cudaMemcpyHostToDevice);

      cudaMemcpy(&(temp_transitions[i]->postset), &d_transition_postset, sizeof(CudaTimedOutputArc **),
                 cudaMemcpyHostToDevice);

      cudaMemcpy(&(temp_transitions[i]->inhibitorArcs), &d_transition_inhibitorArcs, sizeof(CudaTimedInhibitorArc **),
                 cudaMemcpyHostToDevice);

      cudaMemcpy(&(temp_transitions[i]->transportArcs), &d_transition_transportArcs, sizeof(CudaTimedTransportArc **),
                 cudaMemcpyHostToDevice);

      free(temp_transition_postset);
      free(temp_transition_inhibitorArcs);
      free(temp_transition_preset);
      free(temp_transition_transportArcs);
    }

    cudaMemcpy(d_transitions, temp_transitions, sizeof(CudaTimedTransition *) * h_petrinet->transitionsLength,
               cudaMemcpyHostToDevice);

    free(temp_transitions);
  }

  CudaTimedArcPetriNet *cuda_allocator(CudaTimedArcPetriNet *h_petrinet) {

    CudaTimedPlace **d_places = cuda_allocate_places(h_petrinet, place_map);
    printf("Allocated places\n");

    CudaTimedTransition **d_transitions = cuda_allocate_timedTransitions(h_petrinet, transition_map);
    printf("Allocated transitions\n");

    CudaTimedInputArc **d_inputArcs = cuda_allocate_inputArcs(h_petrinet, place_map, transition_map, inputArc_map);
    printf("Allocated inputArcs\n");

    CudaTimedOutputArc **d_outputArcs = cuda_allocate_outputArcs(h_petrinet, place_map, transition_map, outputArc_map);
    printf("Allocated outputArcs\n");

    CudaTimedTransportArc **d_transportArcs =
        cuda_allocate_transportArcs(h_petrinet, place_map, transition_map, transportArc_map);
    printf("Allocated transportArcs\n");

    CudaTimedInhibitorArc **d_inhibitorArcs =
        cuda_allocate_inhibitorArcs(h_petrinet, place_map, transition_map, inhibitorArc_map);
    printf("Allocated inhibitorArcs\n");

    allocate_circular_dependencies_places(d_places, inputArc_map, transportArc_map, inhibitorArc_map, outputArc_map,
                                          h_petrinet);
    printf("Allocated circular dependencies for places\n");

    allocate_circular_dependencies_transitions(d_transitions, inputArc_map, outputArc_map, transportArc_map,
                                               inhibitorArc_map, h_petrinet);
    printf("Allocated circular dependencies for transitions\n");

    CudaTimedArcPetriNet temp;

    temp.places = d_places;
    temp.placesLength = h_petrinet->placesLength;

    temp.inputArcs = d_inputArcs;
    temp.inputArcsLength = h_petrinet->inputArcsLength;

    temp.transitions = d_transitions;
    temp.transitionsLength = h_petrinet->transitionsLength;

    temp.outputArcs = d_outputArcs;
    temp.outputArcsLength = h_petrinet->outputArcsLength;

    temp.transportArcs = d_transportArcs;
    temp.transportArcsLength = h_petrinet->transportArcsLength;

    temp.inhibitorArcs = d_inhibitorArcs;
    temp.inhibitorArcsLength = h_petrinet->inhibitorArcsLength;

    temp.maxConstant = h_petrinet->maxConstant;
    temp.gcd = h_petrinet->gcd;

    CudaTimedArcPetriNet *d_petrinet;
    cudaMalloc(&d_petrinet, sizeof(CudaTimedArcPetriNet));
    cudaMemcpy(d_petrinet, &temp, sizeof(CudaTimedArcPetriNet), cudaMemcpyHostToDevice);
    printf("Allocated petrinet\n");

    return d_petrinet;
  }
};

} // namespace VerifyTAPN::Alloc

#endif // CUDAPETRINETALLOCATOR_CUH