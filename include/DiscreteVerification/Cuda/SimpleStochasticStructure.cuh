#ifndef SIMPLE_STOCHASTIC_STRUCTURE_CUH_
#define SIMPLE_STOCHASTIC_STRUCTURE_CUH_

// TODO: Check if this is correct

/*#include <sstream>*/
#include <cuda_runtime.h>
#include <string>

namespace VerifyTAPN::Cuda::SimpleSMC {

enum FiringMode { Oldest, Youngest, Random };

//Not sure about this one
__host__ __device__ const char* firingModeName(FiringMode type);

enum DistributionType {
  Constant,
  Uniform,
  Exponential,
  Normal,
  Gamma,
  Erlang,
  DiscreteUniform,
  Geometric
};

//Not sure about this one
__host__ __device__ const char* distributionName(DistributionType type);

struct SMCUniformParameters {
  double a;
  double b;
};
struct SMCExponentialParameters {
  double rate;
};
struct SMCNormalParameters {
  double mean;
  double stddev;
};
struct SMCConstantParameters {
  double value;
};
struct SMCGammaParameters {
  double shape;
  double scale;
};
struct SMCDiscreteUniformParameters {
  int a;
  int b;
};
struct SMCGeometricParameters {
  double p;
};

union DistributionParameters {
  SMCUniformParameters uniform;
  SMCExponentialParameters exp;
  SMCNormalParameters normal;
  SMCConstantParameters constant;
  SMCGammaParameters gamma;
  SMCDiscreteUniformParameters discreteUniform;
  SMCGeometricParameters geometric;
};

struct Distribution {
  DistributionType type;
  DistributionParameters parameters;

  template <typename T>
   __host__ __device__ double sample(T &engine, const unsigned int precision = 0) const {
    double date = 0;
    switch (type) {
    case Constant:
      date = parameters.constant.value;
      break;
    case Uniform:
      date = curand_uniform(parameters.uniform.a,
                                            parameters.uniform.b)(engine);
      break;
    case Exponential:
      //TODO
      break;
    case Normal:
      date = curand_normal(parameters.normal.mean,
                                      parameters.normal.stddev)(engine);
      break;
    case Gamma:
    case Erlang:
      //TODO
      break;
    case DiscreteUniform:
      //TODO
      break;
    case Geometric:
      //TODO
      break;
    }
    if (precision > 0) {
      double factor = pow(10.0, precision);
      date = round(date * factor) / factor;
    }
    return max(date, 0.0);
  }

  __host__ __device__ static Distribution urgent();

  __host__ __device__ static Distribution defaultDistribution();

  __host__ __device__ static Distribution fromParams(int distrib_id, double param1, double param2);

  std::string toXML() const;
};

} // namespace VerifyTAPN::Atler::SimpleSMC

#endif
