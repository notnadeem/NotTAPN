#ifndef CUDA_STOCHASTIC_STRUCTURE_CUH_
#define CUDA_STOCHASTIC_STRUCTURE_CUH_

// TODO: Check if this is correct

/*#include <sstream>*/

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>

namespace VerifyTAPN::Cuda::CudaSMC
{

  enum FiringMode
  {
    Oldest,
    Youngest,
    Random
  };

  // Not sure about this one
  __host__ __device__ const char *firingModeName(FiringMode type);

  enum DistributionType
  {
    Constant,
    Uniform,
    Exponential,
    Normal,
    Gamma,
    Erlang,
    DiscreteUniform,
    Geometric
  };

  // Not sure about this one
  __host__ __device__ const char *distributionName(DistributionType type);

  struct SMCUniformParameters
  {
    double a;
    double b;
  };
  struct SMCExponentialParameters
  {
    double rate;
  };
  struct SMCNormalParameters
  {
    double mean;
    double stddev;
  };
  struct SMCConstantParameters
  {
    double value;
  };
  struct SMCGammaParameters
  {
    double shape;
    double scale;
  };
  struct SMCDiscreteUniformParameters
  {
    int a;
    int b;
  };
  struct SMCGeometricParameters
  {
    double p;
  };

  union DistributionParameters
  {
    SMCUniformParameters uniform;
    SMCExponentialParameters exp;
    SMCNormalParameters normal;
    SMCConstantParameters constant;
    SMCGammaParameters gamma;
    SMCDiscreteUniformParameters discreteUniform;
    SMCGeometricParameters geometric;
  };

  __device__ double gamrnd_d(double shape, double scale, curandState_t *state);

  struct Distribution
  {
    DistributionType type;
    DistributionParameters parameters;

    __device__ double sample(curandState_t *state, const unsigned int precision = 0) const
    {
      if (!isValid()) {
        printf("Error: Invalid parameters for distribution type %d\n", type);
        __trap();
      }
      double date = 0;
      switch (type)
      {
      case Constant:
        date = parameters.constant.value;
        break;
      case Uniform:
        date = parameters.uniform.a + (parameters.uniform.b - parameters.uniform.a) * curand_uniform_double(state);
        break;
      case Exponential:
        date = -logf(1 - curand_uniform_double(state)) / parameters.exp.rate;
        break;
      case Normal:
        date = parameters.normal.mean + parameters.normal.stddev * curand_normal_double(state);
        break;
      case Gamma:
      case Erlang:
        date = gamrnd_d(parameters.gamma.shape, parameters.gamma.scale, state);
        break;
      case DiscreteUniform:
        date = parameters.discreteUniform.a +
               (int)(curand_uniform_double(state) * (parameters.discreteUniform.b - parameters.discreteUniform.a + 1));
        break;
      case Geometric:
        date = (int)(log(1 - curand_uniform_double(state)) / log(1.0 - parameters.geometric.p));
        break;
      }
      if (precision > 0)
      {
        double factor = pow(10.0, precision);
        date = round(date * factor) / factor;
      }
      return max(date, 0.0);
    }

    // Helper function implementation
    __device__ double gamrnd_d(double shape, double scale, curandState_t *state) const
    {
      if (shape >= 1)
      {
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / sqrt(9.0 * d);

        while (true)
        {
          double z, u, v;
          do
          {
            z = curand_normal_double(state); 
            v = 1.0 + c * z;                
          } while (v <= 0);

          v = v * v * v;                    
          u = curand_uniform_double(state); 

          if (u < 1.0 - 0.0331 * z * z * z * z)
          {
            return d * v * scale;
          }

          if (log(u) < 0.5 * z * z + d * (1.0 - v + log(v)))
          {
            return d * v * scale;
          }
        }
      }
      else
      {
  
        double r = gamrnd_d(shape + 1.0, scale, state);
        double u = curand_uniform_double(state);
        return r * pow(u, 1.0 / shape);
      }
    }


  __device__ bool isValid() const {
      switch (type) {
        case Constant:
          return parameters.constant.value >= 0; 
        case Uniform:
          return parameters.uniform.a < parameters.uniform.b;
        case Exponential:
          return parameters.exp.rate > 0;
        case Normal:
          return parameters.normal.stddev >= 0;
        case Gamma:
        case Erlang:
          return parameters.gamma.shape > 0 && parameters.gamma.scale > 0; 
        case DiscreteUniform:
          return parameters.discreteUniform.a <= parameters.discreteUniform.b;
        case Geometric:
          return parameters.geometric.p > 0 && parameters.geometric.p < 1;
        default:
          return false; 
      }
    }


    __host__ __device__ static Distribution urgent();

    __host__ __device__ static Distribution defaultDistribution();

    __host__ __device__ static Distribution fromParams(int distrib_id, double param1, double param2);

    std::string toXML() const;
  };

  __device__ int getRandomTokenIndex(curandState_t *state, int maxValue)
  {
    // Generate random float in [0,1)
    float rand = curand_uniform(state);
    // Scale to range [0, maxValue)
    return static_cast<int>(rand * maxValue);
  }

} // namespace VerifyTAPN::Cuda::CudaSMC

#endif
