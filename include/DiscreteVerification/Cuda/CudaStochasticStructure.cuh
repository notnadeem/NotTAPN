#ifndef CUDA_STOCHASTIC_STRUCTURE_CUH_
#define CUDA_STOCHASTIC_STRUCTURE_CUH_

// TODO: Check if this is correct

/*#include <sstream>*/

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>

namespace VerifyTAPN::Cuda::SimpleSMC
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

    __device__ double sample(curandState_t* state, const unsigned int precision = 0) const
    {
      double date = 0;
      switch (type)
      {
      case Constant:
        date = parameters.constant.value;
        break;
      case Uniform:
        date = parameters.uniform.a +
               (parameters.uniform.b - parameters.uniform.a) *
                   curand_uniform(state);
        break;
      case Exponential:
        date = -logf(curand_uniform(state)) / parameters.exp.rate;
        break;
      case Normal:
        date = parameters.normal.mean +
               parameters.normal.stddev * curand_normal(state);
        break;
      case Gamma:
      case Erlang:
        date = gamrnd_d(parameters.gamma.shape,
                        parameters.gamma.scale, state);
        break;
      case DiscreteUniform:
        date = parameters.discreteUniform.a +
               (int)(curand_uniform(state) *
                     (parameters.discreteUniform.b -
                      parameters.discreteUniform.a + 1));
        break;
      case Geometric:
        date = (int)(log(curand_uniform(state)) /
                     log(1.0 - parameters.geometric.p));
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
    __device__ double gamrnd_d(double shape, double scale, curandState_t *state)
    {
        if (shape >= 1)
        {
            double d = shape - 1.0 / 3.0;
            double c = 1.0 / sqrt(9.0 * d);
            while (true)
            {
                double z = curand_normal(state);
                double u = curand_uniform(state);
                double v = pow(1.0 + c * z, 3.0);
                if (z > -1.0 / c && log(u) < (z * z / 2.0 + d - d * v + d * log(v)))
                {
                    return d * v * scale;
                }
            }
        }
        else
        {
            double r = gamrnd_d(shape + 1.0, scale, state);
            double u = curand_uniform(state);
            return r * pow(u, 1.0 / shape);
        }
    }

    __global__ void setup(curandState_t *d_states) {};

    __host__ __device__ static Distribution urgent();

    __host__ __device__ static Distribution defaultDistribution();

    __host__ __device__ static Distribution fromParams(int distrib_id, double param1, double param2);

    std::string toXML() const;
  };

} // namespace VerifyTAPN::Cuda::CudaSMC

#endif
