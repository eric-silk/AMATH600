#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <assert.h>

// Borrowed from the transformations example code
// https://docs.nvidia.com/cuda/thrust/index.html#transformations
struct daxpy_functor
{
  const double a;
  daxpy_functor(double _a) : a(_a) {}
  __host__ __device__
    double operator()(const double& x, const double& y) const
    {
      return a * x + y;
    }
};

void daxpy(double A, thrust::device_vector<double>& X, thrust::device_vector<double>& Y)
{
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), daxpy_functor(A));
}

struct to_zero_functor
{
  const double epsilon;
  to_zero_functor(double _epsilon) : epsilon(_epsilon) {}
  __host__ __device__
    double operator()(const double& x) const
    {
      if (std::abs(x) <= epsilon)
        return 0;
      else
        return x;
    }
};

void to_zero(double epsilon, thrust::device_vector<double>& X)
{
  thrust::transform(X.begin(), X.end(), X.begin(), to_zero_functor(epsilon));
}

