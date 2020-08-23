#ifndef FUNCTORS_CUH
#define FUNCTORS_CUH
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

// This isn't actually working as expected, I think :(
struct mac_functor
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    // Y = Y + A*B
    thrust::get<2>(t) += thrust::get<0>(t) + thrust::get<1>(t);
  }
};

// y += a*b
template <typename T>
void mac(thrust::device_vector<T>& y, const thrust::device_vector<T>& a, const thrust::device_vector<T>& b)
{
  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin(), y.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end(), y.end())),
                   mac_functor());
}
#endif//FUNCTORS_CUH
