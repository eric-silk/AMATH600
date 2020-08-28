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

struct matvec_functor
{
  // taken from here and modified to be a matrix vector product (r=1):
  // https://stackoverflow.com/a/56070858/8341166
  thrust::device_ptr<double> A, B;
  const size_t rows, cols;

  //  Matrix vector, r = 1
  matvec_functor(thrust::device_ptr<double> _A,
                 thrust::device_ptr<double> _B,
                 const size_t _rows,
                 const size_t _cols)
    : A(_A)
    , B(_B)
    , rows(_rows)
    , cols(_cols)
  {
    // NTD
  };

  __host__ __device__
  double operator()(size_t row)
  {
    double sum = 0.0;
    for (size_t i = 0; i < cols; ++i)
    {
      sum += A[rows*row + i] * B[i];
    }

    return sum;
  }
};

struct matmat_functor
{
  // taken from here and modified as needed
  // https://stackoverflow.com/a/56070858/8341166
  thrust::device_ptr<double> A, B;
  // A rows, A cols, B cols
  const size_t m, n, r;

  matmat_functor(thrust::device_ptr<double> _A,
                 thrust::device_ptr<double> _B,
                 const size_t _m,
                 const size_t _n,
                 const size_t _r)
    : A(_A)
    , B(_B)
    , m(_m)
    , n(_n)
    , r(_r)
  {
    // NTD
  }

  __host__ __device__
  float operator()(size_t C_result_location)
  {
    double sum = 0.0;
    size_t row = C_result_location / r;
    size_t col = C_result_location - (row * r);
    for (size_t i = 0;  i < m; i++)
    {
      //sum += A[col + row*i] * B[col + row*i];
      sum += A[row*n + i] * B[r*i+col];
    }
    return sum;
  }
};

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
