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

// This feels clumsy. But, hopefully will work for now.
struct matvec_functor
{
  // TODO consider references/ptrs rather than constructed copies?
  const thrust::device_vector<size_t>& m_col_indices;
  const thrust::device_vector<size_t>& m_row_indices;
  thrust::device_vector<size_t> map;
  const thrust::device_vector<double>& m_storage;
  const size_t num_cols;

  const thrust::device_vector<double>& x;

  matvec_functor(const thrust::device_vector<size_t>& col_indices,
                 const thrust::device_vector<size_t>& row_indices,
                 const thrust::device_vector<double>& storage,
                 const size_t num_cols,
                 const thrust::device_vector<double>& x)
    : m_col_indices(col_indices)
    , m_row_indices(row_indices)
    , m_storage(storage)
    , num_cols(num_cols)
    , x(x)
  {
    // NTD
  }

  __device__
  double operator()(size_t row_i)
  {
    thrust::device_vector<double> tmp_vector(num_cols, 0);
    // TODO thrust::sequence or thrust::remove_copy_if?
    thrust::copy(m_col_indices.begin() + m_row_indices[row_i],
                 m_col_indices.begin() + m_row_indices[row_i+1],
                 map.begin());
    // A corresponding gather shouldn't be needed for this
    thrust::scatter(m_storage.begin() + m_row_indices[row_i],
                    m_storage.begin() + m_row_indices[row_i+1],
                    map.begin(),
                    tmp_vector.begin());
    return thrust::inner_product(tmp_vector.begin(), tmp_vector.end(), x.begin(), 0);
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
