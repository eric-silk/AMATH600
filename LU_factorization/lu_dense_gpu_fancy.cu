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

#include "strided_iterator.cuh"

struct LU
{
  LU() = default;
  std::vector<double> L;
  std::vector<double> U;
  size_t n;
};

// Borrowed from the transformations example code
// https://docs.nvidia.com/cuda/thrust/index.html#transformations
struct daxpy_functor
{
  const double a;
  daxpy_functor(double _a) : a(_a) {}
  __host__ __device__
    double operator()(const double& x, const double&y) const
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

LU LU_factorization(const std::vector<double>& A, size_t n);
void print_matrix(const std::vector<double>& A, size_t n);
double rand_0_1(void);

int main(int argc, char **argv)
{
  int n = 0;
  if (argc >= 2)
  {
    n = atoi(argv[1]);
  }
  else
  {
    n = 1024;
  }

  std::vector<double> A;
  A.resize(n*n);
  std::srand(std::time(nullptr));
  std::generate(A.begin(), A.end(), rand_0_1);

  auto factored = LU_factorization(A, n);

  return 0;
}

double rand_0_1(void)
{
  return ((double) rand() / (RAND_MAX));
}

void print_matrix(const std::vector<double>& A, size_t n)
{
  for (size_t row = 0; row < n; ++row)
  {
    for (size_t col = 0; col < n; ++col)
    {
      std::cout << A[n*row+col] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

LU LU_factorization(const std::vector<double>& A, const size_t n)
{
  assert(A.size() == n*n);
  std::vector<double> U(n*n);
  std::vector<double> L(n*n, 0);

  thrust::host_vector<double>   U_h = A;
  thrust::device_vector<double> U_d = U_h;
  thrust::device_vector<double> Coeffs(n);

  // Let's start with just iterating manually over columns
  // Probably replace this with a counting iterator
  for (size_t col = 0; col < n; ++col)
  {
    // Constant iterator for the current top row
    thrust::constant_iterator<double> numerator(U_d[col*(n+1)]);
    // strided iterator for the coff calcs
    strided_range denominator(U_d.begin()+(n*col)+n, U.end(), n);
    // Coeff iterator
    auto first = thrust::make_zip_iterator(thrust::make_tuple(numerator, denominator.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(numerator, denominator.end()));

    thrust::copy(first, last, std::ostream_iterator<double>(std::cout, "\n"));
  }

  LU retval;
  retval.U = U;
  retval.L = L;

  return retval;
}
