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
#include "functors_lu.cuh"

struct LU
{
  LU() = default;
  std::vector<double> L;
  std::vector<double> U;
  size_t n;
};

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
  if (n < 20)
  {
    std::cout << "A:" << std::endl;
    print_matrix(A, n);
    std::cout << "U:" << std::endl;
    print_matrix(factored.U, n);
    std::cout << "L:" << std::endl;
    print_matrix(factored.L, n);
  }

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

void print_device_matrix(const thrust::device_vector<double>& A, const size_t n)
{
  std::cout << "Device matrix:" << std::endl;
  for (size_t row = 0; row < n; ++row)
  {
    for (size_t col = 0; col < n; ++col)
    {
      std::cout << A[n*row+col] << " ";
    }
    std::cout << std::endl;
  }
}


LU LU_factorization(const std::vector<double>& A, const size_t n)
{
  typedef thrust::device_vector<double>::iterator Iterator;
  assert(A.size() == n*n);

  thrust::host_vector<double>   U_h = A;
  thrust::device_vector<double> U_d = U_h;
  thrust::host_vector<double>   L_h(n*n);
  thrust::device_vector<double> L_d(n*n);
  thrust::device_vector<double> Coeffs(n-1);


  for (size_t col = 0; col < n-1; ++col)
  {
    // Constant iterator for the current top row
    thrust::constant_iterator<double> denominator(U_d[col*n+col]);
    // strided iterator for the coff calcs
    strided_range<Iterator> numerator(U_d.begin()+(n*col)+col+n, U_d.end(), n);
    // Coeff iterator
    thrust::transform(numerator.begin(),
                      numerator.end(),
                      denominator,
                      Coeffs.begin()+col,
                      thrust::divides<double>());
    strided_range<Iterator> lower_column(L_d.begin()+(n*col)+col+n, L_d.end(), n);
    thrust::copy(Coeffs.begin()+col, Coeffs.end(), lower_column.begin());

    const size_t top_start_offset = col*n+col;
    const size_t top_end_offset   = top_start_offset + n - col;
    // It seems like this could be fused with some magic iterators such that
    // the row reductions are done in parallel
    for (size_t row = col+1; row < n; ++row)
    {
      const size_t bot_start_offset = row*n+col;
      const double local_coeff = Coeffs[row-1];
      thrust::transform(U_d.begin()+top_start_offset,
                        U_d.begin()+top_end_offset,
                        U_d.begin()+bot_start_offset,
                        U_d.begin()+bot_start_offset,
                        daxpy_functor(-Coeffs[row-1]));
    }
  }

  to_zero(1e-12, U_d);

  U_h = U_d;
  L_h = L_d;
  for (size_t i = 0; i < n; ++i)
  {
    L_h[n*i+i] = 1;
  }
  LU retval;
  retval.U.resize(n*n);
  retval.L.resize(n*n);
  thrust::copy(U_h.begin(), U_h.end(),  retval.U.begin());
  thrust::copy(L_h.begin(), L_h.end(),  retval.L.begin());

  return retval;
}
