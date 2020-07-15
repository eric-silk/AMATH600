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
  // Actually do a linspace for now
  for (size_t i = 1; i < (n*n)+1; ++i)
  {
    A[i-1] = i;
  }
  std::cout << "A:" << std::endl;
  print_matrix(A, n);

  auto factored = LU_factorization(A, n);

  std::cout << "U:" << std::endl;
  print_matrix(factored.U, n);
  std::cout << "L:" << std::endl;
  print_matrix(factored.L, n);

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

  // Initialize them
  std::copy(A.begin(), A.end(), U.begin());
  for (size_t i = 0; i < n; ++i)
  {
    L[i*(n+1)] = 1.0;
  }

  thrust::host_vector<double> top_row_host(n);
  thrust::device_vector<double> top_row_dev(n);
  thrust::host_vector<double> reducing_row_host(n);
  thrust::device_vector<double> reducing_row_dev(n);

  for (size_t col = 0; col < n-1; ++col)
  {
    std::copy(U.begin()+(col*n), U.begin()+((col+1)*n), top_row_host.begin());
    top_row_dev = top_row_host;
    std::cout << "(num, den): ";
    for (int row = col+1; row < n; ++row)
    {
      size_t num_coeff = row*n+col;
      size_t den_coeff = col*n+col;
      double coeff = -(U[num_coeff] / U[den_coeff]);
      std::cout << "(" << A[num_coeff] << ", " << A[den_coeff] << ") ";

      // Copy the Rows to the host vector, then device vector
      size_t start_loc = row*n+col;
      size_t end_loc = (row+1)*n;
      std::copy(U.begin()+(start_loc), U.begin()+(end_loc), reducing_row_host.begin());
      reducing_row_dev = reducing_row_host;
      // Scale and add
      daxpy(coeff, top_row_dev, reducing_row_dev);
      reducing_row_host = reducing_row_dev;

      thrust::copy(reducing_row_host.begin(),
                   reducing_row_host.end()-col,
                   U.begin()+start_loc);
    }
    std::cout << std::endl << "Col: " << col << std::endl;
    print_matrix(U, n);
  }

  LU retval;
  retval.U = U;
  retval.L = L;

  return retval;
}
