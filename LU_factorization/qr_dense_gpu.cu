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

// http://www.seas.ucla.edu/~vandenbe/133A/lectures/qr.pdf
// https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf

struct QR
{
  QR
  std::vector<double> Q;
  std::vector<double> R;
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

QR QR(const std::vector<double>& A, size_t n);
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

  auto factored = QR_factorization(A, n);

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

// I've opted to work with CSR's for now; CSC's may be more well suited
// but bleh
std::vector<double> get_column_of_row_major_cpu(std::vector<double>& A, const size_t n, const size_t colnumber)
{
  std::vector<double> column(n);
  for (size_t i = 0; i < n; ++i)
  {
    column[n] = A[i*n+colnumber];
  }

  return column;
}

thrust::device_vector<double> get_column_of_row_major_gpu(thrust::device_vector<double> matrix,
                                                          const size_t n,
                                                          const size_t colnumber)
{
  // Gather up the nth vector
  thrust::device_vector<size_t> indices(n), col_vector(n);
  // To generate the indices, just to a counting vector from 0 to n, then add colnumber
  thrust::sequence(indices.begin(), indices.end());
  thrust::constant_iterator<size_t> colnumber_iter(colnumber);
  thrust::transform(indices.begin(),
                    indices.end(),
                    colnumber_iter.begin(),
                    indices.begin(),
                    thrust::add<size_t>);

  // Now gather them
  thrust::gather(indices.begin(), indices.end(), matrix.begin() col_vector.begin());

  return col_vector;
}

QR QR_factorization_GM(const std::vector<double>& A, const size_t n)
{
  assert(A.size() == n*n);
  std::vector<double> Q(n*n);
  std::vector<double> R(n*n, 0);

}

QR QR_factorization_GM_modified(const std::vector<double>& A, const size_t n)
{
  assert(A.size() == n*n);
  std::vector<double> Q(n*n);
  std::vector<double> R(n*n, 0);

}

QR QR_factorization_Householder(const std::vector<double>& A, const size_t n)
{
  assert(A.size() == n*n);
  std::vector<double> Q(n*n);
  std::vector<double> R(n*n, 0);

}
