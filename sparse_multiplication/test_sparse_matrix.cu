#include "sparse_matrix.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <limits>

int main(int argc, char** argv)
{
  size_t n;
  if (argc >= 2)
  {
    n = atoi(argv[1]);
  }
  else
  {
    n = 6;
  }

  HostCSRMatrix host_csr(n, n);
  
  // Generate N random numbers and n i,j coords
  std::random_device rd;
  std::mt19937 gen(rd());
  // The distribution is inclusive, hence the n-1
  std::uniform_int_distribution<size_t> coord_distr(0, n-1);
  const double double_min = std::numeric_limits<double>::min();
  const double double_max = std::numeric_limits<double>::max();
  std::uniform_real_distribution<double> val_distr(double_min, double_max);

  host_csr.open_for_pushback();
  for (size_t i = 0; i < n; ++i)
  {
    size_t row = coord_distr(gen);
    size_t col = coord_distr(gen);
    double value = val_distr(gen);
    host_csr.push_back(row, col, value);
  }
  host_csr.close_for_pushback();

  auto dev_csr = host_to_dev(host_csr);
}
