#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <limits>
#include <assert.h>
#include <cmath>
#include "sparse_matrix.cuh"

constexpr double EPSILON = 10e-12;

int main(int argc, char** argv)
{

  std::vector<std::string> input_matrices = {"../data/jgl009.mtx"};
  std::vector<std::string> output_vectors = {"../data/jgl009_id.mtx"};
  for (size_t i = 0; i < input_matrices.size(); ++i)
  {
    std::cout << "Reading: " << input_matrices[i] << std::endl;;
    HostCSRMatrix host_csr(input_matrices[i]);
    std::cout << "Reading: " << output_vectors[i] << std::endl;;
    HostCSRMatrix host_csr_result(output_vectors[i]);
    std::cout << "Read matrices." << std::endl;

    assert(host_csr.num_rows() == host_csr.num_cols());
    assert(host_csr.num_cols() == host_csr_result.num_rows());
    assert(host_csr_result.num_cols() == 1);
    std::cout << "Passed asserts. Rehydrating." << std::endl;
    thrust::host_vector<double> host_result = host_csr_result.rehydrate();
    std::cout << "Rehydrated." << std::endl;
    const size_t n = host_csr.num_rows();

    auto dev_csr = host_to_dev(host_csr);
    
    thrust::device_vector<double> id_vector(n);
    thrust::device_vector<double> out_vector(n);
    thrust::fill(id_vector.begin(), id_vector.end(), 1);

    dev_csr.matvec(id_vector, out_vector);
    for (size_t i = 0; i < n; ++i)
    {
      if (abs(out_vector[i] - host_result[i]) > EPSILON)
      {
        std::cout << i << ": " << host_result[i] << ", " << out_vector[i] << std::endl;
      }
    }
  }
}
