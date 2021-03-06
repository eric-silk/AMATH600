#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <limits>
#include <assert.h>
#include <cmath>
#include <iostream>
#include "sparse_matrix.cuh"

//constexpr double EPSILON = 10e-12;

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
    auto host_result = host_csr_result.rehydrate();

    std::cout << "Rehydrated." << std::endl;
    const size_t n = host_csr_result.num_rows();
    const size_t m = host_csr_result.num_cols();
    for (size_t row = 0; row < n; ++row) {
      for (size_t col = 0; col < m; ++col) {
        std::cout << host_result[row*m + col] << " ";
      }
      std::cout << std::endl;
    }

    auto dev_csr = host_to_dev(host_csr);
    
    thrust::device_vector<double> id_vector(n);
    thrust::device_vector<double> out_vector(n);
    thrust::fill(id_vector.begin(), id_vector.end(), 1);
    thrust::fill(out_vector.begin(), out_vector.end(), 0);

    dev_csr.matvec(id_vector, out_vector);
    std::cout << "n: " << n << std::endl;
    std::cout << "i, correct, rslt" << std::endl;
    for (size_t i = 0; i < n; ++i)
    {
      std::cout << i << ": " << host_result[i] << ", " << out_vector[i] << std::endl;
    }
  }

  std::cout << "Now doing MatMat..." << std::endl;
  HostCSRMatrix input_csr("../data/jgl009.mtx");
  HostCSRMatrix correct_output_csr("../data/jgl009_squared.mtx");
  auto dev_csr = host_to_dev(input_csr);
  DeviceCSRMatrix outloc = dev_csr.matmat(dev_csr);
  auto result = outloc.rehydrate();
  for (size_t row = 0; row < outloc.num_rows(); ++row)
  {
    for (size_t col = 0; col < outloc.num_cols(); ++col)
    {
      std::cout << result[row*outloc.num_cols() + col] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "==================" << std::endl;
  auto real_result = correct_output_csr.rehydrate();
  for (size_t row = 0; row < outloc.num_rows(); ++row)
  {
    for (size_t col = 0; col < outloc.num_cols(); ++col)
    {
      std::cout << real_result[row*correct_output_csr.num_cols() + col] << " ";
    }
    std::cout << std::endl;
  }
}
