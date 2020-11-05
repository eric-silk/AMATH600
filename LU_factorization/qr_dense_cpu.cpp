#include <iostream>
#include <Eigen/Dense>
#include "qr_dense_cpu.h"

bool wiki_example(void);

int main(int argc, char *argv[])
{
  std::cout << "Starting." << std::endl;
  bool rslt = wiki_example();
  std::cout << "Wiki worked: " << rslt << std::endl;

  return 0;
}

bool wiki_example(void)
{
  // https://en.wikipedia.org/wiki/QR_decomposition#Example_2
  Eigen::MatrixXd A = (Eigen::Matrix3d() << 12, -51, 4, 6, 167, -68, -4, 24, -41).finished();
  std::cout << "A:" << std::endl << A << std::endl;
  auto qr_result = QR(A);
  std::cout << "Result: " << std::endl << qr_result.Q << std::endl;
  return true;
}
