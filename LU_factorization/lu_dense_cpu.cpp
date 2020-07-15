#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void dump_to_csv(std::string fname, const Eigen::Ref<const Eigen::MatrixXd>& mat)
{
  std::ofstream file(fname.c_str());
  file << mat.format(CSVFormat);
  file.close();
}

int main(int argc, char **argv)
{
  int n = 0;
  if (argc >= 2)
  {
    n = atoi(argv[1]);
  }
  else
  {
    n = 10;
  }

  Eigen::Matrix<double,
                Eigen::Dynamic,
                Eigen::Dynamic,
                Eigen::RowMajor> A, L;
  A = Eigen::Matrix<double,
                    Eigen::Dynamic,
                    Eigen::Dynamic,
                    Eigen::RowMajor>::Random(n, n);
  L = Eigen::Matrix<double,
                    Eigen::Dynamic,
                    Eigen::Dynamic,
                    Eigen::RowMajor>::Identity(n, n);
  dump_to_csv(std::string("full.csv"), A);

  // Naive, should use OpenMP or similar to go fastah
  for (int col_i = 0; col_i < A.cols()-1; ++col_i)
  {
    for (int row_i = col_i+1; row_i < A.rows(); ++row_i)
    {
      double coeff = A.coeff(row_i, col_i) / A.coeff(col_i, col_i);
      A.row(row_i) -= coeff*A.row(col_i);
      L.coeffRef(row_i, col_i) += coeff;
    }
  }
  dump_to_csv(std::string("upper.csv"), A);
  dump_to_csv(std::string("lower.csv"), L);
}
