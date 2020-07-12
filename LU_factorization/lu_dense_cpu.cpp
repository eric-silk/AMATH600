#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void dump_to_csv(std::string fname, const Eigen::Matrix<double, 10, 10>& mat)
{
  std::ofstream file(fname.c_str());
  file << mat.format(CSVFormat);
  file.close();
}
void dump_to_csv(std::string fname, const Eigen::Matrix<double, 3, 3>& mat)
{
  std::ofstream file(fname.c_str());
  file << mat.format(CSVFormat);
  file.close();
}

int main(int argc, char **argv)
{
  //Eigen::Matrix<double, 10, 10> A = Eigen::Matrix<double, 10, 10>::Random();
  Eigen::Matrix<double, 3, 3> A; A << 1,2,3,4,5,6,7,8,9;
  Eigen::Matrix<double, 3, 3> L = Eigen::Matrix<double, 3, 3>::Identity();
  dump_to_csv(std::string("full.csv"), A);

  // Naive
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
