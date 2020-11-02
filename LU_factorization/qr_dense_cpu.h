#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

enum QRTechnique
{
  Householder,
  Givens
};



template<typename T=double>
struct QRMatrices
{
  Eigen::MatrixBase<T> Q;
  Eigen::MatrixBase<T> R;
};

template<typename T=double>
void QR_col_op(Eigen::MatrixBase<T>& A, size_t col_i)
{
  auto e = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(A.rows()-col_i, 1);
  e(0) = 1;
  auto y = A.block(col_i, col_i, A.rows()-col_i, 1);
  T sign_y = (y(0) >= 0 ? 1 : -1);
  auto w = y - sign_y * y.norm() * e;
  auto v = w.normalized();
  // TODO
  auto reflector Eigen::Matrix<T, EguuIdentity
}

template<typename T=double>
QRMatrices<T> QR(Eigen::MatrixBase<T>& a)
{
}

