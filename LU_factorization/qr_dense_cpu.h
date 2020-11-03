#ifndef QR_DENSE_CPU_H
#define QR_DENSE_CPU_H

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
  auto two_vk_tkt = 2*v*v.transpose();
  size_t I_rows = two_vk_tkt.rows();
  size_t I_cols = two_vk_tkt.cols();

  auto I = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(I_rows, I_cols);
  auto new_y = y * (I - two_vk_tkt);

  y = new_y;
}

template<typename T=double>
QRMatrices<T> QR(const Eigen::MatrixBase<T>& A)
// TODO should A be modified in place?
{
  for (size_t col = 0; col < A.cols(); ++col)
  {
    QR_col_op(A, col);
  }

  QRMatrices<T> retval;
  retval.Q = A;
}

#endif//QR_DENSE_CPU_H
