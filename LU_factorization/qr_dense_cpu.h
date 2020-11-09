#ifndef QR_DENSE_CPU_H
#define QR_DENSE_CPU_H

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

enum QRTechnique
{
  Householder,
  Givens
};

template<typename T=double>
struct QRMatrices
{
  QRMatrices(Matrix<T> Q_, Matrix<T> R_)
    : Q(Q_)
    , R(R_)
  {
    // NTD
  }
  QRMatrices(void)
    : Q()
    , R()
  {
    // NTD
  }
  Matrix<T> Q;
  Matrix<T> R;
};

template<typename T=double>
void QR_col_op(const Matrix<T>& A, Matrix<T>& Q, Matrix<T>& R, size_t col_i)
{
  Vector<T> e = Vector<T>::Zero(A.rows()-col_i, 1);
  e(0) = 1;
  auto x = R.block(col_i, col_i, A.rows()-col_i, 1);

  T sign_x = (A(col_i, col_i) >= 0 ? 1 : -1);
  T alpha = x.norm();
  auto u = x - sign_x * alpha * e;
  auto v = u.normalized();
  auto two_vk_tkt = 2*v*v.transpose();
  size_t I_rows = two_vk_tkt.rows();
  size_t I_cols = two_vk_tkt.cols();

  Matrix<T> I = Matrix<T>::Identity(I_rows, I_cols);
  Matrix<T> Q_i = Matrix<T>::Identity(A.rows(), A.cols());
  Q_i.block(col_i, col_i, A.rows()-col_i, A.cols()-col_i) = (I - two_vk_tkt);
  R = Q_i*R;
  Q = Q * Q_i.transpose();
}

template<typename T=double>
QRMatrices<T> QR(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A)
{
  Matrix<T> R = A;
  Matrix<T> Q = Matrix<T>::Identity(A.rows(), A.cols());
  for (size_t col = 0; col < A.cols()-1; ++col)
  {
    QR_col_op(A, Q, R, col);
  }

  QRMatrices<T> retval(Q, R);
  return retval;
}

#endif//QR_DENSE_CPU_H
