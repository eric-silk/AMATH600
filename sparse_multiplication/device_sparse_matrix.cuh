#ifndef DEVICE_SPARSE_MATRIX_CUH
#define DEVICE_SPARSE_MATRIX_CUH
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <assert.h>

#include "host_sparse_matrix.cuh"

class HostCSRMatrix
{
  public:
    HostCSRMatrix(size_t rows, size_t cols)
      : m_is_open(false)
      , m_num_rows(rows)
      , m_num_cols(cols)
      , m_row_indices(m_num_rows + 1, 0)
    {
      // NTD
    }

    HostCSRMatrix& operator=(const HostCSRMatrix& other)
    {
      if (&other == this)
      {
        return *this;
      }
      this->m_is_open = other.m_is_open;
      this->m_num_rows = other.m_num_rows;
      this->m_num_cols = other.m_num_cols;
      this->m_row_indices = other.m_row_indices;
      this->m_col_indices = other.m_col_indices;
      this->m_storage = other.m_storage;

      return *this;
    }

    HostCSRMatrix& operator=(const DeviceCSRMatrix& other)
    {
      if (&other == this)
      {
        return *this;
      }
      this->m_is_open = other->m_is_open;
      this->m_num_rows = other->m_num_rows;
      this->m_num_cols = other->m_num_cols;
      this->m_row_indices = other->m_row_indices;
      this->m_col_indices = other->m_col_indices;
      this->m_storage = other->m_storage;

      return *this;
    }

    void open_for_pushback(void)
    {
      m_is_open = true;
    }

    void close_for_pushback(void)
    {
      // TODO Thrust-ize these ops?
      m_is_open = false;
      for (size_t i = 0; i < m_num_rows; ++i)
      {
        m_row_indices[i+1] += m_row_indices[i];
      }
      for (size_t i = m_num_rows; i > 0; --i)
      {
        m_row_indices[i] = m_row_indices[i-1];
      }
      m_row_indices[0] = 0;
    }
    
    void push_back(size_t row, size_t col, double value)
    {
       assert(m_is_open);
       assert(row < m_num_rows);
       assert(col < m_num_rows);

       // TODO thrust-ize?
       ++m_row_indices[row];
       m_col_indices.push_back(col);
       m_storage.push_back(value);
    }

    void clear(void)
    {
      m_col_indices.clear();
      m_storage.clear();
      thrust::fill(m_row_indices.begin(), m_row_indices.end(), 0);
    }

    size_t num_rows(void) const { return m_num_rows; };
    size_t num_cols(void) const { return m_num_cols; };
    size_t num_nonzeros(void) const { return m_storage.size(); };

  private:
    friend class DeviceCSRMatrix;
    bool m_is_open;
    size_t m_num_rows, m_num_cols;
    thrust::host_vector<size_t> m_row_indices, m_col_indices;
    thrust::host_vector<double> m_storage;
};
#endif//DEVICE_SPARSE_MATRIX_CUH
