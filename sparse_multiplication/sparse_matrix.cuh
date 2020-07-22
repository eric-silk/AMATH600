#ifndef SPARSE_MATRIX_CUH
#define SPARSE_MATRIX_CUH
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <assert.h>

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

    HostCSRMatrix(size_t rows,
                  size_t cols,
                  thrust::host_vector<size_t> row_indices,
                  thrust::host_vector<size_t> col_indices,
                  thrust::host_vector<double> storage)
      : m_is_open(false)
      , m_num_rows(rows)
      , m_num_cols(rows)
      , m_row_indices(row_indices)
      , m_col_indices(col_indices)
      , m_storage(storage)
    {
      // NTD
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

    thrust::host_vector<size_t> get_row_indices(void) const { return m_row_indices; };
    thrust::host_vector<size_t> get_col_indices(void) const { return m_col_indices; };
    thrust::host_vector<double> get_storage(void) const { return m_storage; };

  private:
    bool m_is_open;
    size_t m_num_rows, m_num_cols;
    thrust::host_vector<size_t> m_row_indices, m_col_indices;
    thrust::host_vector<double> m_storage;
};

class DeviceCSRMatrix
{
  public:
    DeviceCSRMatrix(size_t rows, size_t cols)
      : m_is_open(false)
      , m_num_rows(rows)
      , m_num_cols(cols)
      , m_row_indices(m_num_rows + 1, 0)
    {
      // NTD
    }

    DeviceCSRMatrix(size_t rows,
                    size_t cols,
                    thrust::device_vector<size_t> row_indices,
                    thrust::device_vector<size_t> col_indices,
                    thrust::device_vector<double> storage)
      : m_is_open(false)
      , m_num_rows(rows)
      , m_num_cols(cols)
      , m_row_indices(row_indices)
      , m_col_indices(col_indices)
      , m_storage(storage)
    {
      // NTD
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

    thrust::device_vector<size_t> get_row_indices(void) const { return m_row_indices; };
    thrust::device_vector<size_t> get_col_indices(void) const { return m_col_indices; };
    thrust::device_vector<double> get_storage(void) const { return m_storage; };

  private:
    bool m_is_open;
    size_t m_num_rows, m_num_cols;
    thrust::device_vector<size_t> m_row_indices, m_col_indices;
    thrust::device_vector<double> m_storage;
};

// Copy constructors produced a circular dependece I MIGHT be able to get around
// using a base class and inheritance, but meh.
// Templates proved obnoxious.
// We're going C style, baby.
HostCSRMatrix dev_to_host(const DeviceCSRMatrix& dev)
{
  return HostCSRMatrix(dev.num_rows(),
                       dev.num_cols(),
                       dev.get_row_indices(),
                       dev.get_col_indices(),
                       dev.get_storage());
}

DeviceCSRMatrix host_to_dev(const HostCSRMatrix& host)
{
  return DeviceCSRMatrix(host.num_rows(),
                         host.num_cols(),
                         host.get_row_indices(),
                         host.get_col_indices(),
                         host.get_storage());
}

#endif//HOST_SPARSE_MATRIX_CUH
