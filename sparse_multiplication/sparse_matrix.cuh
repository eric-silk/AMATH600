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

#include "mm_io.hpp"
#include "functors.cuh"

// I wasn't able to get templated types playing nicely. This would be an obvious improvement, I think.
class HostCSRMatrix
{
  public:
    HostCSRMatrix()
      : m_is_open(false)
      , m_num_rows(0)
      , m_num_cols(0)
      , m_row_indices()
    {
      // NTD
    }

    HostCSRMatrix(const std::string& matrix_name)
    {
      this->read_csrmatrix(matrix_name);
    }

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
      // TODO validate this and make it better
      m_col_indices.clear();
      m_col_indices.resize(0);
      m_storage.clear();
      m_col_indices.resize(0);
      thrust::fill(m_row_indices.begin(), m_row_indices.end(), 0);
    }

    // Courtesy of Dr. Lumsdaine from AMATH583 (PS6, specifically!)
    void read_csrmatrix(const std::string& filename)
    {
      std::tuple<size_t, size_t, std::vector<std::tuple<size_t, size_t, double>>> pack = read_mm(filename);

      size_t M = std::get<0>(pack);
      size_t N = std::get<1>(pack);
      std::vector<std::tuple<size_t, size_t, double>> aos = std::get<2>(pack);

      // sort by row
      std::sort(aos.begin(), aos.end(), [] (auto &a, auto &b) -> bool {
          return (std::get<0>(a) < std::get<0>(b) );
        } );
      
      this->open_for_pushback();
      m_num_rows = M;
      m_num_cols = N;
      m_row_indices.resize(m_num_rows + 1);
      thrust::fill(m_row_indices.begin(), m_row_indices.end(), 0);

      for (size_t k = 0; k < aos.size(); ++k) {
        size_t i = std::get<0>(aos[k]);
        size_t j = std::get<1>(aos[k]);
        double v = std::get<2>(aos[k]);
        this->push_back(i, j, v);
      }

      this->close_for_pushback();
    }

    size_t num_rows(void) const { return m_num_rows; };
    size_t num_cols(void) const { return m_num_cols; };
    size_t num_nonzeros(void) const { return m_storage.size(); };

    thrust::host_vector<double> rehydrate(void)
    {
      thrust::host_vector<double> dense(m_num_rows * m_num_cols);
      thrust::fill(dense.begin(), dense.end(), 0);
      for (size_t row = 0; row < m_row_indices.size()-1; ++row)
      {
        size_t colid_start = m_row_indices[row];
        size_t colid_end = m_row_indices[row+1];
        for (size_t colid = colid_start; colid < colid_end; ++colid)
        {
          // TODO Segfault here, fix
          size_t col = m_col_indices[colid];
          double value = m_storage[colid];
          assert((row*m_num_cols + col) < dense.size());
          assert(col < m_storage.size());
          dense[row * m_num_cols + col] = m_storage[colid];
        }
      }

      return dense;
    }

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

    void matvec(const thrust::device_vector<double>& x, thrust::device_vector<double>& y) const
    {
      assert(m_num_rows == y.size());
      const thrust::host_vector<size_t> col_indices = m_col_indices;
      const thrust::host_vector<size_t> row_indices = m_row_indices;
      thrust::fill(y.begin(), y.end(), 0);
      thrust::host_vector<double> h_tmp_vector(m_num_cols), h_map(m_num_cols);
      thrust::device_vector<double> tmp_vector(m_num_cols), map(m_num_cols);
      // The mapping vector for rehydrating the CSR representation
      for (size_t i = 0; i < m_num_rows; ++i)
      {
        thrust::fill(map.begin(), map.end(), 0);
        thrust::fill(tmp_vector.begin(), tmp_vector.end(), 0);
        // Create the map, sloppily
        for(size_t j = row_indices[i]; j < row_indices[j+1]; ++j)
        {
          assert(col_indices[j] < map.size());
          h_map[col_indices[j]] = 1;
        }
        map = h_map;
        // A corresponding reduce shouldn't be needed for this
        thrust::scatter(x.begin()+row_indices[i],
                        x.begin()+row_indices[i+1],
                        map.begin(),
                        tmp_vector.begin());
        mac(y, tmp_vector, x);
      }
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
