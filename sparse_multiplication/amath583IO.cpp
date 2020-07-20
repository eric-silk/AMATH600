//
// This file is part of the course materials for AMATH483/583 at the University of Washington,
// Spring 2019
//
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// Author: Andrew Lumsdaine
//
// Gutted and modified by Eric

#include "amath583IO.hpp"
#include "CSRMatrix.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

// ----------------------------------------------------------------
//
// Sparse I/O
//
// ----------------------------------------------------------------
CSRMatrix read_csrmatrix(const std::string& filename) {
  std::tuple<size_t, size_t, std::vector<std::tuple<size_t, size_t, double>>> pack = read_mm(filename);

  size_t M = std::get<0>(pack);
  size_t N = std::get<1>(pack);
  std::vector<std::tuple<size_t, size_t, double>> aos = std::get<2>(pack);

  // sort by row
  std::sort(aos.begin(), aos.end(), [] (auto &a, auto &b) -> bool {
      return (std::get<0>(a) < std::get<0>(b) );
    } );
  
  CSRMatrix A(M, N);
  A.open_for_push_back();

  for (size_t k = 0; k < aos.size(); ++k) {
    size_t i = std::get<0>(aos[k]);
    size_t j = std::get<1>(aos[k]);
    double v = std::get<2>(aos[k]);
    A.push_back(i, j, v);
  }
  A.close_for_push_back();

  return A;
}


void write_csrmatrix(const CSRMatrix& A, std::string filename) {
  std::ofstream output_file(filename);
  std::cout << "%%MatrixMarket matrix coordinate real general" << std::endl;
  std::cout << A.num_rows() << " " << A.num_cols() << " " << A.num_nonzeros() << std::endl;
  A.stream_coordinates(output_file);
}
