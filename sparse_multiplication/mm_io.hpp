#ifndef MM_IO_H
#define MM_IO_H

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <vector>

std::tuple<size_t, size_t, std::vector<std::tuple<size_t, size_t, double>>>
read_mm(std::istream& input_stream) {
  std::string              string_input;
  bool                     file_symmetry = false;
  bool                     pattern = false;
  bool                     binary64 = false;
  std::vector<std::string> header(6);

  // %%MatrixMarket matrix coordinate integer symmetric
  // %%MatrixMarket matrix coordinate real general [binary32, binary64]
  // %%MatrixMarket matrix coordinate pattern symmetric

  std::getline(input_stream, string_input);
  std::stringstream h(string_input);
  for (auto& s : header)
    h >> s;

  if (   header[0] != "%%MatrixMarket"
      || header[1] != "matrix"
      || header[2] != "coordinate") {

    std::cerr << "Unsupported format: " << 
      (header[0] != "%%MatrixMarket") << " " <<
      (header[1] != "matrix") << " " <<
      (header[2] != "coordinate") << std::endl;


    std::cerr << "Unsupported format: " << header[0] << " " << header[1] << " " << header[2] << std::endl;
    throw;
  }

  if (header[3] == "pattern") {
    pattern = true;
  } else if (header[3] == "real" || header[3] == "integer") {
    pattern = false;
  } else {
    std::cerr << "Bad mmio format (value type): " << header[3] << std::endl;
    throw;
  }

  if (header[4] == "symmetric") {
    file_symmetry = true;
  } else if (header[4] == "unsymmetric" || header[4] == "general") {
    file_symmetry = false;
  } else {
    std::cerr << "Bad mmio format (symmetry): " << header[4] << std::endl;
    throw;
  }

  if (header[5] != "") {
    if (header[5] == "binary64") {
      binary64 = true;
    } else {
      throw;
    }
  } 

  while (std::getline(input_stream, string_input)) {
    if (string_input[0] != '%') break;
  }
  size_t n0, n1, nnz;
  std::stringstream(string_input) >> n0 >> n1 >> nnz;

  std::vector<std::tuple<size_t, size_t, double>> aos(0);
  aos.reserve(nnz);


  if (binary64) {
    std::vector<std::tuple<size_t, size_t>> bos(nnz);
    input_stream.read((char*)bos.data(), bos.size()*sizeof(std::tuple<size_t, size_t>));
    for (auto &j : bos) {
      aos.push_back( { std::get<0>(j), std::get<1>(j), 1.0 } );
    }


  } else {

    for (size_t i = 0; i < nnz; ++i) {
      std::string buffer;
      size_t      d0, d1;
      
      std::getline(input_stream, buffer);
      
      double      v = 1.0;
      if (true == pattern) {
	std::stringstream(buffer) >> d0 >> d1;
      } else {
	std::stringstream(buffer) >> d0 >> d1 >> v;
      }
      aos.push_back({ d0 - 1, d1 - 1, v });
      if (file_symmetry == true) {
	aos.push_back({ d1 - 1, d0 - 1, v });
      }
    }
  }

  return std::make_tuple(n0, n1, aos);
}


std::tuple<size_t, size_t, std::vector<std::tuple<size_t, size_t, double>>>
read_mm(std::string filename) {
  std::ifstream input_file(filename, std::ios::binary);
  return read_mm(input_file);
}

#endif//MM_IO_H
