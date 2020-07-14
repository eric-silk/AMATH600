#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <ctime>

int main(void)
{
  // generate 32M random numbers serially
  thrust::host_vector<int> h_vec(32 << 20);
  std::srand(std::time(nullptr));
  std::generate(h_vec.begin(), h_vec.end(), rand);
  std::vector<int> local_vec;
  local_vec.resize(32 << 20);
  std::copy(h_vec.begin(), h_vec.end(), local_vec.begin());
  std::cout << "First:" << local_vec[0] << std::endl;
  std::cout << "Last:" << local_vec.back() << std::endl;

  // transfer data to the device
  thrust::device_vector<int> d_vec = h_vec;

  // sort data on the device (846M keys per second on GeForce GTX 480)
  thrust::sort(d_vec.begin(), d_vec.end());

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  std::copy(h_vec.begin(), h_vec.end(), local_vec.begin());
  std::cout << "First:" << local_vec[0] << std::endl;
  std::cout << "Last:" << local_vec.back() << std::endl;

  return 0;
}
