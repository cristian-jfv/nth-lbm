#include "d2q9.hpp"

namespace d2q9
{
    const torch::Tensor E = torch::tensor(
            {4.0/ 9.0,
             1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0, 1.0/ 9.0,
             1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0},
            torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));

    const torch::Tensor W = torch::tensor(
            {{0.0, 1.0, 0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0},
             {0.0, 0.0, 1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0}},
            torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA));
}