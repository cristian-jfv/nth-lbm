#ifndef NTH_LBM_D2Q9_HPP
#define NTH_LBM_D2Q9_HPP

#include <torch/torch.h>

namespace d2q9
{
    using torch::Tensor;
    // Lattice velocities
    extern const Tensor E;
    // Lattice weights
    extern const Tensor W;
}

#endif //NTH_LBM_D2Q9_HPP
