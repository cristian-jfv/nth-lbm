#ifndef NTH_LBM_D2Q9_HPP
#define NTH_LBM_D2Q9_HPP

#include <torch/torch.h>

namespace d2q9
{
    // Lattice velocities
    extern const torch::Tensor E;
    // Lattice weights
    extern const torch::Tensor W;
    const double ics2 = 3.0;
    const double cs2 = 1.0/3.0;
    const double ics4 = 9.0;
    const double cs4 = 1.0/9.0;
}

#endif //NTH_LBM_D2Q9_HPP
