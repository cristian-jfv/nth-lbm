#ifndef NTH_LBM_ANGULAR_DIRECTIONS_HPP
#define NTH_LBM_ANGULAR_DIRECTIONS_HPP

#include <torch/torch.h>

class angular_directions
{
public:
    angular_directions(int A, int D);
    torch::Tensor omega;
    double domega;
};


#endif //NTH_LBM_ANGULAR_DIRECTIONS_HPP
