#ifndef NTH_LBM_FLUID_FLOW_HPP
#define NTH_LBM_FLUID_FLOW_HPP

#include <torch/torch.h>

class fluid_flow
{
public:
    fluid_flow();
    torch::Tensor u;
    torch::Tensor rho;
};


#endif //NTH_LBM_FLUID_FLOW_HPP
