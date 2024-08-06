#ifndef NTH_LBM_DELAYED_PRECURSORS_HPP
#define NTH_LBM_DELAYED_PRECURSORS_HPP

#include <torch/torch.h>
#include "neutron_transport.hpp"
#include "fluid_flow.hpp"
#include "angular_directions.hpp"
#include "params.hpp"

class delayed_precursors
{
public:
    delayed_precursors(const params::delayed_precursors& dp, const params::domain& d);
    torch::Tensor Cd;
    const torch::Tensor ld; // lambda d in papers
    void step(const neutron_transport& nt, const fluid_flow& ff);

private:
    const angular_directions ad;
    const torch::Tensor del_beta;
    torch::Tensor ef;
    torch::Tensor af;
    torch::Tensor cf;
    torch::Tensor S;
    void eval_equilibrium(const fluid_flow& ff);
    void eval_sources(const neutron_transport& nt, const fluid_flow& ff);
    void collide();
    void advect();
    void boundary_condition();

};


#endif //NTH_LBM_DELAYED_PRECURSORS_HPP
