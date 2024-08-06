#ifndef NTH_LBM_NEUTRON_TRANSPORT_HPP
#define NTH_LBM_NEUTRON_TRANSPORT_HPP

#include <torch/torch.h>
#include "angular_directions.hpp"
#include "params.hpp"

class neutron_transport
{
public:
    neutron_transport(const params::neutron_transport& nt,
                      const params::domain& d,
                      double b_tot);
    torch::Tensor snf; // Scalar Neutron Flux
    torch::Tensor q; // heat source due to neutron fission (q" in papers)
    const double ikeff;
    const torch::Tensor fis_cs; // fission cross section
    void step(const torch::Tensor& Cd, const torch::Tensor& ld);
private:
    const double tot_beta; // tot_beta is 1.0 - beta_tot in papers
    const angular_directions ad;
    torch::Tensor anf; // Angular Neutron Flux
    // Distribution functions
    torch::Tensor ef;  // equilibrium
    torch::Tensor af;  // advection
    torch::Tensor cf;  // collision
    torch::Tensor S;   // source term
    const torch::Tensor xi; // constant term for eq. d.f.
    const torch::Tensor eps; // constant term for source d.f.
    // Cross sections
    const torch::Tensor ins_cs; // inscattering
    const torch::Tensor tot_cs; // total
    // Spectra
    const torch::Tensor pro_chi; // Prompt neutrons
    const torch::Tensor del_chi; // Delayed neutrons
    torch::Tensor init_xi(int A, int Q, const angular_directions& _ad);
    torch::Tensor init_eps(int A, int Q, const angular_directions& _ad);
    torch::Tensor init_pro_chi(int G);
    torch::Tensor init_del_chi(int G, int Gd);
    // Time step functions
    void eval_equilibrium();
    void eval_sources(const torch::Tensor& Cd, const torch::Tensor& ld);
    void collide();
    void advect();
    void boundary_condition();
};

#endif //NTH_LBM_NEUTRON_TRANSPORT_HPP
