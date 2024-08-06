#ifndef NTH_LBM_PARAMS_HPP
#define NTH_LBM_PARAMS_HPP

#include <toml++/toml.hpp>
#include <torch/torch.h>
#include <iostream>
#include <string>

namespace params
{
    template<typename T>
    T try_value(const toml::node_view<const toml::node>& tbl, const std::string name)
    {
        std::optional<T> op = tbl[name].template value<T>();
        if (op.has_value()) return op.value();
        else throw std::runtime_error(name + " not defined in parameters file");
    }

    template<typename T>
    T try_value(const toml::node_view<toml::node>& tbl, const std::string name)
    {
        std::optional<T> op = tbl[name].template value<T>();
        if (op.has_value()) return op.value();
        else throw std::runtime_error(name + " not defined in parameters file");
    }

    struct conversion_factors
    {
        const double C_l;
        const double C_u;
        const double C_t;
        const double C_rho;
        const double Re;
        explicit conversion_factors(const toml::table& tbl);
        double init_C_l(const toml::table& tbl);
        double init_C_u(const toml::table& tbl);
        double init_Re(const toml::table& tbl);
    };
    std::ostream& operator<<(std::ostream& os, const conversion_factors& cf);

    struct domain
    {
        const int R, C;
        const int T, nr_snapshots, snapshot_period;
        const int G;
        const int A;
        const int Gd;
        const int D=2;
        const int Q=9;
        const std::string file_prefix;
        domain(const toml::node_view<toml::node>& tbl);
    };
    std::ostream& operator<<(std::ostream& os, const domain& d);

    struct neutron_transport
    {
        const double avg_emi_n;
        const torch::Tensor tot_cs;
        const torch::Tensor ins_cs;
        const torch::Tensor fis_cs;
        neutron_transport(const toml::node_view<toml::node>& tbl,
                          const conversion_factors& cf,
                          const domain& d);
        torch::Tensor init_tot_cs(const toml::node_view<toml::node> &tbl,
                                  const conversion_factors &cf,
                                  const domain &d);
        torch::Tensor init_ins_cs(const toml::node_view<toml::node>& tbl,
                                  const conversion_factors &cf,
                                  const domain& d);
        torch::Tensor init_fis_cs(const toml::node_view<toml::node>& tbl,
                                  const conversion_factors &cf,
                                  const domain& d);
    };
    std::ostream& operator<<(std::ostream& os, const neutron_transport& nt);

    struct delayed_precursors
    {
        const torch::Tensor ld;
        const torch::Tensor bd;
        delayed_precursors(const toml::node_view<toml::node>& tbl, const conversion_factors& cf, const domain& d);
        torch::Tensor init_ld(const toml::node_view<toml::node>& tbl, const conversion_factors& cf, const domain& d);
        torch::Tensor init_bd(const toml::node_view<toml::node>& tbl, const domain& d);
    };
    std::ostream& operator<<(std::ostream& os, const delayed_precursors& dp);

    struct fluid_flow
    {
        const double u_d;
        const double rho;
        const double nu;
        const double tau_0;
        const double Cs;
        fluid_flow(const toml::node_view<toml::node>& tbl, const conversion_factors& cf);
    };
    std::ostream& operator<<(std::ostream& os, const fluid_flow& d);

    struct heat_transfer
    {
        heat_transfer(const toml::node_view<const toml::node>& tbl, const domain& d);
    };
}

#endif //NTH_LBM_PARAMS_HPP
