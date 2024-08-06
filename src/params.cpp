#include "params.hpp"

params::conversion_factors::conversion_factors(const toml::table& tbl):
C_l(init_C_l(tbl)),
C_u(init_C_u(tbl)),
C_t(C_l/C_u),
C_rho(try_value<double>(tbl["fluid-flow"], "rho")),
Re(init_Re(tbl))
{}

std::ostream& params::operator<<(std::ostream& os, const conversion_factors& cf)
{
    return os << "conversion factors:\n"
              << "C_l=" << cf.C_l
              << "\nC_u=" << cf.C_u
              << "\nC_t=" << cf.C_t
              << "\nC_rho=" << cf.C_rho
              << "\nRe=" << cf.Re
              << std::endl;
}

double params::conversion_factors::init_C_l(const toml::table& tbl)
{
    double L = try_value<double>(tbl["domain"], "characteristic_length");
    double hat_L = try_value<double>(tbl["domain"], "columns");
    return L/hat_L;
}

double params::conversion_factors::init_C_u(const toml::table& tbl)
{
    double u = try_value<double>(tbl["fluid-flow"], "u");
    double hat_u = try_value<double>(tbl["domain"], "lattice_speed");
    return u/hat_u;
}

double params::conversion_factors::init_Re(const toml::table& tbl)
{
    double L = try_value<double>(tbl["domain"], "characteristic_length");
    double u = try_value<double>(tbl["fluid-flow"], "u");
    double nu_0 = try_value<double>(tbl["fluid-flow"], "kinematic_viscosity");
    return L*u/nu_0;
}

params::domain::domain(const toml::node_view<toml::node>& tbl):
R(try_value<int>(tbl, "rows")),
C(try_value<int>(tbl, "columns")),
T(try_value<int>(tbl, "time_steps")),
nr_snapshots(try_value<int>(tbl, "nr_snapshots")),
snapshot_period((int)(T/nr_snapshots)),
G(try_value<int>(tbl, "G")),
A(try_value<int>(tbl, "A")),
Gd(try_value<int>(tbl, "Gd")),
file_prefix(try_value<std::string>(tbl, "file_prefix"))
{}

std::ostream& params::operator<<(std::ostream& os, const domain& d)
{
    return os << "domain parameters:\n"
              << "R=" << d.R
              << "\nC=" << d.C
              << "\nT=" << d.T
              << "\nnr_snapshots=" << d.nr_snapshots
              << "\nsnapshot_period=" << d.snapshot_period
              << "\nG=" << d.G << " (nr. energy groups)"
              << "\nA=" << d.A << " (nr. angular directions)"
              << "\nGd=" << d.Gd << " (nr. delayed precursors families)"
              << "\nfile_prefix=" << d.file_prefix
              << std::endl;
}

params::neutron_transport::neutron_transport
(
        const toml::node_view<toml::node>& tbl,
        const conversion_factors& cf,
        const domain& d
):
avg_emi_n(try_value<double>(tbl, "avg_emi_n")),
tot_cs(init_tot_cs(tbl, cf, d)),
ins_cs(init_ins_cs(tbl, cf, d)),
fis_cs(init_fis_cs(tbl, cf, d))
{}

std::ostream& params::operator<<(std::ostream& os, const neutron_transport& nt)
{
    return os << "neutron transport parameters:\n"
              << "avg_emi_n=" << nt.avg_emi_n
              << "\ntot_cs=\n" << nt.tot_cs
              << "\nins_cs=\n" << nt.ins_cs
              << "\nfis_cs=\n" << nt.fis_cs
              << std::endl;
}

torch::Tensor
params::neutron_transport::init_tot_cs
(
        const toml::node_view<toml::node> &tbl,
        const conversion_factors &cf,
        const domain &d
)
{
    torch::Tensor _tot_cs = torch::zeros({d.G}, torch::kCUDA);
    const double f = 100.0*cf.C_l;
    for (int i = 0; i < d.G; ++i)
    {
        // The parameter file has values in cm-1
        _tot_cs[i] = f*try_value<double>(tbl, "t"+std::to_string(i+1));
    }
    return _tot_cs;
}

torch::Tensor
params::neutron_transport::init_ins_cs
(
        const toml::node_view<toml::node> &tbl,
        const conversion_factors &cf,
        const params::domain &d
)
{
    torch::Tensor _ins_cs = torch::zeros({d.G, d.G}, torch::kCUDA);
    const double f = 100.0*cf.C_l;
    for (int r = 0; r < d.G; ++r)
    {
        for (int c = 0; c < d.G; ++c)
        {
            std::string name{"s"+std::to_string(c+1)+std::to_string(r+1)};
            _ins_cs[r][c] = f*try_value<double>(tbl,name);
        }
    }
    return _ins_cs;
}

torch::Tensor
params::neutron_transport::init_fis_cs
(
        const toml::node_view<toml::node> &tbl,
        const conversion_factors &cf,
        const params::domain &d
)
{
    torch::Tensor _fis_cs = torch::zeros({d.G}, torch::kCUDA);
    const double f = 100.0*cf.C_l;
    for (int i = 0; i < d.G; ++i)
    {
        std::string name{"nuf"+std::to_string(i+1)};
        _fis_cs[i] = f* try_value<double>(tbl, name);
    }
    return _fis_cs;
}

params::delayed_precursors::delayed_precursors
(
        const toml::node_view<toml::node> &tbl,
        const conversion_factors& cf,
        const domain& d
):
ld(init_ld(tbl, cf, d)),
bd(init_bd(tbl, d))
{}

std::ostream& params::operator<<(std::ostream& os, const delayed_precursors& dp)
{
    return os << "delayed precursors:\n"
              << "ld=\n" << dp.ld
              << "\nbd=\n" << dp.bd
              << std::endl;
}

torch::Tensor params::delayed_precursors::init_ld
(
        const toml::node_view<toml::node>& tbl,
        const conversion_factors& cf,
        const domain& d
)
{
    torch::Tensor _ld = torch::zeros({d.Gd}, torch::kCUDA);
    for (int i = 0; i < d.Gd; ++i)
    {
        _ld[i] = cf.C_t*try_value<double>(tbl, "l"+std::to_string(i+1));
    }
    return _ld;
}

torch::Tensor params::delayed_precursors::init_bd
(
        const toml::node_view<toml::node>& tbl,
        const domain& d
)
{
    torch::Tensor _bd = torch::zeros({d.Gd}, torch::kCUDA);
    for (int i = 0; i < d.Gd; ++i)
    {
        _bd[i] = try_value<double>(tbl, "b"+std::to_string(i+1));
    }
    return _bd;
}

params::fluid_flow::fluid_flow(const toml::node_view<toml::node> &tbl, const conversion_factors& cf):
u_d(try_value<double>(tbl, "u")/cf.C_u),
rho(try_value<double>(tbl, "rho")/cf.C_rho),
nu(try_value<double>(tbl, "kinematic_viscosity")/(cf.C_l*cf.C_u)),
tau_0(0.5+3.0*nu),
Cs(try_value<double>(tbl, "Cs"))
{}

std::ostream& params::operator<<(std::ostream& os, const fluid_flow& ff)
{
    return os << "fluid-flow parameters:\n"
              << "u_d=" << ff.u_d
              << "\nrho=" << ff.rho
              << "\nnu=" << ff.nu
              << "\ntau_0=" << ff.tau_0
              << "\nCs=" << ff.Cs
              << std::endl;
}
