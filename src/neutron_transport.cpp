#include <iostream>

#include "neutron_transport.hpp"
#include "d2q9.hpp"

neutron_transport::neutron_transport
(
        const params::neutron_transport& nt,
        const params::domain& d,
        double b_tot
):
ikeff(1.0),
tot_beta(b_tot),
ad(d.A, d.D),
xi(init_xi(d.A, d.Q, ad)),
eps(init_eps(d.A, d.Q, ad)),
ins_cs(nt.ins_cs),
tot_cs(nt.tot_cs),
fis_cs(nt.fis_cs),
pro_chi(init_pro_chi(d.G)),
del_chi(init_del_chi(d.G, d.Gd))
{
    auto dev = torch::kCUDA;
    snf = torch::zeros({d.R,d.C,d.G}, dev);
    q   = torch::zeros({d.R,d.C}, dev);
    anf = torch::zeros({d.R,d.C,d.G,d.A}, dev);
    ef  = torch::zeros({d.R,d.C,d.G,d.A,d.Q}, dev);
    af  = torch::zeros({d.R,d.C,d.G,d.A,d.Q}, dev);
    cf  = torch::zeros({d.R,d.C,d.G,d.A,d.Q}, dev);
    S   = torch::zeros({d.R,d.C,d.G,d.A,d.Q}, dev);

}

void neutron_transport::step(const torch::Tensor& Cd, const torch::Tensor& ld)
{
    eval_equilibrium();
    eval_sources(Cd, ld);
    collide();
    advect();
    boundary_condition();
}

void neutron_transport::eval_equilibrium()
{
    ef.copy_(anf.mul(xi));
}

void neutron_transport::eval_sources(const torch::Tensor& Cd, const torch::Tensor& ld)
{
    S.copy_(
        eps.mul((
        ins_cs.matmul(anf.sum(-1).unsqueeze(-1))
        + anf.mul(tot_cs)
        + (tot_beta*ikeff)*pro_chi.mul(snf.mul(fis_cs).sum(-1).unsqueeze(-1)).unsqueeze(-1)
        + del_chi.matmul(Cd.mul(ld).unsqueeze(-1))
        ).unsqueeze(-1))
    );
}

void neutron_transport::collide()
{
    cf.copy_(ef + S);
}

void neutron_transport::boundary_condition()
{

}

void neutron_transport::advect()
{
    using torch::indexing::None;
    using torch::indexing::Slice;
    using torch::indexing::Ellipsis;

    // Advect
    // f0
    af.index({Ellipsis,0}) = cf.index({Ellipsis,0}).clone().detach();

    // f1
    //print("f1");
    af.index({Slice(1,None), Slice(),Ellipsis,1}) = cf.index({Slice(0,-1), Slice(),Ellipsis,1}).clone().detach();
    af.index({0, Slice(),Ellipsis,1}) = cf.index({-1, Slice(),Ellipsis,1}).clone().detach();

    // f2
    //print("f2");
    af.index({Slice(), Slice(1,None),Ellipsis,2}) = cf.index({Slice(), Slice(0,-1),Ellipsis,2}).clone().detach();
    af.index({Slice(), 0,Ellipsis,2}) = cf.index({Slice(), -1,Ellipsis,2}).clone().detach();

    // f3
    //print("f3");
    af.index({Slice(0,-1), Slice(),Ellipsis,3}) = cf.index({Slice(1,None), Slice(),Ellipsis,3}).clone().detach();
    af.index({-1, Slice(),Ellipsis,3}) = cf.index({0,Slice(),Ellipsis,3}).clone().detach();

    // f4
    //print("f4");
    af.index({Slice(), Slice(0,-1),Ellipsis,4}) = cf.index({Slice(), Slice(1,None),Ellipsis,4}).clone().detach();
    af.index({Slice(), -1,Ellipsis,4}) = cf.index({Slice(), 0,Ellipsis,4}).clone().detach();

    // f5
    //print("f5");
    af.index({Slice(1,None), Slice(1,None),Ellipsis,5}) = cf.index({Slice(0,-1), Slice(0,-1),Ellipsis,5}).clone().detach();
    af.index({0, Slice(1,None),Ellipsis,5}) = cf.index({-1, Slice(0,-1),Ellipsis,5}).clone().detach();
    af.index({Slice(1,None), 0,Ellipsis,5}) = cf.index({Slice(0,-1), -1,Ellipsis,5}).clone().detach();
    af.index({0, 0,Ellipsis,5}) = cf.index({-1, -1,Ellipsis,5}).clone().detach();

    // f6
    //print("f6");
    af.index({Slice(0,-1), Slice(1,None),Ellipsis,6}) = cf.index({Slice(1,None), Slice(0,-1),Ellipsis,6}).clone().detach();
    af.index({-1, Slice(1,None),Ellipsis,6}) = cf.index({0, Slice(0,-1),Ellipsis,6}).clone().detach();
    af.index({Slice(0,-1), 0,Ellipsis,6}) = cf.index({Slice(1,None), -1,Ellipsis,6}).clone().detach();
    af.index({-1, 0,Ellipsis,6}) = cf.index({0, -1,Ellipsis,6}).clone().detach();

    // f7
    //print("f7");
    af.index({Slice(0,-1), Slice(0,-1),Ellipsis,7}) = cf.index({Slice(1,None), Slice(1,None),Ellipsis,7}).clone().detach();
    af.index({-1, Slice(0,-1),Ellipsis,7}) = cf.index({0, Slice(1,None),Ellipsis,7}).clone().detach();
    af.index({Slice(0,-1), -1,Ellipsis,7}) = cf.index({Slice(1,None), 0,Ellipsis,7}).clone().detach();
    af.index({-1, -1,Ellipsis,7}) = cf.index({0, 0,Ellipsis,7}).clone().detach();

    // f8
    //print("f8");
    af.index({Slice(1,None), Slice(0,-1),Ellipsis,8}) = cf.index({Slice(0,-1), Slice(1,None),Ellipsis,8}).clone().detach();
    af.index({0, Slice(0,-1),Ellipsis,8}) = cf.index({-1, Slice(1,None),Ellipsis,8});
    af.index({Slice(1,None), -1,Ellipsis,8}) = cf.index({Slice(0,-1), 0,Ellipsis,8});
    af.index({0, -1,Ellipsis,8}) = cf.index({-1, 0,Ellipsis,8}).clone().detach();
}

torch::Tensor neutron_transport::init_xi(int A, int Q, const angular_directions &_ad)
{
    using d2q9::E;
    using d2q9::W;
    using d2q9::cs2;
    using d2q9::ics2;
    using d2q9::ics4;
    using torch::Tensor;
    using torch::indexing::Slice;

    Tensor _xi = torch::zeros({A,Q}, torch::kCUDA);
    const Tensor I = torch::diagflat(torch::tensor({1.0, 1.0}, torch::kCUDA));
    Tensor o = torch::zeros({2,1}, torch::kCUDA);
    Tensor e = torch::zeros({2,1}, torch::kCUDA);
    // Elementwise initialization is fine since this is only executed once
    for (int a = 0; a < A; ++a)
    {
        for (int i = 0; i < Q; ++i)
        {
            o.copy_(_ad.omega.index({Slice(),a}).unsqueeze(-1).detach().clone());
            e.copy_(E.index({Slice(),i}).unsqueeze(-1).detach().clone());
            _xi.index_put_({a,i},
               W[i]*( 1.0
               + ics2*(e.t().matmul(o))
               + 2.0*ics4*((o.matmul(o.t()) - cs2*I).mul(e.matmul(e.t()) - cs2*I).sum()) ));
        }
    }

    return _xi;
}

torch::Tensor neutron_transport::init_eps(int A, int Q, const angular_directions &_ad)
{
    using torch::Tensor;
    using torch::indexing::Slice;
    using d2q9::W;
    using d2q9::E;
    using d2q9::ics2;

    Tensor _eps = torch::zeros({A,Q}, torch::kCUDA);

    const Tensor I = torch::diagflat(torch::tensor({1.0, 1.0}, torch::kCUDA));
    Tensor o = torch::zeros({2,1}, torch::kCUDA);
    Tensor e = torch::zeros({2,1}, torch::kCUDA);

    for (int a = 0; a < A; ++a)
    {
        for (int i = 0; i < Q; ++i)
        {
            o.copy_(_ad.omega.index({Slice(),a}).unsqueeze(-1).detach().clone());
            e.copy_(E.index({Slice(),i}).unsqueeze(-1).detach().clone());
            _eps.index_put_({a,i},
                W[i]*(1.0 + ics2*(e.t().matmul(o)))
            );
        }
    }
    return _eps;
}

torch::Tensor neutron_transport::init_pro_chi(int G)
{
    return (1.0/G)*torch::ones({G}, torch::kCUDA);
}

torch::Tensor neutron_transport::init_del_chi(int G, int Gd)
{
    return (1.0/(G*Gd))*torch::ones({G,Gd}, torch::kCUDA);
}
