#include "delayed_precursors.hpp"
#include "d2q9.hpp"

#define ux ff.u.index({Ellipsis, 0}).unsqueeze(-1)
#define uy ff.u.index({Ellipsis, 1}).unsqueeze(-1)
#define Ex d2q9::E.index({0,Ellipsis})
#define Ey d2q9::E.index({1,Ellipsis})

delayed_precursors::delayed_precursors
(
        const params::delayed_precursors& dp,
        const params::domain& d
):
ad(d.A, d.D),
ld(dp.ld),
del_beta(dp.bd)
{
    auto dev = torch::kCUDA;
    Cd = torch::zeros({d.R, d.C, d.Gd}, dev);
    ef = torch::zeros({d.R, d.C, d.Gd, d.Q}, dev);
    af = torch::zeros({d.R, d.C, d.Gd, d.Q}, dev);
    cf = torch::zeros({d.R, d.C, d.Gd, d.Q}, dev);
    S  = torch::zeros({d.R, d.C, d.Gd, d.Q}, dev);
}

void delayed_precursors::step(const neutron_transport& nt, const fluid_flow& ff)
{
    eval_equilibrium(ff);
    eval_sources(nt, ff);
    collide();
    advect();
    boundary_condition();
}

void delayed_precursors::eval_equilibrium(const fluid_flow& ff)
{
    using torch::indexing::Ellipsis;
    using d2q9::W;
    using d2q9::cs2;
    using d2q9::ics2;
    using d2q9::ics4;
    ef.copy_(Cd.unsqueeze(-1).mul((W*(
        1.0 + ics2*(ux*Ex + uy*Ey)
        + 0.5*ics4*((ux*ux - cs2)*(Ex*Ex - cs2) + 2.0*ux*uy*Ex*Ey + (uy*uy - cs2)*(Ey*Ey - cs2))
    )).unsqueeze(-2))
    );
}

void delayed_precursors::eval_sources(const neutron_transport& nt, const fluid_flow& ff)
{
    using d2q9::E;
    using d2q9::W;
    using d2q9::ics2;

    S.copy_(
            (W*(1.0+2.0*ics2*ff.u.matmul(E))).unsqueeze(-2)
            *( nt.ikeff*del_beta*(nt.fis_cs*nt.snf).sum(-1).unsqueeze(-1) - ld*Cd).unsqueeze(-1)
            );
}

void delayed_precursors::collide()
{
    cf.copy_(ef + S);
}

void delayed_precursors::boundary_condition()
{

}

void delayed_precursors::advect()
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

