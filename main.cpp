#include <iostream>
#include <torch/torch.h>
#include <toml++/toml.hpp>
#include "src/angular_directions.hpp"
#include "src/neutron_transport.hpp"
#include "src/delayed_precursors.hpp"
#include "src/params.hpp"
#include "src/d2q9.hpp"

using std::cout;
using std::endl;

using torch::Tensor;
using torch::indexing::Slice;

using d2q9::E;
using d2q9::W;

int main(int argc, char* argv[])
{
    // Read parameters
    toml::table tbl;
    try
    {
        tbl = toml::parse_file(argv[1]);
    }
    catch (const toml::parse_error& err)
    {
        std::cerr << "Parsing failed:\n" << err << endl;
        return 1;
    }

    const params::conversion_factors conversion_factors{tbl};
    cout << conversion_factors << endl;
    const params::domain domain{tbl["domain"]};
    cout << domain << endl;
    const params::fluid_flow fluid_flow{tbl["fluid-flow"], conversion_factors};
    cout << fluid_flow << endl;
    const params::delayed_precursors delayed_precursors{tbl["delayed-precursors"], conversion_factors, domain};
    cout << delayed_precursors << endl;
    const params::neutron_transport neutron_transport{tbl["neutron-transport"], conversion_factors, domain};
    cout << neutron_transport << endl;

    class neutron_transport nt{neutron_transport, domain, delayed_precursors.bd.sum().item<double>()};
    class delayed_precursors dp{delayed_precursors, domain};

    return 0;
}
