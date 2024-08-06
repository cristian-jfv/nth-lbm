#include <cmath>

#include "angular_directions.hpp"

angular_directions::angular_directions(int A, int D)
{
    // A: number of angular directions
    // D: dimension of the domain
    omega = torch::zeros({D, A});

    domega = 2.0 * 3.141592654 / A;
    // Initialize omega
    for (int a = 0; a < A; ++a)
    {
        omega.index({0, a}) = std::cos(a * domega);
        omega.index({1, a}) = std::sin(a * domega);
    }
}