#include "hrl_manipulation_task/util.h"


double dist_Point_to_Segment(const KDL::Vector &P,const KDL::Vector &S0,const KDL::Vector &S1)
{
    // std::cout << S0 << S1 << std::endl;

    KDL::Vector v = S1 - S0;
    KDL::Vector w = P - S0;

    double c1 = KDL::dot(w,v);
    if ( c1 <= 0 )
        return (P-S0).Norm();

    double c2 = KDL::dot(v,v);
    if ( c2 <= c1 )
        return (P-S1).Norm();

    double b = c1 / c2;
    KDL::Vector Pb = S0 + b * v;
    return (P-Pb).Norm();
}

