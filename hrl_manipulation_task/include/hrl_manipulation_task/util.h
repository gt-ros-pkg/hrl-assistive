#ifndef UTIL_H_
#define UTIL_H_

#include <kdl/frames_io.hpp>
#include <vector>

#include <stdio.h>
#include <iostream>

double dist_Point_to_Segment(const KDL::Vector &P,const KDL::Vector &S0,const KDL::Vector &S1);


#endif
