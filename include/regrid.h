#ifndef REGRID_KNMI_H
#define REGRID_KNMI_H
#include "stdio.h"

extern "C" void regrid_knmi(const size_t nx, const size_t ny, 
                 const double lat_a[], const double lon_a[], const double height_a[],
                 const double lat[], const double lon[], double height[]);
#endif
