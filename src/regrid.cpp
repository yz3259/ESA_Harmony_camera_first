#include "regrid.h"
#include <cmath>
#include <stdexcept>

static const int N_WINDOW = 10;


void triangle_to_uniform(const size_t nx, const size_t ny,
                const double jx, const double jy,
                const double lat[], const double lon[], double height[],
                const double lat123[], const double lon123[], const double height123[]){
#   define height(ix,iy) height[ix + nx*(iy)]
#   define lat(ix,iy)    lat[ix + nx*(iy)]
#   define lon(ix,iy)    lon[ix + nx*(iy)]
 
    const int 
       ix_start = std::min((int) nx-1,std::max(0,(int) std::round(jx-N_WINDOW))),
       ix_end   = std::min((int) nx-1,std::max(0,(int) std::round(jx+N_WINDOW))),
       iy_start = std::min((int) ny-1,std::max(0,(int) std::round(jy-N_WINDOW))),
       iy_end   = std::min((int) ny-1,std::max(0,(int) std::round(jy+N_WINDOW)));

    const double
       min_lat = std::min(lat123[0], std::min(lat123[1], lat123[2])),
       max_lat = std::max(lat123[0], std::max(lat123[1], lat123[2])),
       min_lon = std::min(lon123[0], std::min(lon123[1], lon123[2])),
       max_lon = std::max(lon123[0], std::max(lon123[1], lon123[2]));

    for (int ix = ix_start; ix<=ix_end; ix++) {
       for (int iy = iy_start; iy<=iy_end; iy++) {
          if (lat(ix,iy) < min_lat or lat(ix,iy) > max_lat) continue;
          if (lon(ix,iy) < min_lon or lon(ix,iy) > max_lon) continue;
          /* (lat(ix,iy), lon(ix,iy) ) = (lat123[0],lon123[0]) + 
           *    alpha * (lat123[1]-lat123[0], lon123[1]-lon123[0]) + 
           *    beta  * (lat123[2]-lat123[0], lon123[2]-lon123[0])
           *
           * (lat(ix,iy)-lat123[0], lon(ix,iy)-lon123[0] ) = 
           *             lat123[1]-lat123[0], lat123[2]-lat123[0]    alpha
           *             lon123[1]-lon123[0], lon123[2]-lon123[0]    beta
           */
          double A00 = lat123[1] - lat123[0], A01 = lat123[2]-lat123[0],
                 A10 = lon123[1] - lon123[0], A11 = lon123[2]-lon123[0],
                 b0  = lat(ix,iy) - lat123[0], b1 = lon(ix,iy)-lon123[0],
                 detA = A00 * A11 - A01*A10,
                 alpha = (A11 * b0 - A01*b1)/detA,
                 beta  = (-A10 * b0 + A00*b1)/detA,
                 check_lat = lat123[0] + alpha * (lat123[1]-lat123[0]) + beta * (lat123[2]-lat123[0]),
                 check_lon = lon123[0] + alpha * (lon123[1]-lon123[0]) + beta * (lon123[2]-lon123[0]);
          if (std::abs(lat(ix,iy)-check_lat)>1e-7) throw std::runtime_error("lat is niet goed");
          if (std::abs(lon(ix,iy)-check_lon)>1e-7) throw std::runtime_error("lon is niet goed");
          if (alpha < -0.001 or beta<-0.001 or alpha + beta > 1.001) continue;

          height(ix,iy) = std::max(height(ix,iy), 
                            height123[0] + alpha * (height123[1]-height123[0])
                                         + beta  * (height123[2]-height123[0]));
       }
    }
#   undef height_a
#   undef lat_a
#   undef lon_a
#   undef height_loc
#   undef lat_loc
#   undef lon_loc
}
void local_to_uniform(const size_t nx, const size_t ny,
                const double jx, const double jy,
                const double lat[], const double lon[], double height[],
                const double lat_loc[], const double lon_loc[], const double height_loc[]){

#   define height(ix,iy) height[ix + nx*(iy)]
#   define lat(ix,iy)    lat[ix + nx*(iy)]
#   define lon(ix,iy)    lon[ix + nx*(iy)]
#   define height_loc(dix,diy) height_loc[dix + 2*(diy)]
#   define lat_loc(dix,diy)    lat_loc[dix + 2*(diy)]
#   define lon_loc(dix,diy)    lon_loc[dix + 2*(diy)]
 
    int nnonzeros = 0;
    for (int dix = 0; dix<=1; dix++) {
       for (int diy = 0; diy<=1; diy++) {
           if (height_loc(dix,diy)>0) nnonzeros++;
       }
    }
    if (nnonzeros <= 2) return;

    double lat123[3], lon123[3], height123[3];
    /*             __
     *  triangle:  |/   
     */
    lat123[0] = lat_loc(0,0);    lon123[0] = lon_loc(0,0);    height123[0] = height_loc(0,0);
    lat123[1] = lat_loc(1,1);    lon123[1] = lon_loc(1,1);    height123[1] = height_loc(1,1);
    lat123[2] = lat_loc(1,0);    lon123[2] = lon_loc(1,0);    height123[2] = height_loc(1,0);
    if (height123[0]>0 and height123[1]>0 and height123[2]>0) {
       triangle_to_uniform(nx, ny, jx, jy, lat, lon, height, lat123, lon123, height123);
       if (nnonzeros==3) return;
    }


    /*             
     *  triangle:  /|
     */
    lat123[2] = lat_loc(0,1);    lon123[2] = lon_loc(0,1);    height123[2] = height_loc(0,1);
    if (height123[0]>0 and height123[1]>0 and height123[2]>0) {
       triangle_to_uniform(nx, ny, jx, jy, lat, lon, height, lat123, lon123, height123);
       if (nnonzeros==3) return;
    }
    if (nnonzeros==4) return;

    /*             __
     *  triangle:  \|
     */
    lat123[0] = lat_loc(1,0);    lon123[0] = lon_loc(1,0);    height123[0] = height_loc(1,0);
    if (height123[0]>0 and height123[1]>0 and height123[2]>0) {
       triangle_to_uniform(nx, ny, jx, jy, lat, lon, height, lat123, lon123, height123);
       return;
    }

    /*             
     *  triangle:  |\
     */
    lat123[1] = lat_loc(0,0);    lon123[1] = lon_loc(0,0);    height123[1] = height_loc(0,0);
    if (height123[0]>0 and height123[1]>0 and height123[2]>0) {
       triangle_to_uniform(nx, ny, jx, jy, lat, lon, height, lat123, lon123, height123);
       return;
    }

#   undef height_a
#   undef lat_a
#   undef lon_a
#   undef height_loc
#   undef lat_loc
#   undef lon_loc
}

void local_grid(const size_t nx, 
                const size_t ix, const size_t iy,
                const double lat_a[], const double lon_a[], const double height_a[],
                double lat_loc[], double lon_loc[], double height_loc[]) {

#   define height_a(ix,iy) height_a[ix + nx*(iy)]
#   define lat_a(ix,iy)    lat_a[ix + nx*(iy)]
#   define lon_a(ix,iy)    lon_a[ix + nx*(iy)]
#   define height_loc(dix,diy) height_loc[dix + 2*(diy)]
#   define lat_loc(dix,diy)    lat_loc[dix + 2*(diy)]
#   define lon_loc(dix,diy)    lon_loc[dix + 2*(diy)]
    for (int dix = 0; dix<=1; dix++) {
       for (int diy = 0; diy<=1; diy++) {
           lat_loc(dix,diy)    = lat_a(ix+dix,iy+diy);
           lon_loc(dix,diy)    = lon_a(ix+dix,iy+diy);
           height_loc(dix,diy) = height_a(ix+dix,iy+diy);
       }
    }
#   undef height_a
#   undef lat_a
#   undef lon_a
#   undef height_loc
#   undef lat_loc
#   undef lon_loc
}

void regrid_knmi(const size_t nx, const size_t ny, 
                 const double lat_a[], const double lon_a[], const double height_a[],
                 const double lat[], const double lon[], double height[]) {
#   define height(ix,iy)   height[ix + nx*(iy)]
#   define height_a(ix,iy) height_a[ix + nx*(iy)]
#   define lat(ix,iy)      lat[ix + nx*(iy)]
#   define lon(ix,iy)      lon[ix + nx*(iy)]
#   define lat_a(ix,iy)    lat_a[ix + nx*(iy)]
#   define lon_a(ix,iy)    lon_a[ix + nx*(iy)]
    for (size_t ix=0; ix<nx-1; ix++) {
        for (size_t iy=0; iy<ny-1; iy++) {
            int jx = (int) ( (double) ix + (lon(ix,iy)   - lon_a(ix,iy))/std::abs(lon(1,0)-lon(0,0))),
                jy = (int) ( (double) iy + (lat_a(ix,iy) -   lat(ix,iy))/std::abs(lat(0,1)-lat(0,0)));
            if (jx<0 and false) {
               printf("From %ld,%ld: starting at (%d,%d)\n", ix,iy, jx,jy);
            }
            for (int it=0;it<3;it++) {
                if ( jx<=0 or jx>=(int) nx-1 or jy<=0 or jy>=(int) ny-1) break;
                for (;jy>0 and lat(jx,jy) > lat_a(ix,iy);jy--){};
                for (;jy<(int)ny-1 and lat(jx,jy+1) < lat_a(ix,iy);jy++){}
                for (;jx>0 and lon(jx,jy) < lon_a(ix,iy);jx--){};
                for (;jx<(int) nx-1 and lon(jx+1,jy) > lon_a(ix,iy);jx++){}
                if ( jx==0 or jx==(int) nx or jy==0 or jy==(int) ny) break;
                if (std::isnan(lon(jx,jy)) or
                    std::isnan(lon(jx+1,jy)) or
                    std::isnan(lon(jx,jy+1)) or
                    std::isnan(lon(jx+1,jy+1)) or
                    std::isnan(lat_a(ix,iy)) or
                    std::isnan(lon_a(ix,iy)) or
                    std::isnan(lat(jx,jy)) or
                    std::isnan(lat(jx+1,jy)) or
                    std::isnan(lat(jx,jy+1)) or
                    std::isnan(lat(jx+1,jy+1)) or
                    jx==0 or jx==(int) nx or jy==0 or jy==(int) ny or 
                    ( (lon(jx+1,jy)-lon_a(ix,iy)) * (lon(jx,jy)-lon_a(ix,iy)) <=0  and
                      (lat(jx,jy+1)-lat_a(ix,iy)) * (lat(jx,jy)-lat_a(ix,iy)) <=0)) break;
                if (it>=2) {
                    printf("it: %d\n", it);
                    printf("ix,iy: %ld,%ld\n", ix,iy);
                    printf("jx,jy: %d,%d\n", jx,jy);
                    printf("lon: %e,%e\n", lon(jx+1,jy)-lon_a(ix,iy), lon(jx,jy)-lon_a(ix,iy));
                    printf("lat: %e,%e\n", lat(jx,jy+1)-lat_a(ix,iy), lat(jx,jy)-lat_a(ix,iy));
                }
            }

            if (jx<0 and false) {
               printf("Done %ld,%ld: starting at (%d,%d)\n\n", ix,iy, jx,jy);
            }

            double lat_loc[4], lon_loc[4], height_loc[4];
            local_grid(nx, ix, iy, lat_a, lon_a, height_a, lat_loc, lon_loc, height_loc);
            local_to_uniform(nx, ny, jx, jy, lat, lon, height, lat_loc, lon_loc, height_loc);
        }
    }
#   undef height
#   undef height_a
#   undef lat
#   undef lon
#   undef lat_a
#   undef lon_a
}

