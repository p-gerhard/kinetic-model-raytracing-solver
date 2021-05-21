#ifndef RAYTRACER_CL
#define RAYTRACER_CL

/* Number of particles */
#define NP _np_

/* Space and time parameters */
#define DIM _dim_

#if (DIM == 3)
#define IS_3D
#endif

#define BOX_X _box_x_
#define BOX_Y _box_y_
#define BOX_Z _box_z_
#define TMAX _tmax_

/* Source parameters */
#define SRC_X _src_x_
#define SRC_Y _src_y_
#define SRC_Z _src_z_
#define SRC_R _src_r_

/* Receptor parameters */
#define RCP_X _rcp_x_
#define RCP_Y _rcp_y_
#define RCP_Z _rcp_z_
#define RCP_R _rcp_r_

/* Boundary parameters */
#define ALPHA _alpha_
#define BETA _beta_

#include "geometry.cl"

__kernel void rt_init_particles(__global float *x, __global float *v)
{
	int id_part = get_global_id(0);

	/* Position */
	x[DIM * id_part + 0] = SRC_X;
	x[DIM * id_part + 1] = SRC_Y;
#ifdef IS_3D
	x[DIM * id_part + 2] = SRC_Z;
#endif

	/* Velocity */
#ifdef IS_3D
	float3 vi = (float3)(v[DIM * id_part + 0], v[DIM * id_part + 1],
						 v[DIM * id_part + 2]);
#else
	float3 vi = (float2)(v[DIM * id_part + 0], v[DIM * id_part + 1], 0.f);
#endif

	float3 vn = normalize(vi);
	v[DIM * id_part + 0] = vn.x;
	v[DIM * id_part + 1] = vn.y;

#ifdef IS_3D
	v[DIM * id_part + 2] = vn.z;
#else
	v[DIM * id_part + 2] = 0.f;
#endif
}

__kernel void rt_push_particles(__global const float *rand, __global float *x,
								__global float *v, __global float *t)
{
	int id_part = get_global_id(0);

	float ti = t[id_part];

	if (isgreaterequal(ti, 0.f)) {
/* Local copy of particle's data */
#ifdef IS_3D
		float3 xi = (float3)(x[DIM * id_part + 0], x[DIM * id_part + 1],
							 x[DIM * id_part + 2]);

		float3 vi = (float3)(v[DIM * id_part + 0], v[DIM * id_part + 1],
							 v[DIM * id_part + 2]);

#else
		float3 xi = (float3)(x[DIM * id_part + 0], x[DIM * id_part + 1], 0.f);
		float3 vi = (float3)(v[DIM * id_part + 0], v[DIM * id_part + 1], 0.f);
#endif

		/* Local copy of random numbers (in [eps, 1-eps[) */
		const float eps = 1e-5f;
		const float rd_al = fmin(fmax(eps, rand[4 * id_part + 0]), 1.f - eps);
		const float rd_bt = fmin(fmax(eps, rand[4 * id_part + 1]), 1.f - eps);
		const float rd_th = fmin(fmax(eps, rand[4 * id_part + 2]), 1.f - eps);
		const float rd_ph = fmin(fmax(eps, rand[4 * id_part + 3]), 1.f - eps);

		rt_push_one_particle(rd_al, rd_bt, rd_th, rd_ph, &xi, &vi, &ti);

		/* Update global data */
		x[DIM * id_part + 0] = xi.x;
		x[DIM * id_part + 1] = xi.y;

		v[DIM * id_part + 0] = vi.x;
		v[DIM * id_part + 1] = vi.y;

#ifdef IS_3D
		x[DIM * id_part + 2] = xi.z;
		v[DIM * id_part + 2] = vi.z;
#endif
		t[id_part] = ti;
	}
}
#endif