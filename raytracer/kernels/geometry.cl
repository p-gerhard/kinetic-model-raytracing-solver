#ifndef GEOMETRY_CL
#define GEOMETRY_CL

/*
 * Compute the intersections points p1 (entrance) p2 (exit) between a ray of 
 * parameters (xi, vi) and a spere of parameters (sphere_c, sphere_r2). 
 */
bool rt_intersect_sphere(const float3 sphere_c, const float sphere_r2,
						 const float3 xi, const float3 vi, float *p1, float *p2)
{
	const float3 L = sphere_c - xi;
	const float L_dot_v = dot(L, vi);

	/* Case 1: v is not in the direction of the sphere */
	if (L_dot_v < 0) {
		return false;
	}

	/* Case 2: the particle is passing nexto the sphere */
	const float d2 = dot(L, L) - L_dot_v * L_dot_v;
	if (d2 > sphere_r2) {
		return false;
	}

	/* Case 3: the particle collides the sphere */
	const float half_inside_path_len = sqrt(sphere_r2 - d2);
	*p1 = L_dot_v - half_inside_path_len;
	*p2 = L_dot_v + half_inside_path_len;

	return true;
}

/*
 * Compute the distance (dist) between a ray of parameters (xi, vi) and the 
 * intersection point in the plane of parameters (face_p, face_n). 
 */
bool rt_intersect_face(const float3 face_p, const float3 face_n,
					   const float3 xi, const float3 vi, float *dist)

{
	/* Avoid pure parallel case */
	float tol = 1e-6f;
	float n_dot_vi = dot(face_n, vi);

	if (n_dot_vi > tol) {
		const float3 u = face_p - xi;
		*dist = dot(u, face_n) / n_dot_vi;
		/* Avoid collision in -v direction */
		return (*dist >= 0.f);
	}

	return false;
}

/*
 * Compute the minimal distance (dist_min) and on which face of the box
 * (id_face) the next collision will occur. 
 */
int rt_intersect_box(const float3 x, const float3 v, float *dist_min)
{
	float dist = 0;
	float dist_min_tmp = MAXFLOAT;
	int id_min_face;

#ifdef IS_3D
	/* Point that belong to the face */
	const float3 face_p[NB_FACES] = {
		(float3)(BOX_X, 0, 0), (float3)(0, 0, 0),	  (float3)(0, BOX_Y, 0),
		(float3)(0, 0, 0),	   (float3)(0, 0, BOX_Z), (float3)(0, 0, 0)
	};

	/* Outgoing normal vector to the face */
	const float3 face_n[NB_FACES] = { (float3)(1, 0, 0), (float3)(-1, 0, 0),
									  (float3)(0, 1, 0), (float3)(0, -1, 0),
									  (float3)(0, 0, 1), (float3)(0, 0, -1) };
#else
	/* Point that belong to the face */
	const float3 face_p[NB_FACES] = { (float3)(BOX_X, 0, 0), (float3)(0, 0, 0),
									  (float3)(0, BOX_Y, 0),
									  (float3)(0, 0, 0) };

	/* Outgoing normal vector to the face */
	const float3 face_n[NB_FACES] = { (float3)(1, 0, 0), (float3)(-1, 0, 0),
									  (float3)(0, 1, 0), (float3)(0, -1, 0) };
#endif

/* Loop over all boundaries of the box */
#pragma unroll
	for (unsigned int id_face = 0; id_face < NB_FACES; id_face++) {
		if (rt_intersect_face(face_p[id_face], face_n[id_face], x, v, &dist)) {
			if (dist < dist_min_tmp) {
				dist_min_tmp = dist;
				id_min_face = id_face;
			}
		}
	}

	/* Return the biggest non null distance */
	*dist_min = dist_min_tmp;
	return id_min_face;
}

/*
 * Update the velocity vector vi in function to the rebound type.The type of 
 * collision (specular/diffuse) is choosen using the random number
 * rd_but. If the collision is diffuse we use rd_th and rd_ph to set a new 
 * direction for the particule. The direction is choosen uniformly on the interior 
 * half-sphere/circle on the considered boundary point. 
 */
static float3 rt_update_velocity(const int id_face, const float rd_bt,
								 const float rd_th, const float rd_ph,
								 const float3 vi)
{
	float3 v_new;

#ifdef IS_3D
	/* Normal vector to the face n = t1 x t2 */
	const float3 face_n[NB_FACES] = { (float3)(1, 0, 0), (float3)(-1, 0, 0),
									  (float3)(0, 1, 0), (float3)(0, -1, 0),
									  (float3)(0, 0, 1), (float3)(0, 0, -1) };

	/* Tangential vector t1 */
	const float3 face_t1[NB_FACES] = { (float3)(0, 1, 0),  (float3)(0, -1, 0),
									   (float3)(-1, 0, 0), (float3)(1, 0, 0),
									   (float3)(1, 0, 0),  (float3)(1, 0, 0) };

	/* Tangential vector t2 */
	const float3 face_t2[NB_FACES] = { (float3)(0, 0, 1), (float3)(0, 0, 1),
									   (float3)(0, 0, 1), (float3)(0, 0, 1),
									   (float3)(0, 1, 0), (float3)(0, -1, 0) };
#else
	/* Normal vector to the face n = t1 x t2 */
	const float3 face_n[NB_FACES] = { (float3)(1, 0, 0), (float3)(-1, 0, 0),
									  (float3)(0, 1, 0), (float3)(0, -1, 0) };

	/* Tangential vector t1 */
	const float3 face_t1[NB_FACES] = { (float3)(0, 1, 0), (float3)(0, -1, 0),
									   (float3)(-1, 0, 0), (float3)(1, 0, 0) };

	/* Tangential vector t2 */
	const float3 face_t2[NB_FACES] = { (float3)(0, 0, 1), (float3)(0, 0, 1),
									   (float3)(0, 0, 1), (float3)(0, 0, 1) };
#endif

	if (rd_bt < 0) {
		/* Specular rebound */
		v_new = vi - 2.f * dot(vi, face_n[id_face]) * face_n[id_face];

		/* Diffuse rebound (sinus sampling) */
	} else {
		const float sin_th = sqrt(rd_th);
		const float cos_th = sqrt(1.f - sin_th * sin_th);

#ifdef IS_3D
		const float phi = 2.f * M_PI * rd_ph;
		const float cos_ph = cos(phi);
		const float sin_ph = sin(phi);

		const float3 v1 = cos_th * -face_n[id_face];
		const float3 v2 = sin_th * cos_ph * face_t1[id_face];
		const float3 v3 = sin_th * sin_ph * face_t2[id_face];

		v_new = v1 + v2 + v3;
#else
		const int sgn = (rd_ph < 0.5f) ? 1 : -1;
		const float3 v1 = cos_th * -face_n[id_face];
		const float3 v2 = sin_th * sgn * face_t1[id_face];
		v_new = v1 + v2;
#endif
	}
	return v_new;
}

/*
 * Transport one particle to next collision point. If the life time ti>TMAX we
 * kill the particle setting its life time ti at -1. The absorption (alpha) on 
 * the boundary is controlled through the random number (rd_al \in [0, 1]). 
 * If the particle, it s position stays on the boundary and its life time is 
 * set at -1. In any other case, the particle position is update on 
 * the new boundary and the velocity vector updated.
 */
void rt_push_one_particle(const float rd_al, const float rd_bt,
						  const float rd_th, const float rd_ph, float3 *xi,
						  float3 *vi, float *ti)
{
	float dist;
	int id_face = rt_intersect_box(*xi, *vi, &dist);

	float t_new = *ti + dist;
	float3 xi_old = *xi;

	if (isless(t_new, TMAX)) {
		/* Case 1: next even is a collision with a boundary */
		*xi = *xi + dist * (*vi);

		if (rd_al > ALPHA) {
			/* Particle is absorbed */
			*ti = -1.f;
		} else {
			/* Updated of particle's veloctiy and lifetime */
			(*ti) = t_new;
			(*vi) = rt_update_velocity(id_face, rd_bt, rd_th, rd_ph, *vi);
		}

		(*vi) = rt_update_velocity(id_face, rd_bt, rd_th, rd_ph, *vi);

		if (rd_al > ALPHA) {
		}

	} else {
		/* Case 2: next even is NOT a collision with a boundary */
		*xi = *xi + (TMAX - *ti) * (*vi);
		/* Kill the particle using negative life time*/
		*ti = -1.f;
	}
}
#endif