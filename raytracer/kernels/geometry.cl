#ifndef GEOMETRY_CL
#define GEOMETRY_CL

#define NB_FACE 6

bool rt_intersect_face(const float3 face_p, const float3 face_n, const float3 x,
					   const float3 v, float *dist)

{
	/* Avoid pure parallel case */
	float tol = 1e-6f;
	float n_dot_v = dot(face_n, v);

	if (n_dot_v > tol) {
		const float3 u = face_p - x;
		*dist = dot(u, face_n) / n_dot_v;
		/* Avoid collision in -v direction */
		return (*dist >= 0.f);
	}

	return false;
}

int rt_intersect_box(const float3 x, const float3 v, float *dist_min)
{
	float dist = 0;
	float dist_min_tmp = FLT_MAX;
	int id_min_face;

	const float3 face_p[NB_FACE] = { (float3)(BOX_X, 0, 0), (float3)(0, 0, 0),
									 (float3)(0, BOX_Y, 0), (float3)(0, 0, 0),
									 (float3)(0, 0, BOX_Z), (float3)(0, 0, 0) };

	const float3 face_n[NB_FACE] = { (float3)(1, 0, 0), (float3)(-1, 0, 0),
									 (float3)(0, 1, 0), (float3)(0, -1, 0),
									 (float3)(0, 0, 1), (float3)(0, 0, -1) };

/* Loop over all edges of the box */
#pragma unroll
	for (unsigned int id_face = 0; id_face < NB_FACE; id_face++) {
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

static float3 rt_update_velocity(const int id_face, const float rd_beta,
								 const float rd_theta, const float rd_phi,
								 const float3 vi)
{
	float3 v_new;

	/* Normal vector to the face n = t1 x t2 */
	const float3 face_n[NB_FACE] = { (float3)(1, 0, 0), (float3)(-1, 0, 0),
									 (float3)(0, 1, 0), (float3)(0, -1, 0),
									 (float3)(0, 0, 1), (float3)(0, 0, -1) };

	/* Tangential vector t1 */
	const float3 face_t1[NB_FACE] = { (float3)(0, 1, 0),  (float3)(0, -1, 0),
									  (float3)(-1, 0, 0), (float3)(1, 0, 0),
									  (float3)(1, 0, 0),  (float3)(1, 0, 0) };

	/* Tangential vector t2 */
	const float3 face_t2[NB_FACE] = { (float3)(0, 0, 1), (float3)(0, 0, 1),
									  (float3)(0, 0, 1), (float3)(0, 0, 1),
									  (float3)(0, 1, 0), (float3)(0, -1, 0) };

	if (rd_beta < 0) {
		/* Specular rebound */
		v_new = vi - 2.f * dot(vi, face_n[id_face]) * face_n[id_face];

		/* Diffuse rebound (sinus sampling) */
	} else {
		const float sin_th = sqrt(rd_theta);
		const float cos_th = sqrt(1.f - sin_th * sin_th);

		const float phi = 2.f * M_PI * rd_phi;
		const float cos_ph = cos(phi);
		const float sin_ph = sin(phi);

		const float3 v1 = cos_th * -face_n[id_face];
		const float3 v2 = sin_th * cos_ph * face_t1[id_face];
		const float3 v3 = sin_th * sin_ph * face_t2[id_face];

		v_new = v1 + v2 + v3;
	}
	return v_new;
}

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
