// LBM code
// disassemble 1D index to 3D coordinates (n -> x,y,z)
uint3 coordinates(const uint n) 
{ 
	const uint3 t  = (uint3)(n,n,n);
	const uint3 sh = (uint3)(NX_BITS, NY_BITS, NZ_BITS);
	const uint3 msk= (uint3)(NX_MASK, NY_MASK, NZ_MASK);
	return (t>>sh) & msk;// n = x+(y+z*Ny)*Nx
}
// assemble 1D index from 3D coordinates (x,y,z -> n)
uint index(uint3 v) { 
	const uint3 sh = (uint3)(NX_BITS, NY_BITS, NZ_BITS);
	v<<=sh; 
	return v.x|v.y|v.z; // n = x+(y+z*Ny)*Nx
}
// 3D coordinates to 3D position
float3 position(uint3 v) { 
	const float3 N_2= (float3)((NX-1)*0.5f, (NY-1)*0.5f, (NZ-1)*0.5f);
	return convert_float3(v) - N_2;
}
// mirror position into periodic boundaries
float3 mirror_position(const float3 p) { 
	const float3 N= (float3)((NX-1)*0.5f, (NY-1)*0.5f, (NZ-1)*0.5f);
	return sign(p)*(fmod(fabs(p) + N_2, (float)def_Nx)-0.5f*(float)def_Nx);
}

// calculate f_equilibrium from density and velocity field (perturbation method / DDF-shifting)
void calculate_f_eq(const float rho, float3 u, float* feq) { 
	const float c3=-3.0f*dot(u,u);//dot = (sq(ux)+sq(uy)+sq(uz));
	, rhom1=rho-1.0f; // c3 = -2*sq(u)/(2*sq(c)), rhom1 is arithmetic optimization to minimize digit extinction
	u *= 3.0f;
	feq[ 0] = def_w0*(rho*0.5f*c3 + rho - 1.0f); // 000 (identical for all velocity sets)
	
#if defined(D3Q27)
	const float u0=ux+uy, u1=ux+uz, u2=uy+uz; 
	const float u3=ux-uy, u4=ux-uz, u5=uy-uz; 
	const float u6=ux+uy+uz;
	const float u7=ux+uy-uz, u8=ux-uy+uz, u9=-ux+uy+uz;
	
	const float rhos=def_ws*rho, rhoe=def_we*rho, rhoc=def_wc*rho;
	const float rhom1s=def_ws*rhom1, rhom1e=def_we*rhom1, rhom1c=def_wc*rhom1;
f_eq1 = (rhos*(0.5f*(u*u + c3) + u) + rhom1s); 
f_eq2 = (rhos*(0.5f*(u*u + c3) - u) + rhom1s); 
f_eq7 = (rhoe*(0.5f*(v*v + c3) + v) + rhom1e); 
f_eq8 = (rhoe*(0.5f*(v*v + c3) - v) + rhom1e); 
f_eq13= (rhoc*(0.5f*(w*w + c3) + w) + rhom1c); 
f_eq14= (rhoc*(0.5f*(w*w + c3) - w) + rhom1c); 

	feq[ 1] = (rhos*(0.5f*(ux*ux + c3) + ux) + rhom1s); 
	feq[ 2] = (rhos*(0.5f*(ux*ux + c3) - ux) + rhom1s); // +00 -00
	feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3),  uy), rhom1s); 
	feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	feq[ 5] = fma(rhos, fma(0.5f, fma(uz, uz, c3),  uz), rhom1s); 
	feq[ 6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
	
	feq[ 7] = fma(rhoe, fma(0.5f, fma(u0, u0, c3),  u0), rhom1e); 
	feq[ 8] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
	feq[ 9] = fma(rhoe, fma(0.5f, fma(u1, u1, c3),  u1), rhom1e); 
	feq[10] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +0+ -0-
	feq[11] = fma(rhoe, fma(0.5f, fma(u2, u2, c3),  u2), rhom1e); 
	feq[12] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), -u2), rhom1e); // 0++ 0--
	
	feq[13] = fma(rhoe, fma(0.5f, fma(u3, u3, c3),  u3), rhom1e); 
	feq[14] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), -u3), rhom1e); // +-0 -+0
	feq[15] = fma(rhoe, fma(0.5f, fma(u4, u4, c3),  u4), rhom1e); 
	feq[16] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), -u4), rhom1e); // +0- -0+
	feq[17] = fma(rhoe, fma(0.5f, fma(u5, u5, c3),  u5), rhom1e); 
	feq[18] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), -u5), rhom1e); // 0+- 0-+
	
	feq[19] = fma(rhoc, fma(0.5f, fma(u6, u6, c3),  u6), rhom1c); 
	feq[20] = fma(rhoc, fma(0.5f, fma(u6, u6, c3), -u6), rhom1c); // +++ ---
	feq[21] = fma(rhoc, fma(0.5f, fma(u7, u7, c3),  u7), rhom1c); 
	feq[22] = fma(rhoc, fma(0.5f, fma(u7, u7, c3), -u7), rhom1c); // ++- --+
	feq[23] = fma(rhoc, fma(0.5f, fma(u8, u8, c3),  u8), rhom1c); 
	feq[24] = fma(rhoc, fma(0.5f, fma(u8, u8, c3), -u8), rhom1c); // +-+ -+-
	feq[25] = fma(rhoc, fma(0.5f, fma(u9, u9, c3),  u9), rhom1c); 
	feq[26] = fma(rhoc, fma(0.5f, fma(u9, u9, c3), -u9), rhom1c); // -++ +--
#endif

} // calculate_f_eq()
int color_average(uchar4 c1, uchar4 c2) { // (c1+c2)/s
	return rhadd(a,b);
/*	return as_int((uchar4)((uchar)((cc1.x+cc2.x)/2u), (uchar)((cc1.y+cc2.y)/2u), (uchar)((cc1.z+cc2.z)/2u), (uchar)0u)); */
}
uchar4 color_dim(uchar4 c, float x) 
{
	return convert_uchar4_sat_rte(convert_float4(c)*x); // округление к ближайшему целому
/*	
	const int r = clamp((int)fma((float)((c>>16)&255), x, 0.5f), 0, 255);
	const int g = clamp((int)fma((float)((c>> 8)&255), x, 0.5f), 0, 255);
	const int b = clamp((int)fma((float)( c     &255), x, 0.5f), 0, 255);
	return (r&255)<<16|(g&255)<<8|(b&255); */
}
uchar4 color_mix(const uchar4 c1, const uchar4 c2, const float w) { // w*c1+(1-w)*c2
	return convert_uchar4_sat_rte(mix(convert_float4(c2[gid]),convert_float4(c1[gid]), w));
/*	return mix(convert_half4(c2),convert_half4(c1), (half)w);
	const float3 fc1=(float3)((float)cc1.x, (float)cc1.y, (float)cc1.z), fc2=(float3)((float)cc2.x, (float)cc2.y, (float)cc2.z);
	const float3 fcm = fma(w, fc1, fma(1.0f-w, fc2, (float3)(0.5f, 0.5f, 0.5f)));
	return as_int((uchar4)((uchar)fcm.x, (uchar)fcm.y, (uchar)fcm.z, (uchar)0u)); */
}
uchar4 color_mix_3(uchar4 c0, uchar4 c1, uchar4 c2, float w0, float w1, float w2) { // w1*c1+w2*c2+w3*c3, w0+w1+w2 = 1
	return convert_uchar4_sat_rte(convert_float4(c0)*w0 + convert_float4(c1)*w1 + convert_float4(c2)*w2);
/*	const uchar4 cc0=as_uchar4(c0), cc1=as_uchar4(c1), cc2=as_uchar4(c2);
	const float3 fc0=(float3)((float)cc0.x, (float)cc0.y, (float)cc0.z),  fc1=(float3)((float)cc1.x, (float)cc1.y, (float)cc1.z), fc2=(float3)((float)cc2.x, (float)cc2.y, (float)cc2.z);
	const float3 fcm = fma(w0, fc0, fma(w1, fc1, fma(w2, fc2, (float3)(0.5f, 0.5f, 0.5f))));
	return as_int((uchar4)((uchar)fcm.x, (uchar)fcm.y, (uchar)fcm.z, (uchar)0u));*/
}
void neighbors(const uint n, uint* j) { // calculate neighbor indices
	uint x0, xp, xm, y0, yp, ym, z0, zp, zm;
	calculate_indices(n, &x0, &xp, &xm, &y0, &yp, &ym, &z0, &zp, &zm);
	j[0] = n;
#if defined(D2Q9)
	j[ 1] = xp+y0; j[ 2] = xm+y0; // +00 -00
	j[ 3] = x0+yp; j[ 4] = x0+ym; // 0+0 0-0
	j[ 5] = xp+yp; j[ 6] = xm+ym; // ++0 --0
	j[ 7] = xp+ym; j[ 8] = xm+yp; // +-0 -+0
#elif defined(D3Q15)
	j[ 1] = xp+y0+z0; j[ 2] = xm+y0+z0; // +00 -00
	j[ 3] = x0+yp+z0; j[ 4] = x0+ym+z0; // 0+0 0-0
	j[ 5] = x0+y0+zp; j[ 6] = x0+y0+zm; // 00+ 00-
	j[ 7] = xp+yp+zp; j[ 8] = xm+ym+zm; // +++ ---
	j[ 9] = xp+yp+zm; j[10] = xm+ym+zp; // ++- --+
	j[11] = xp+ym+zp; j[12] = xm+yp+zm; // +-+ -+-
	j[13] = xm+yp+zp; j[14] = xp+ym+zm; // -++ +--
#elif defined(D3Q19)
	j[ 1] = xp+y0+z0; j[ 2] = xm+y0+z0; // +00 -00
	j[ 3] = x0+yp+z0; j[ 4] = x0+ym+z0; // 0+0 0-0
	j[ 5] = x0+y0+zp; j[ 6] = x0+y0+zm; // 00+ 00-
	j[ 7] = xp+yp+z0; j[ 8] = xm+ym+z0; // ++0 --0
	j[ 9] = xp+y0+zp; j[10] = xm+y0+zm; // +0+ -0-
	j[11] = x0+yp+zp; j[12] = x0+ym+zm; // 0++ 0--
	j[13] = xp+ym+z0; j[14] = xm+yp+z0; // +-0 -+0
	j[15] = xp+y0+zm; j[16] = xm+y0+zp; // +0- -0+
	j[17] = x0+yp+zm; j[18] = x0+ym+zp; // 0+- 0-+
#elif defined(D3Q27)
	j[ 1] = xp+y0+z0; j[ 2] = xm+y0+z0; // +00 -00
	j[ 3] = x0+yp+z0; j[ 4] = x0+ym+z0; // 0+0 0-0
	j[ 5] = x0+y0+zp; j[ 6] = x0+y0+zm; // 00+ 00-
	j[ 7] = xp+yp+z0; j[ 8] = xm+ym+z0; // ++0 --0
	j[ 9] = xp+y0+zp; j[10] = xm+y0+zm; // +0+ -0-
	j[11] = x0+yp+zp; j[12] = x0+ym+zm; // 0++ 0--
	j[13] = xp+ym+z0; j[14] = xm+yp+z0; // +-0 -+0
	j[15] = xp+y0+zm; j[16] = xm+y0+zp; // +0- -0+
	j[17] = x0+yp+zm; j[18] = x0+ym+zp; // 0+- 0-+
	j[19] = xp+yp+zp; j[20] = xm+ym+zm; // +++ ---
	j[21] = xp+yp+zm; j[22] = xm+ym+zp; // ++- --+
	j[23] = xp+ym+zp; j[24] = xm+yp+zm; // +-+ -+-
	j[25] = xm+yp+zp; j[26] = xp+ym+zm; // -++ +--
#endif
} // neighbors()