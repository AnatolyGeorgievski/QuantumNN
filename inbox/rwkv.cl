#ifndef N
#define N 256
#define Nd 160
#define Ne 180
#endif
static inline float relu2(float x) {x = max(x,0.0f); return x*x; }
static inline float sigma(float x) {return exp(-exp(x));}
static inline float elu  (float x) {return x>=0? x: exp(x)-1;}
// ELU+1
static inline float elu1 (float x) {return x>=0? x+1: exp(x);}
static inline float gelu (float x) {return 0.5f*x*(1.0f+tanh(1.702f*x));}
static inline float silu (float x) {return x/(1.0f+exp(-x));}
static inline float sigmoid(float x) {return 1.0f/(1.0f+exp(-x));}
__kernel void softmax(__global float *r, __global float *x){
	uint gid  = get_local_id(0);
	uint group_size = get_local_size(0);
	float x_max = x[0];
	for(int i=1; i<N; i++)
		x_max = max(x[i], x_max);
	x_max = work_group_reduce_add(x_max)/ group_size;

	float sum = 0;
	for(int i=0; i<N; i++)
		sum += exp(x[i] - x_max);
	sum = work_group_reduce_add(sum)/ group_size;
	if (gid==0)
		r[gid] = sum;//sigma(sum)
}
/*! GRU unit 

```math
\begin{aligned}
z_{t} &=\sigma (W_{z}x_{t}+U_{z}h_{t-1} + b_{z})\\
r_{t} &=\sigma (W_{r}x_{t}+U_{r}h_{t-1} + b_{r})\\
{\hat {h}}_{t}&=\phi (W_{h}x_{t}+U_{h}(r_{t}\odot h_{t-1}) + b_{h})\\
h_{t}&=(1-z_{t})\odot h_{t-1}+z_{t}\odot {\hat {h}}_{t}
\end{aligned}
```

Variables ($d$ denotes the number of input features and $e$ the number of output features):
* $x_{t}\in \mathbb {R} ^{d}$: input  vector
* $h_{t}\in \mathbb {R} ^{e}$: output vector
* ${\hat {h}}_{t}\in \mathbb {R} ^{e}$: candidate activation vector
* $z_{t}\in (0,1)^{e}$: update gate vector
* $r_{t}\in (0,1)^{e}$: reset  gate vector
* $W\in \mathbb {R} ^{e\times d}$, $U\in \mathbb {R} ^{e\times e}$ and $b\in \mathbb {R} ^{e}$: 
parameter matrices and vector which need to be learned during training

*/
__kernel void gru_unit (__global float *h, __global float4 *x, __global float4 *h, __global float4 *w, __global float4 *u, __local float *r){
	
	float rt = 0, zt = 0, ht = 0;
	for(int i=0; i< Nd; i++){
		rt += dot(wr[i],x[i]);
		zt += dot(wz[i],x[i]);
		ht += dot(wh[i],x[i]);
	}
	for(int i=0; i< Ne; i++){
		rt += dot(ur[i],h[i]);
		zt += dot(uz[i],h[i]);
	}
	rt = work_group_reduce_add(rt);
	zt = work_group_reduce_add(zt);
	rt = sigmoid(rt - br);
	zt = sigmoid(zt - bz);
	r[t] = rt;
// барьер - work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for(int i=0; i< Ne; i++)
		ht += dot(uh[i],(r[i]*h[i]));
	ht = work_group_reduce_add(ht);
	h[t] = (1-zt)*h[t] + zt*tanh(ht - bh);
}
/* Gated элемент сверточной сети */
__kernel void gate_layer(__global float *r, __global float4 *x, /* __global float4 *h,*/ __global float4 *w, __global float4 *u){
	uint id  = get_local_id(0);
	uint group_size = get_local_size(0);
	
	float sum =0;
	float4 n=0;// нормировка
	for(int i=0; i<Nd; i++){
		sum += dot(w[i],x[i]);
		n   += w[i];
	}
	for(int i=0; i<Ne; i++){
		sum += dot(u[i],h[i]);
		n   += u[i];
	}

	sum = work_group_reduce_add(sum)/ group_size;
	float norm = n.x+n.y+n.z+n.w;
	norm = work_group_reduce_add(norm)/ group_size;
	if (id==0) {
		float bias = 0.5f;
		r[id] = sigmoid(sum/norm - bias);
	}
}
__kernel void wkv(__global float *r, __global float4 *x, /* __global float4 *h,*/ __global float4 *w)
{
	float sum =0;
	for(int i=0; i<N; i++)
		sum += dot(w[i],x[i]);
}

/*! 
$lerp(a,b, \mu) = mix(a,b, \mu) = a + (b-a) \odot \mu$  
$\mu_x \in \mathbb{R}^D$ - для векторов $\{r,k,v,g\}$

$D$- размерность модели. $D/h$ - разбиение модели на группы (work_group, "head").

$$a_t = lerp_a(x_t, x_{t-1}) W_a, a \in \{r,k,v,g\}$$
$$w = \exp(-\exp(\omega)), \omega \in \mathbb{R}^{D/h}$$

Data dependant 
lora_a 
$$mu = lora_\box(x) = \lambda_\box + \tanh(x A_\box)B_\box$$

$$ddlerp_a(a,b) = mix(a+ (b-a)\odot lora_\box (a+ (b-a)\odot \mu_x)$$

Оператор WKV может быть представленн через рекурсию
```math
\begin{aligned}
wkv' &= s + diag(u) \cdot k^\mathsf{T} \cdot v\\
s_t &= diag(w) \cdot s_{t-1} + k^\mathsf{T} \cdot v
\end{aligned}
```

	\see <https://arxiv.org/pdf/2404.05892>
 */
__kernel void wkv6(_global float *out, __global float4 *x, /* __global float4 *h,*/ __global float4 *w)
{
	uint t  = get_global_id(0);
	float s[N] = {0};
	float vt = as_float(v[t])
	float y  = 0;
	for(int i=0; i<HeadSize; i++){// см. рекурентное определение оператора WKV, N=D/h
		float4 kv = k[i]*vt;
		y += dot(r[i], (u[i]*kv + s[i]));
		s[i] = w[i]*s[i] + kv;
	}
	y = work_group_reduce_add(y);
	out[t] = y;
}

__kernel void mu(__global float *r, __global float *x, __global float *z, __global float *w, __global float *mu){
	uint gid  = get_global_id(0);
	uint offs = gid*N;
	x+=offs; z+=offs; r+=offs, mu+=offs, w+=offs;
	for(int i=0; i<N; i++)
		r[i] = w[i]*mix(z[i],x[i], mu[i]);
		// r[i] = w[i]*((1-\mu)x_{t-1} + \mu x_t);
}
__kernel void biquad(__global float *r, __global float *x, __global float2 *h, __global float4 *w)
{
	uint i = get_local_id(0);
	//for(int i=0; i< N; i++, x++, r++, h++)
	{
		float y = x[i]  + h[i].s0;
		h[i].s0 = w[i].s0*x[i] + w[i].s1*y + h[i].s1; 
		h[i].s1 = w[i].s2*x[i] + w[i].s3*y; 
		r[i] = y;
	}
}

__kernel void lpfa(){
//	h_t =((1-\mu_x)\odot x_{t-1} + \mu_x  x_t)\odot \mu_h + h_{t-1} (1-\mu_h)

}
__kernel void avg(__global float *r, __global float *x, __global float2 *h, __global float4 *w)
{
	uint  i = get_local_id(0);
	float y = (1-mu)*h[i].s0  + mu*((1-mu2)*x[t] + mu2*h[i].s1);
}
__kernel void layer_norm(__global float *r, __global float *x)
{
	uint group_size = get_local_size(0);

	float mean=0;
	for(int i=0; i< N; i++)
		mean += x[i]; 
	mean = work_group_reduce_add(mean)/ group_size;
	float sum=0;
	for(int i=0; i< N; i++){
		float dx  = x[i]-mean;
		sum += dx*dx;
	}
	sum = work_group_reduce_add(sum)/ group_size;
	float inv_std = rsqrt(sum/N);
	for(int i=0; i< N; i++)
		r[i]  = (x[i]-mean)*inv_std;
#if 0 // пример
	if(lid==0)// atomic reduce in global memory
		atomic_add(result, mean);
#endif
}