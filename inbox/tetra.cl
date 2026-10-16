
typedef float4 vertex_t;
typedef struct _Tetra Tetra_t;
struct _Tetra {
	uint4 idx;		// индексы соседей
//	vertex_t v;	// координаты вершин 2D или 3D
};
/* расчет уравнения в ячейке тетраэдра */
__kernel void model_mesh(__constant Tetra_t* mesh,__global float *values, __constant float4 *weights){

	size_t tet_idx = get_global_id (1)*get_global_size(0) + get_global_id(0);// 
	Tetra_t t = mesh[tet_idx];
	float4 weight= weights[tet_idx];
	float S = values[tet_idx];
	for (int k=0; k<4; k++){// по граням соседи
		// составляем уравнение, в котором участвует t и t_k
		S += (values[tet_idx] - values[t.idx[k]])*weight[k];// просто пример
	}
	values[tet_idx] = S;	
}
/* загрузка вершин сетки */
__kernel void load_verices(__global float4* vertices, __constant char *data){
	int idx = get_global_id (0);
	char16 s0 =vload16(0, data+idx*77+ 0);
	char16 s1 =vload16(0, data+idx*77+ 25);
	char16 s2 =vload16(0, data+idx*77+ 50);
	const char16 Z = (char16)('0');
	s0 -= Z;
	s1 -= Z;
	s2 -= Z;
	long4 v;
	for (int j=0; j<16; j++){
		if (s0[j]>=(char)0) v.x = v.x*10 + s0[j];
		if (s1[j]>=(char)0) v.y = v.y*10 + s1[j];
		if (s2[j]>=(char)0) v.z = v.z*10 + s2[j];
	}
	v.w = 0;
	// надо разобраться со знаком и степенью
	vertices[idx] = convert_float4(v);
}
__kernel void load_mesh_index(__global Tetra_t* mesh,__constant char *data){
	int idx = get_global_id (0);// *get_global_size(0) + get_global_id(0);
	char8 n[4];
	uint4 num = 0;
	for (int i=0; i<4; i++){// четыре вершины
		n[i] =vload8(0, data+idx*42+10*i);
		n[i] -= (char8)('0');
	}
	for (int i=0; i<4; i++){// четыре вершины
		for (int j=0; j<8; j++){
			if (n[i][j]>=(char)0) num[i] = num[i]*10 + n[i][j];
		}
	}
	mesh[idx].idx = num;
}