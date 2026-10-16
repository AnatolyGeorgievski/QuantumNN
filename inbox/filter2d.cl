
/* расчет уравнения в ячейке тетраэдра */
__kernel void filter2d(__global float* dest,__constant float *values, __constant float weights[8][8]){

	
	int2 coord;
	coord.x = get_global_id (0);
	coord.y = get_global_id (1);
	size_t size_x = 4096;//get_global_size(0); 
	float S = 0;
	int x,y;
	for (y=0; y<8; y++){
	for (x=0; x<8; x++){
		S += values[(coord.x+size_x*coord.y)+(x+size_x*y)]*weights[y][x];// просто пример
	}
	}
	dest[coord.x+size_x*coord.y] = S;	
}