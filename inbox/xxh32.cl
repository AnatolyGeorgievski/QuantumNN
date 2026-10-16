/*! \brief Расчет некриптографического хеш, используется в качестве контрольной суммы

 */
#define SIZE 1024
__kernel void xxh32(__global uint *H, __constant uint* data/*, size_t data_len*/)
{
	const uint Prime1 = 2654435761U;
	const uint Prime2 = 2246822519U;
	const uint Prime3 = 3266489917U;
	const uint Prime4 =  668265263U;
	const uint Prime5 =  374761393U;

	uint gid = get_global_id(0);// + get_global_size(0)*get_global_id(1);
	uint size = get_global_size(0);

	uint hash = ~0u;//H[0];
    if (SIZE>=16){
        uint4 state = (uint4){Prime1 + Prime2, Prime2, 0, -Prime1};
        state+=hash;
        int blocks = SIZE>>4;
        int  i;
        for (i=0; i<blocks; i++) {// вектор 128 бит
            uint4 block = vload4(i+gid*size, data);
            state = rotate((state + block*Prime2), 13) * Prime1;
        }
		state = rotate(state,  (uint4)(1,7,12,18));
        hash  = state.x + state.y + state.z + state.w;
    } else {
        hash = hash + Prime5;
    }
    hash += SIZE;
    
/*	uint data_len = SIZE & 0xF;
    int i;
    for (i=0; i < data_len>>2; i++, data+=4)
        hash = ROTL32(hash + *(uint32_t*)data * Prime3, 17) * Prime4;
    for (i*=4; i < data_len; i++, data++)
        hash = ROTL32(hash + *data * Prime5, 11) * Prime1; */
    hash ^= hash >> 15;
    hash *= Prime2;
    hash ^= hash >> 13;
    hash *= Prime3;
    hash ^= hash >> 16;
    H[gid] = hash;
}

