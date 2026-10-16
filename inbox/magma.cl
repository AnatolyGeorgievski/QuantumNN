/*! Национальый алгоритм блочного шифрования с длиной блока 64 бит. TC26 Магма

	Copyright (c) 2015-2022 Anatoly Georgievskii <anatoly.georgievski@gmail.com>

	\see ГОСТ Р 34.12-2015 Информационная технология. КРИПТОГРАФИЧЕСКАЯ ЗАЩИТА ИНФОРМАЦИИ. Блочные шифры
	\see http://tc26.ru


	\see Р 1323565.1.022—2018 Функции выработки производного ключа



Multilinear Galois Mode (MGM)
	[Р 1323565.1.026—2019] Режимы работы блочных шифров, реализующие
аутентифицированное шифрование
	https://datatracker.ietf.org/doc/html/rfc9058
	https://datatracker.ietf.org/doc/html/draft-smyshlyaev-mgm-20
Отладка
$ gcc -DTEST_MAGMA -march=native -o magma magma.c

AMD RDNA 2,3: "RDNA3" Instruction Set Architecture

S_BFE_U32 38
Extract an unsigned bitfield from the first input using field offset and size encoded in the second input, store
the result into a scalar register and set SCC iff the result is nonzero.

D0.u = ((S0.u >> S1.u[4 : 0].u) & ((1U << S1.u[22 : 16].u) - 1U));

S_BFM_B32 42
Calculate a bitfield mask given a field offset and size and store the result in a scalar register.

D0.u = (((1U << S0.u[4 : 0].u) - 1U) << S1.u[4 : 0].u)

V_BFM_B32 797
Bitfield modify. S0 is the bitfield width and S1 is the bitfield offset.

D0.u = 32'U(((1 << S0[4 : 0].u) - 1) << S1[4 : 0].u)

V_BFE_U32 528
Bitfield extract. Extract unsigned bitfield from first operand using field offset in second operand and field size
in third operand.
D0.u = ((S0.u >> S1.u[4 : 0].u) & 32'U((1 << S2.u[4 : 0].u) - 1))

V_BFI_B32 530 -- см. bitselect(s2,s1,s0);
Bitfield insert. Using a bitmask from the first operand, merge bitfield in second operand with packed value in
third operand.
D0.u = ((S0.u & S1.u) | (~S0.u & S2.u))


//clang -Xclang -finclude-default-header -cl-std=CL2.0 -march=rocketlake -S -O3 -o magma.s magma.cl
clang --target=amdgcn -mcpu=gfx1102 -Xclang -finclude-default-header -cl-mad-enable -cl-std=CL2.0 -S -O3 -o magma.s magma.cl
llvm-mca --march=amdgcn --mcpu=gfx1102 -timeline --timeline-max-cycles=210 magma.s | less
 */
#define ROL(x, n) (((x)<<(n))|((x)>>(32-n)))
// Bit Field Extract
#define BEXTR(x, n, len) (((x) >> (n)) & ((1 << (len))-1))
#define BFI(x, y, n, len) x = ((x) & ~(((1 << (len))-1)<<(n))) | ((y & ((1 << (len))-1))<<(n))


/*! \brief Подстановка SBOX

    Реализация эффективно работает на архитектуре ARM, задействует команды
    UBFX -- Unsigned Bit Field Extract
    BFI  -- Bit Field Insert
*/

// Bit Field Extract, преобразует в инструкцию bextrl
/* 5.1.1 Нелинейное биективное преобразование */

__constant const uint sbox[8][16] = {
/* π0' = */ {12, 4, 6, 2, 10, 5, 11, 9, 14, 8, 13, 7, 0, 3, 15, 1},
/* π1' = */ {6, 8, 2, 3, 9, 10, 5, 12, 1, 14, 4, 7, 11, 13, 0, 15},
/* π2' = */ {11, 3, 5, 8, 2, 15, 10, 13, 14, 1, 7, 4, 12, 9, 6, 0},
/* π3' = */ {12, 8, 2, 1, 13, 4, 15, 6, 7, 0, 10, 5, 3, 14, 9, 11},
/* π4' = */ {7, 15, 5, 10, 8, 1, 6, 13, 0, 9, 3, 14, 11, 4, 2, 12},
/* π5' = */ {5, 13, 15, 6, 9, 2, 12, 10, 11, 7, 8, 1, 4, 3, 14, 0},
/* π6' = */ {8, 14, 2, 5, 6, 9, 1, 12, 15, 4, 11, 0, 13, 10, 3, 7},
/* π7' = */ {1, 7, 14, 13, 0, 5, 8, 3, 4, 15, 10, 6, 9, 12, 11, 2}
};

static inline uint t(uint a)
{

	uint r=0;
	int i;
	#pragma unroll
	for (i=0; i<8; i++){
		uint s = sbox[i][BEXTR(a,(i*4),4)];
		//r = bitselect(r, s<<(i*4), (uint)0xF<<(i*4));
		r |= (uint)(s)<<(i*4);
	}
	return r;
}

static inline uint g(uint k, uint a)
{
	return rotate(t(k+a), (uint)11);// rotate left
}
__kernel void magma_encrypt(__global uint2 *v, __global uint2 *T, __constant uint *K){
	//uint gid = get_global_id(0);
	uint gid = get_global_id(0);//+get_global_id(1)*get_global_size(0);
	uint n1 = v[gid].lo, n2 = v[gid].hi;
	for (int i=0; i<3; i++)
	{
		n2 ^= g(K[7], n1);
		n1 ^= g(K[6], n2);
		n2 ^= g(K[5], n1);
		n1 ^= g(K[4], n2);
		n2 ^= g(K[3], n1);
		n1 ^= g(K[2], n2);
		n2 ^= g(K[1], n1);
		n1 ^= g(K[0], n2);
	} 
	{
		n2 ^= g(K[0], n1);
		n1 ^= g(K[1], n2);
		n2 ^= g(K[2], n1);
		n1 ^= g(K[3], n2);
		n2 ^= g(K[4], n1);
		n1 ^= g(K[5], n2);
		n2 ^= g(K[6], n1);
		n1 ^= g(K[7], n2);
	}
	T[gid] = (uint2)(n1, n2);
}
