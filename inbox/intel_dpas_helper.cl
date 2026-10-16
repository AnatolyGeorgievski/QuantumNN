// The code below shows a functional implementation of one of the
// built-in functions added by this extension.  For this built-in
// function:
//  * M = 2, since the result value, a operand, and acc operand
//    are all vectors with two components.
//  * N = 8, and is equal to the subgroup size.
//  * K = 32, as described by the function name.
//  * The elements of both matrix a and matrix b are signed 8-bit
//    integers.

// This is a helper function that performs the dot product of
// two vectors of four components of 8-bit integer data, and then
// adds a 32-bit integer accumulation value.
static int _dp4a( char4 a, char4 b, int acc ){
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w + acc;
}
__attribute__((overloadable))
static float _dp2a( half2 a, half2 b, float acc ){
    return a.x * b.x + a.y * b.y + acc;
}
__attribute__((overloadable))
static int _dp2a( char2 a, char2 b, int acc ){
    return a.x * b.x + a.y * b.y + acc;
}
__attribute__((overloadable))
static int _dp2a( short2 a, short2 b, int acc ){
    return a.x * b.x + a.y * b.y + acc;
}

// This is a helper function that computes the product of a
// 1 x 32 row vector value shared across the subgroup and a 32 x 1
// column vector, that is added to a full precision accumulation
// value.
static int _mma_i8_i8_k32( int v, int8 b, int acc )
{
    // Note: 8 is the size of the subgroup.
    // As K is 32, and the size of the subgroup is 8, each
    // work item contributes 4 elements of the 1 x K vector.
    // as_char4() is used to reinterpret 32-bits of data
    // as four components of 8-bit data.

    int result = acc;
    for (int i= 0; i<8; i++)
        result = _dp4a( as_char4( sub_group_broadcast( v, i ) ), as_char4( b[i] ), result );
    return result;
}
static int _mma_f16_f16_k16( half2 v, int8 b, float acc ){
    float result = acc;
    for (int i= 0; i<8; i++) result += dot(sub_group_broadcast(v,i), as_half2(b[i]));
}
static int _mma_f16_f16_k16_sg16( half v, half16 b, float acc ){
    float result = acc;
    for (int i= 0; i<16; i++) result += sub_group_broadcast(v,i) * b[i];
}

int2 intel_sub_group_i8_i8_matrix_mad_k32(int2  a, int8  b, int2 acc){
    int2 result;
    result.x = _mma_i8_i8_k32( a.x, b, acc.x );
    result.y = _mma_i8_i8_k32( a.y, b, acc.y );
    return result;
}
float2 intel_sub_group_f16_f16_matrix_mad_k16(half2  a, int8  b, float2 acc){
    float2 result;
    result.x = _mma_f16_f16_k16( a.x, b, acc.x );
    result.y = _mma_f16_f16_k16( a.y, b, acc.y );
    return result;
}
// tf32_tf32_matrix_mad_k8