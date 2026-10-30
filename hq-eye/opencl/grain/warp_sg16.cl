/* subgroup probe Intel (sg 16). Буфер data[] длиной >= 512 + sg. Не Adreno. */
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

__attribute__((intel_reqd_sub_group_size(16)))
kernel void warp_sg16(global float *data) {
    uint ix = get_sub_group_local_id();
    half2 x = as_half2(data[get_global_id(0)]);
    half2 y = as_half2(data[get_local_id(0)]);
    float s = 0.f;
    for (uint i = 0u; i < 512u; i += get_max_sub_group_size()) {
        float a = data[i + ix];
        for (uint k = 0u; k < get_max_sub_group_size(); k++) {
            s += dot(as_half2(sub_group_broadcast(a, k)), x);
            s += dot(as_half2(sub_group_broadcast(a, k)), y);
        }
    }
    data[get_global_id(0)] = s;
}
