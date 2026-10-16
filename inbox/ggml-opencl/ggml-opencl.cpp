// id tensor_set CL_BLOCKING mod and ggml_backend_opencl_set_tensor_async:  NON_BLOCKING

//#define GGML_OPENCL_PROFILING 1
#define _TEST 0
//#define TENSOR_FLOW_TEST 0
#define MAX_PROFI_SIZE 1280  //512

#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// suppress warnings in CL headers for GCC and Clang
#pragma GCC diagnostic ignored "-Woverlength-strings"
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"
#endif

#include "ggml-opencl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml.h"

#include <CL/cl.h>

#include <string.h>

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <fstream>
#include <limits>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <charconv>
#include <bits/stdc++.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define UNUSED(x) (void)(x)

#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            printf("ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            GGML_ASSERT(0);                                         \
        }                                                           \
    } while (0)

//------------------------------------------------------------------------------
// OpenCL
//------------------------------------------------------------------------------
// оптимизации по макро операциям:
static void ggml_cl_gga     (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
static void ggml_cl_mva     (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst);
static void ggml_cl_ffn     (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst);
static void ggml_cl_lora    (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * srcA, const ggml_tensor * srcB, ggml_tensor * dst);
static void ggml_cl_rms_norm_mul(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
static bool ggml_cl_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor);

static cl_mem _get_cl_buffer(const ggml_tensor * tensor, cl_mem_flags mem_flags);
static cl_mem _get_cl_buffer_row(const ggml_tensor * tensor, int row, cl_mem_flags mem_flags);
static cl_int _event_list(const ggml_tensor * node, cl_event *event_list);
static cl_int _event_get (const ggml_tensor * node, cl_event *event_list);

static inline int32_t   _tensor_get_param_i32(const ggml_tensor* tensor, int idx );
static inline _Float32  _tensor_get_param_f32(const ggml_tensor* tensor, int idx );
static inline void      _tensor_set_param_f32(      ggml_tensor* tensor, int idx, float value );

enum GPU_FAMILY {
    ADRENO,
    AMD,
    INTEL,
    NVIDIA,
    UNKNOWN,
};


enum ADRENO_GPU_GEN {
    ADRENO_UNKNOWN,
    A7X,
    A8X,
    X1E,
};
enum _EXTENSIONS {
    // 6.2.2.2. Async Work-group Copy Fence
    _khr_byte_addressable_store,    // 6.2.2.3. Byte-Addressable Storage
    _khr_extended_async_copies,     // 6.2.2.6. Extended Async Copy Functions
    _khr_extended_bit_ops,          // 6.2.2.7. Extended Bit Operations: bitfield insert, bitfield extract, bitreverse
    _khr_fp16,                      // 6.2.2.8. Half-Precision Floating-Point
    _khr_fp64,                      // 6.2.2.9. Double-Precision Floating-Point: Intel B580 (1/12)
    _khr_initialize_memory,         // 6.2.2.13. Initializing Memory
    _khr_integer_dot_product,       // 6.2.2.16. Integer Dot Product DP4A instruction
    _khr_kernel_clock,              // 6.2.2.17. Kernel Clock
    _khr_select_fprounding_mode,    // 6.2.2.22. Select Floating-Point Rounding Mode
    _khr_subgroups,                 // 6.2.2.24. Sub-Groups
    _khr_subgroup_ballot,           // 6.2.2.25. Sub-Group Ballots
    _khr_subgroup_clustered_reduce, // 6.2.2.26. Sub-Group Clustered Reductions
    _khr_subgroup_extended_types,   // 6.2.2.27. Sub-Group Extended Types
    _khr_subgroup_non_uniform_arithmetic,// 6.2.2.28. Sub-Group Non-Uniform Arithmetic
    _khr_subgroup_non_uniform_vote, // 6.2.2.29. Sub-Group Non-Uniform Vote and Election Functions
    _khr_subgroup_rotate,           // 6.2.2.30. Sub-Group Rotation
	_khr_subgroup_shuffle,          // 6.2.2.31. Sub-Group General Purpose Shuffles
	_khr_subgroup_shuffle_relative, // 6.2.2.32. Sub-Group Relative Shuffles
    _khr_work_group_uniform_arithmetic, // 6.2.2.33. Work-group Collective Uniform Arithmetic Functions

    _khr_device_uuid,               
    _khr_suggested_local_work_size,
    _khr_3d_image_writes,
    _khr_depth_images,
    _khr_mipmap_image,
    _khr_mipmap_image_writes,
    _khr_image2d_from_buffer,
    // QCOM
    _qcom_ml_ops,
    _qcom_bitreverse,
    _qcom_dot_product8,
    _qcom_bfloat16_product,
    _qcom_filter_bicubic,
    _qcom_vector_image_ops,
    _qcom_reqd_sub_group_size,
    _qcom_accelerated_image_ops,
    _qcom_subgroup_shuffle,

    // AMD, поддерживает subgroups
    _amd_fp64,
    _amd_vec3, 
    _amd_media_ops, 
    _amd_media_ops2,
    _amd_copy_buffer_p2p,
    _amd_device_attribute_query,
    
    // INTEL
    _intel_required_subgroup_size,
    _intel_media_block_io,
    _intel_bfloat16_conversions,// BFloat16 to FP32
    _intel_bit_instructions,    // 
    _intel_command_queue_families,// B580
    _intel_spirv_subgroups,
    _intel_subgroups,
    _intel_subgroups_ballot,
    _intel_subgroup_2d_block_io,// B580
    _intel_subgroup_local_block_io,
    _intel_subgroup_buffer_prefetch,
    _intel_subgroup_extended_block_read, // B580
    _intel_subgroup_extended_block_read_cacheopts,
    _intel_subgroup_half2_placeholder, // half2 reduce
    _intel_subgroup_matrix_multiply_accumulate,// DPAS instruction
    _intel_subgroup_matrix_multiply_accumulate_bf8,
    _intel_subgroup_matrix_multiply_accumulate_tf32,// B580
    _intel_subgroup_split_matrix_multiply_accumulate,// DPASW instruction: A770, ~B580
};
#define EXT(ex) (1ull<<_##ex) 

/*Thanks, I tried this functionality and got the following value: 
CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE = 16 
CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE = 2 
CL_KERNEL_MAX_NUM_SUB_GROUPS = 64 CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT = 1024 1 1 Where the value of CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE is always 16, 
*/
static size_t _get_max_subgroup_size(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,
    const size_t* local_work_size)
{   size_t sgs;
    cl_device_id device;
    CL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL));
    CL_CHECK(clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, work_dim, local_work_size,
        sizeof(size_t), &sgs, NULL)); 
    return sgs;
}

static cl_int SuggestedLocalWorkSize(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    size_t* suggested_local_work_size)
{// максимальная загрузка будет достигаться при local work =512 Threads= x16 (max sub group size)
    cl_int max_EU = 512;//=160 для B580 CL_DEVICE_MAX_COMPUTE_UNITS перенести в конфигурацию, должно быть два параметра для выбора: общее число слайсов, кратность k8, k16, k32, k64...
    size_t max_subgroup_size=suggested_local_work_size[0];
    size_t max_subgroup_count=1;
    size_t max_num_subgroups =1;
    
    cl_device_id device;
    CL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_int), &max_EU, NULL));
    CL_CHECK(clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_MAX_NUM_SUB_GROUPS,    0, nullptr, sizeof(max_num_subgroups), &max_num_subgroups,NULL));
    CL_CHECK(clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, work_dim*sizeof(size_t), global_work_size, sizeof(max_subgroup_size),  &max_subgroup_size, NULL));
    CL_CHECK(clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,    work_dim*sizeof(size_t), global_work_size, sizeof(max_subgroup_count), &max_subgroup_count,NULL));
    CL_CHECK(clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT, sizeof(size_t),       &max_subgroup_count, sizeof(size_t), suggested_local_work_size,NULL));
    if (suggested_local_work_size[0]==0) {
        if (0) max_EU = (max_EU+1)>>1;
        suggested_local_work_size[0] = max_subgroup_size*MIN((max_EU+max_subgroup_size-1)/max_subgroup_size, max_subgroup_count);
    }
//    suggested_local_work_size[0] = ;
/*  suggested_local_work_size[1] = max_subgroup_count;
    suggested_local_work_size[2] = max_subgroup_size; */
    return 0;
    UNUSED(global_work_offset);
}

static ADRENO_GPU_GEN get_adreno_gpu_gen(const char *device_name) {
    if (strstr(device_name, "730") ||
        strstr(device_name, "740") ||
        strstr(device_name, "750")) {
        return ADRENO_GPU_GEN::A7X;
    }

    if (strstr(device_name, "830")) {
        return ADRENO_GPU_GEN::A8X;
    }

    if (strstr(device_name, "X1")) {
        return ADRENO_GPU_GEN::X1E;
    }

    return ADRENO_GPU_GEN::ADRENO_UNKNOWN;
}

static int get_adreno_cl_compiler_version(const char *driver_version) {
    std::string driver_ver_str(driver_version);
    size_t compiler_ver_pos = driver_ver_str.find("E031");
    size_t compiler_ver_len = 13;
    size_t compiler_ver_offset = 5;

    if (compiler_ver_pos == std::string::npos) {
        compiler_ver_pos = driver_ver_str.find("DX");
        if (compiler_ver_pos == std::string::npos) {
            return -1;
        }
        compiler_ver_len = 11;
        compiler_ver_offset = 3;
    }

    std::string compiler_ver_str = driver_ver_str.substr(compiler_ver_pos, compiler_ver_len);
    std::string major_ver_str = compiler_ver_str.substr(compiler_ver_offset, 2);
    return std::atoi(major_ver_str.c_str());
}

// backend device context
struct ggml_backend_opencl_device_context {
    cl_platform_id platform;
    std::string platform_name;

    cl_device_id device;
    std::string device_name;
};
// классификатор из двух параметров
#define K2(t1,td) ((GGML_TYPE_##t1)<<8|(GGML_TYPE_##td))
#define K3(t1,t2,td) ((GGML_TYPE_##t1)<<8|(GGML_TYPE_##t2)<<8|(GGML_TYPE_##td))
/*!

 */
// backend context
struct ggml_backend_opencl_context {
    cl_device_id device;
    std::string device_name;

    std::string driver_version;

    GPU_FAMILY gpu_family;
    ADRENO_GPU_GEN adreno_gen;

    cl_int alignment;
    size_t max_alloc_size;

    uint64_t extensions;// флаги cl_khr_fp16 ..

    int adreno_wave_size;

    cl_context context;
    cl_command_queue queue;

    cl_program program;     // nn - нейросети
    cl_program program_1;
    cl_program program_2;
    cl_program program_mv;  // собраны базовые подпрограммы для умножения и деквантизации матриц
    cl_program program_la;  // собраны базовые подпрограммы линейной алгебры (blas)
    cl_program program_im2col;
// Универсальные ядра
    std::map<enum ggml_unary_op,cl_kernel> kernel_unary;
    std::map<enum ggml_unary_op,cl_kernel> kernel_unary_f16;
    std::map<enum ggml_unary_op,cl_kernel> kernel_unary_bf16;
    std::map<enum ggml_type,    cl_kernel> kernel_get_rows;
    std::map<enum ggml_type,    cl_kernel> kernel_mv;       //!< умножение матрицы на вектор
    std::map<enum ggml_type,    cl_kernel> kernel_mva;      //!< умножение матрицы на вектор (пост операция add)
    std::map<enum ggml_type,    cl_kernel> kernel_ffn;
    std::map<enum ggml_type,    cl_kernel> kernel_attn;
    std::map<enum ggml_type,    cl_kernel> kernel_lora;     //!< low rank adaptation
    std::map<enum ggml_type,    cl_kernel> kernel_lerp;
    std::map<enum ggml_type,    cl_kernel> kernel_quant;    //!< квантизация по времени f32 в тип т.к E4M3, BF16 
    std::map<unsigned int,      cl_kernel> kernel_cpy;      //!< выбор метода по двум аргументам K2(src->type,dst->type)
    std::map<unsigned int,      cl_kernel> kernel_mm;       //!< поиск подходящего метода по классификатору из двух аргументов
    


    cl_kernel kernel_add, kernel_add_row, kernel_add_flat;
    cl_kernel kernel_mul, kernel_mul_row, kernel_mul_flat;
    cl_kernel kernel_scale;
    cl_kernel kernel_sqrt;
    cl_kernel kernel_sqr;
//    cl_kernel kernel_silu, kernel_silu_4;
//    cl_kernel kernel_gelu, kernel_gelu_4;
//    cl_kernel kernel_relu, kernel_relu_4;
    cl_kernel kernel_clamp;
    cl_kernel kernel_norm;
    cl_kernel kernel_rms_norm, kernel_rms_norm_mul;
    cl_kernel kernel_diag_mask_inf, kernel_diag_mask_inf_8;
    cl_kernel kernel_soft_max;
    cl_kernel kernel_soft_max_f16; // f16 по второму аргументу
    cl_kernel kernel_soft_max_bf16;
    cl_kernel kernel_rope_norm_f32, kernel_rope_norm_f16, kernel_rope_neox_f32, kernel_rope_neox_f16;
    cl_kernel kernel_rope_multi_f32, kernel_rope_multi_f16, kernel_rope_vision_f32, kernel_rope_vision_f16;
    cl_kernel kernel_cpy_f16_f16, kernel_cpy_f16_f32;
    cl_kernel kernel_cpy_bf16_f32,kernel_cpy_f32_bf16;
    cl_kernel kernel_cpy_f32_f16, kernel_cpy_f32_f32;
    cl_kernel kernel_mul_mat_f32_f32;
    cl_kernel kernel_mul_mat_tf32_f32;
    cl_kernel kernel_mul_mat_bf16_f32;
    cl_kernel kernel_mul_mat_f16_f16;
    cl_kernel kernel_mul_mat_f16_f32;
    cl_kernel kernel_mul_mat_f16_f32_l4;
    cl_kernel kernel_mul_mat_bf16_f32_l4;
    cl_kernel kernel_mul_mat_f16_f32_1row;
    cl_kernel kernel_mul_mat_q8_0_f32;
	cl_kernel kernel_mul_mat_q4_0_f32;
	cl_kernel kernel_mul_mat_q4_1_f32;

// операции надо синтезировать из файла *_mv.cl по параметрам <MAT_T, VEC_T>, поддерживаемые кванты MAT_T: q4_0 q4_1 q8_0 q8_1 q4_K q5_K q6_K q8_K f16 bf16
    cl_kernel kernel_mul_mv_q4_0_f32;
    cl_kernel kernel_mul_mv_q4_1_f32;
//    cl_kernel kernel_mul_mv_q8_0_f32;
    cl_kernel kernel_mul_mv_q4_K_f32;
    cl_kernel kernel_mul_mv_q5_K_f32;
    cl_kernel kernel_mul_mv_q6_K_f32;
    cl_kernel kernel_mul_mv_bf16_f32;

    // GGML_OPENCL_USE_ADRENO_KERNELS
    cl_kernel kernel_convert_block_q4_0, kernel_restore_block_q4_0, kernel_mul_mat_q4_0_f32_flat;
    cl_kernel kernel_mul_mat_q4_0_f32_8x_flat;
    cl_kernel kernel_convert_block_q4_0_noshuffle, kernel_mul_mat_q4_0_f32_flat_v0,
              kernel_mul_mat_q4_0_f32_flat_img_v0;
    cl_kernel kernel_mul_mat_q4_0_f32_1d_8x_flat, kernel_mul_mat_q4_0_f32_1d_16x_flat;

    cl_kernel kernel_im2col_f32, kernel_im2col_f16;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // Transpose kernels
    cl_program program_transpose_32;
    cl_program program_transpose_32_16;
    cl_program program_transpose_16;
    cl_kernel kernel_transpose_32;
    cl_kernel kernel_transpose_32_16;
    cl_kernel kernel_transpose_16;

    cl_mem A_s_d_max;            // max scale buffer size for transpose
    cl_mem A_q_d_max;            // max weight buffer size for transpose
    cl_mem B_d_max;              // max activation buffer size for transpose

    // Gemm and Gemv related programs, kernels, etc
    cl_program program_CL_gemm;
    cl_program program_CL_gemv_general;
    cl_program program_CL_gemv_4096_1_11008;
    cl_program program_CL_gemv_4096_1_4096;
    cl_program program_CL_gemv_11008_1_4096;
    cl_program program_CL_gemv_32000_1_4096;
    cl_kernel CL_mul_mat_Ab_Bi_8x4;
    cl_kernel CL_mul_mat_vec_q4_0_f32_1d_4x_flat_general;
    cl_kernel CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_11008;
    cl_kernel CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_4096;
    cl_kernel CL_mul_mat_vec_q4_0_f32_1d_4x_flat_11008_1_4096;
    cl_kernel CL_mul_mat_vec_q4_0_f32_1d_4x_flat_32000_1_4096;
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
};

static ggml_backend_device                 g_ggml_backend_opencl_device;
static ggml_backend_opencl_device_context  g_ggml_ctx_dev_main {
    /*.platform         =*/ nullptr,
    /*.platform_nane    =*/ "",
    /*.device           =*/ nullptr,
    /*.device_name      =*/ "",
};

static int ggml_backend_opencl_n_devices = 0;

// Profiling
#ifdef GGML_OPENCL_PROFILING
struct ProfilingInfo {
    std::string op_name;
    std::string kernel_name;
    // Kernel execution time in nanoseconds.
    cl_ulong delay_ns;
    cl_ulong duration_ns;
    // Global and local work sizes.
    size_t global_size[3];
    size_t local_size[3];
    // Op output size.
    size_t output_size[4];
    size_t src1_size[4];
};

std::vector<ProfilingInfo> g_profiling_info;
#endif

inline std::string read_file(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs) {
    return "";
  }
  std::string text;
  ifs.seekg(0, std::ios::end);
  text.resize(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  ifs.read(&text[0], text.size());
  return text;
}

static cl_program build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer, const std::string &compile_opts) {
    cl_program p;
    char *program_log;
    size_t program_size;
    size_t log_size;
    int err;

    program_size = strlen(program_buffer);

    p = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        GGML_LOG_ERROR("OpenCL error creating program");
        exit(1);
    }

    err = clBuildProgram(p, 0, NULL, compile_opts.c_str(), NULL, NULL);
    if(err < 0) {
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        GGML_LOG_ERROR("ggml_opencl: kernel compile error:\n\n%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return p;
}

static ggml_backend_opencl_context * ggml_cl2_init(ggml_backend_dev_t dev) {
// todo убрать
    static bool initialized = false;
    static ggml_backend_opencl_context *backend_ctx = nullptr;

    if (initialized) {
        return backend_ctx;
    }

    ggml_backend_opencl_device_context *dev_ctx = (ggml_backend_opencl_device_context *)dev->context;
    GGML_ASSERT(dev_ctx);
    GGML_ASSERT(dev_ctx->platform == nullptr);
    GGML_ASSERT(dev_ctx->device == nullptr);
    GGML_ASSERT(backend_ctx == nullptr);

    initialized = true;
    backend_ctx = new ggml_backend_opencl_context();
    backend_ctx->gpu_family = GPU_FAMILY::UNKNOWN;

    cl_int err;

#ifdef GGML_OPENCL_PROFILING
    GGML_LOG_INFO("ggml_opencl: OpenCL profiling enabled\n");
#endif

    struct cl_device;
    struct cl_platform {
        cl_platform_id id;
        unsigned number;
        char name[128];
        char vendor[128];
        struct cl_device * devices;
        unsigned n_devices;
        struct cl_device * default_device;
    };

    struct cl_device {
        struct cl_platform * platform;
        cl_device_id id;
        unsigned number;
        cl_device_type type;
        char name[128];
    };

    enum { NPLAT = 16, NDEV = 16 };

    struct cl_platform platforms[NPLAT];
    unsigned n_platforms = 0;
    struct cl_device devices[NDEV];
    unsigned n_devices = 0;
    struct cl_device * default_device = NULL;

    cl_platform_id platform_ids[NPLAT];
    if (clGetPlatformIDs(NPLAT, platform_ids, &n_platforms) != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml_opencl: plaform IDs not available.\n");
        return backend_ctx;
    }

    for (unsigned i = 0; i < n_platforms; i++) {
        struct cl_platform * p = &platforms[i];
        p->number = i;
        p->id = platform_ids[i];
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_NAME, sizeof(p->name), &p->name, NULL));
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_VENDOR, sizeof(p->vendor), &p->vendor, NULL));

        cl_device_id device_ids[NDEV];
        cl_int clGetDeviceIDsError = clGetDeviceIDs(p->id, CL_DEVICE_TYPE_ALL, NDEV, device_ids, &p->n_devices);
        if (clGetDeviceIDsError == CL_DEVICE_NOT_FOUND) {
            p->n_devices = 0;
        } else {
            CL_CHECK(clGetDeviceIDsError);
        }
        p->devices = p->n_devices > 0 ? &devices[n_devices] : NULL;
        p->default_device = NULL;

        for (unsigned j = 0; j < p->n_devices; j++) {
            struct cl_device * d = &devices[n_devices];
            d->number = n_devices++;
            d->id = device_ids[j];
            d->platform = p;
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_NAME, sizeof(d->name), &d->name, NULL));
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_TYPE, sizeof(d->type), &d->type, NULL));

            if (p->default_device == NULL && d->type == CL_DEVICE_TYPE_GPU) {
                p->default_device = d;
            }
        }

        if (default_device == NULL && p->default_device != NULL) {
            default_device = p->default_device;
        }
    }

    if (n_devices == 0) {
        GGML_LOG_ERROR("ggml_opencl: could find any OpenCL devices.\n");
        return backend_ctx;
    }

    char * user_platform_string = getenv("GGML_OPENCL_PLATFORM");
    char * user_device_string = getenv("GGML_OPENCL_DEVICE");
    int user_platform_number = -1;
    int user_device_number = -1;

    unsigned n;
    if (user_platform_string != NULL && sscanf(user_platform_string, " %u", &n) == 1 && n < n_platforms) {
        user_platform_number = (int)n;
    }
    if (user_device_string != NULL && sscanf(user_device_string, " %u", &n) == 1 && n < n_devices) {
        user_device_number = (int)n;
    }
    if (user_platform_number != -1 && user_device_number != -1) {
        cl_platform* platform = &platforms[user_platform_number];
        if ((unsigned)user_device_number >= platform->n_devices) {
            GGML_LOG_ERROR("ggml_opencl: invalid device number %d\n", user_device_number);
            exit(1);
        }
        default_device = &platform->devices[user_device_number];
    } else {

        struct cl_device * selected_devices = devices;
        unsigned n_selected_devices = n_devices;

        if (user_platform_number == -1 && user_platform_string != NULL && user_platform_string[0] != 0) {
            for (unsigned i = 0; i < n_platforms; i++) {
                struct cl_platform * p = &platforms[i];
                if (strstr(p->name, user_platform_string) != NULL ||
                    strstr(p->vendor, user_platform_string) != NULL) {
                    user_platform_number = (int)i;
                    break;
                }
            }
            if (user_platform_number == -1) {
                GGML_LOG_ERROR("ggml_opencl: no platform matching '%s' was found.\n", user_platform_string);
                exit(1);
            }
        }
        if (user_platform_number != -1) {
            struct cl_platform * p = &platforms[user_platform_number];
            selected_devices = p->devices;
            n_selected_devices = p->n_devices;
            default_device = p->default_device;
            if (n_selected_devices == 0) {
                GGML_LOG_ERROR("ggml_opencl: selected platform '%s' does not have any devices.\n", p->name);
                exit(1);
            }
        }

        if (user_device_number == -1 && user_device_string != NULL && user_device_string[0] != 0) {
            for (unsigned i = 0; i < n_selected_devices; i++) {
                struct cl_device * d = &selected_devices[i];
                if (strstr(d->name, user_device_string) != NULL) {
                    user_device_number = d->number;
                    break;
                }
            }
            if (user_device_number == -1) {
                GGML_LOG_ERROR("ggml_opencl: no device matching '%s' was found.\n", user_device_string);
                exit(1);
            }
        }
        if (user_device_number != -1) {
            selected_devices = &devices[user_device_number];
            n_selected_devices = 1;
            default_device = &selected_devices[0];
        }

        GGML_ASSERT(n_selected_devices > 0);

        if (default_device == NULL) {
            default_device = &selected_devices[0];
        }
    }

    GGML_LOG_INFO("ggml_opencl: selecting platform: '%s'\n", default_device->platform->name);
    GGML_LOG_INFO("ggml_opencl: selecting device: '%s'\n", default_device->name);
    if (default_device->type != CL_DEVICE_TYPE_GPU) {
        GGML_LOG_WARN("ggml_opencl: warning, not a GPU: '%s'.\n", default_device->name);
    }

    dev_ctx->platform = default_device->platform->id;
    dev_ctx->device = default_device->id;
    backend_ctx->device = default_device->id;

    if (strstr(default_device->name, "Adreno")) {
        backend_ctx->gpu_family = GPU_FAMILY::ADRENO;
        backend_ctx->adreno_gen = get_adreno_gpu_gen(default_device->name);

        // Default wave size is 128, A8x uses 64.
        if (backend_ctx->adreno_gen == ADRENO_GPU_GEN::A8X) {
            backend_ctx->adreno_wave_size = 64;
        } else if (backend_ctx->adreno_gen == ADRENO_GPU_GEN::A7X ||
                   backend_ctx->adreno_gen == ADRENO_GPU_GEN::X1E) {
            backend_ctx->adreno_wave_size = 128;
        } else {
            backend_ctx->adreno_wave_size = 128;
            GGML_LOG_WARN("ggml_opencl: Unsupported Adreno GPU: %s, "
                "using wave size %d, "
                "may not work as expected\n",
                backend_ctx->device_name.c_str(), backend_ctx->adreno_wave_size);
        }
    } else if (strstr(default_device->name, "Intel")) {
        backend_ctx->gpu_family = GPU_FAMILY::INTEL;
    } else if (strstr(default_device->name, "Advanced")) {
        backend_ctx->gpu_family = GPU_FAMILY::AMD;
    } else {
        GGML_LOG_ERROR("Unsupported GPU: %s\n", default_device->name);
        backend_ctx->gpu_family = GPU_FAMILY::UNKNOWN;
        return backend_ctx;
    }

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (backend_ctx->gpu_family != GPU_FAMILY::ADRENO) {
        GGML_LOG_ERROR("ggml_opencl: Adreno-specific kernels should not be enabled for non-Adreno GPUs; "
            "run on an Adreno GPU or recompile with CMake option `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF`\n");
        return backend_ctx;
    }
#endif

    // Populate backend device name
    dev_ctx->platform_name = default_device->platform->name;
    dev_ctx->device_name = default_device->name;
    backend_ctx->device_name = default_device->name;

    // A local ref of cl_device_id for convenience
    cl_device_id device = backend_ctx->device;

    // Check device OpenCL version, OpenCL 2.0 or above is required
    size_t device_ver_str_size;
    clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &device_ver_str_size);
    char *device_ver_buffer = (char *)alloca(device_ver_str_size + 1);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, device_ver_str_size, device_ver_buffer, NULL);
    device_ver_buffer[device_ver_str_size] = '\0';
    GGML_LOG_INFO("ggml_opencl: device OpenCL version: %s\n", device_ver_buffer);

    if (strstr(device_ver_buffer, "OpenCL 2") == NULL &&
        strstr(device_ver_buffer, "OpenCL 3") == NULL) {
        GGML_LOG_ERROR("ggml_opencl: OpenCL 2.0 or above is required\n");
        return backend_ctx;
    }

    // Check driver version
    size_t driver_version_str_size;
    clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &driver_version_str_size);
    char *driver_version = (char *)alloca(driver_version_str_size + 1);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, driver_version_str_size, driver_version, NULL);
    driver_version[driver_version_str_size] = '\0';
    GGML_LOG_INFO("ggml_opencl: OpenCL driver: %s\n", driver_version);
    backend_ctx->driver_version = driver_version;

    int adreno_cl_compiler_version = get_adreno_cl_compiler_version(driver_version);
    bool has_vector_subgroup_broadcast =
        adreno_cl_compiler_version >= 47 || adreno_cl_compiler_version == 17;
    GGML_LOG_INFO("ggml_opencl: vector subgroup broadcast support: %s\n",
        has_vector_subgroup_broadcast ? "true" : "false");

    size_t ext_str_size;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_str_size);
    char *ext_buffer = (char *)alloca(ext_str_size + 1);
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_str_size, ext_buffer, NULL);
    ext_buffer[ext_str_size] = '\0'; // ensure it is null terminated
    // Check if ext_buffer contains cl_khr_fp16
    #define CASE(feature) \
        (strstr(ext_buffer, "cl_"#feature) != NULL? 1ull<<_##feature : 0)
//    backend_ctx->fp16_support = CASE(khr_fp16);// strstr(ext_buffer, "cl_khr_fp16") != NULL;
    backend_ctx->extensions  = 0;
    backend_ctx->extensions |= CASE(khr_fp16);
    backend_ctx->extensions |= CASE(khr_fp64);
    backend_ctx->extensions |= CASE(khr_subgroups);
    backend_ctx->extensions |= CASE(khr_integer_dot_product);
    if (backend_ctx->gpu_family == INTEL){
        backend_ctx->extensions |= CASE(intel_subgroups);
        backend_ctx->extensions |= CASE(intel_bfloat16_conversions);
        backend_ctx->extensions |= CASE(intel_subgroup_matrix_multiply_accumulate);
        backend_ctx->extensions |= CASE(intel_subgroup_matrix_multiply_accumulate_bf8);
        backend_ctx->extensions |= CASE(intel_subgroup_matrix_multiply_accumulate_tf32);
    }
    #undef  CASE

    GGML_LOG_INFO("ggml_opencl: device FP16 support: %s\n", (backend_ctx->extensions&EXT(khr_fp16)) ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: device FP64 support: %s\n", (backend_ctx->extensions&EXT(khr_fp64)) ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: device BF16 support: %s\n", (backend_ctx->extensions&EXT(intel_bfloat16_conversions)) ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: device FP8  support: %s\n", (backend_ctx->extensions&EXT(intel_subgroup_matrix_multiply_accumulate_bf8)) ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: device TF32 support: %s\n", (backend_ctx->extensions&EXT(intel_subgroup_matrix_multiply_accumulate_tf32)) ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: device DP4A support: %s\n", (backend_ctx->extensions&EXT(khr_integer_dot_product)) ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: device DPAS support: %s\n", (backend_ctx->extensions&EXT(intel_subgroup_matrix_multiply_accumulate)) ? "true" : "false");

    // fp16 is required
    if ((backend_ctx->extensions & EXT(khr_fp16))==0) {
        GGML_LOG_ERROR("ggml_opencl: device does not support FP16\n");
        return backend_ctx;
    }
    if ((backend_ctx->extensions & (EXT(khr_subgroups)|EXT(intel_subgroups)))==0) {
        GGML_LOG_ERROR("ggml_opencl: device does not support khr_subgroups extension\n");
        return backend_ctx;
    }

    // If OpenCL 3.0 is supported, then check for cl_khr_subgroups, which becomes
    // optional in OpenCL 3.0 (cl_khr_subgroup is mandatory in OpenCL 2.x)
    if (strstr(device_ver_buffer, "OpenCL 3") &&
        (backend_ctx->extensions&(EXT(khr_subgroups)|EXT(intel_subgroups)))==0) {
        GGML_LOG_ERROR("ggml_opencl: device does not support subgroups (cl_khr_subgroups or cl_intel_subgroups) "
            "(note that subgroups is an optional feature in OpenCL 3.0)\n");
        return backend_ctx;
    }
    cl_uint address_bits=0;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &address_bits, NULL));
    GGML_LOG_INFO("ggml_opencl: address bits: %u\n", address_bits);
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &backend_ctx->alignment, NULL));
    GGML_LOG_INFO("ggml_opencl: mem base addr align: %u\n", backend_ctx->alignment);

    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &backend_ctx->max_alloc_size, NULL));
    GGML_LOG_INFO("ggml_opencl: max mem alloc size: %zu MB\n", backend_ctx->max_alloc_size/1024/1024);

    // Check SVM.
    cl_device_svm_capabilities svm_caps;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &svm_caps, NULL));
    GGML_LOG_INFO("ggml_opencl: SVM coarse grain buffer support: %s\n",
        svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM fine grain buffer support: %s\n",
        svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM fine grain system support: %s\n",
        svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM atomics support: %s\n",
        svm_caps & CL_DEVICE_SVM_ATOMICS ? "true" : "false");

    // Print out configurations
#ifdef GGML_OPENCL_SOA_Q
    GGML_LOG_INFO("ggml_opencl: flattening quantized weights representation as struct of arrays (GGML_OPENCL_SOA_Q)\n");
#endif // GGML_OPENCL_SOA_Q

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    GGML_LOG_INFO("ggml_opencl: using kernels optimized for Adreno (GGML_OPENCL_USE_ADRENO_KERNELS)\n");
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    cl_context_properties properties[] = {
        (intptr_t)CL_CONTEXT_PLATFORM, (intptr_t)dev_ctx->platform, 0
    };

    CL_CHECK((backend_ctx->context = clCreateContext(properties, 1, &device, NULL, NULL, &err), err));

    // A local ref of cl_context for convenience
    cl_context context = backend_ctx->context;

    //CL_CHECK((queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err),
    //    (err != CL_INVALID_QUEUE_PROPERTIES && err != CL_INVALID_VALUE ? err :
    //    (queue = clCreateCommandQueue(context, device, 0, &err), err)
    //)));
    cl_command_queue_properties command_queue_props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    command_queue_props |= CL_QUEUE_PROFILING_ENABLE;

    CL_CHECK((backend_ctx->queue = clCreateCommandQueue(context, device, command_queue_props, &err), err));

#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src {
        #include "ggml-opencl.cl.h"
    };
    const std::string kernel_la_src {
        #include "ggml-opencl_la.cl.h"
    };
    const std::string kernel_mv_src {
        #include "ggml-opencl_mv.cl.h"
    };
    const std::string kernel_mm_src {
        #include "ggml-opencl_mm.cl.h"
    };
#else
    const std::string kernel_src    = read_file("ggml-opencl.cl");
    const std::string kernel_src    = read_file("ggml-opencl_la.cl");
    const std::string kernel_mv_src = read_file("ggml-opencl_mv.cl");
    const std::string kernel_mm_src = read_file("ggml-opencl_mm.cl");
#endif

    std::string compile_opts =
        "-cl-std=CL3.0 -cl-mad-enable "
        "-cl-unsafe-math-optimizations "
        "-cl-finite-math-only "
        "-cl-fast-relaxed-math "
        ;
    backend_ctx->program    = build_program_from_source(context, device, kernel_src.c_str(),    compile_opts);
    backend_ctx->program_mv = build_program_from_source(context, device, kernel_mv_src.c_str(), compile_opts);
    backend_ctx->program_la = build_program_from_source(context, device, kernel_la_src.c_str(), compile_opts);
    backend_ctx->program_1  = build_program_from_source(context, device, kernel_mm_src.c_str(), compile_opts);

    // Non matmul kernels.

    CL_CHECK((backend_ctx->kernel_get_rows[GGML_TYPE_F32 ] = clCreateKernel(backend_ctx->program, "kernel_get_rows_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_get_rows[GGML_TYPE_BF16] = clCreateKernel(backend_ctx->program, "kernel_get_rows_bf16", &err), err));
    if (backend_ctx->extensions & EXT(intel_bfloat16_conversions)){
        CL_CHECK((backend_ctx->kernel_cpy[K2(BF16,F32)]  = clCreateKernel(backend_ctx->program, "kernel_cpy_bf16_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_bf16_f32       = clCreateKernel(backend_ctx->program, "kernel_cpy_bf16_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_f32_bf16       = clCreateKernel(backend_ctx->program, "kernel_cpy_f32_bf16", &err), err));
    }
    CL_CHECK((backend_ctx->kernel_get_rows[GGML_TYPE_Q4_0]      = clCreateKernel(backend_ctx->program, "kernel_get_rows_q4_0", &err), err));
    CL_CHECK((backend_ctx->kernel_get_rows[GGML_TYPE_Q4_1]      = clCreateKernel(backend_ctx->program, "kernel_get_rows_q4_1", &err), err));
    CL_CHECK((backend_ctx->kernel_get_rows[GGML_TYPE_Q4_K]      = clCreateKernel(backend_ctx->program, "kernel_get_rows_q4_K", &err), err));
//    CL_CHECK((backend_ctx->kernel_get_rows_q5_K      = clCreateKernel(backend_ctx->program, "kernel_get_rows_q5_K", &err), err));
    CL_CHECK((backend_ctx->kernel_get_rows[GGML_TYPE_Q6_K]      = clCreateKernel(backend_ctx->program, "kernel_get_rows_q6_K", &err), err));
    CL_CHECK((backend_ctx->kernel_get_rows[GGML_TYPE_Q8_0]      = clCreateKernel(backend_ctx->program, "kernel_get_rows_q8_0", &err), err));

    CL_CHECK((backend_ctx->kernel_add                = clCreateKernel(backend_ctx->program, "kernel_add", &err), err));
    CL_CHECK((backend_ctx->kernel_add_row            = clCreateKernel(backend_ctx->program, "kernel_add_row", &err), err));
    CL_CHECK((backend_ctx->kernel_add_flat           = clCreateKernel(backend_ctx->program, "kernel_add_flat", &err), err));

    CL_CHECK((backend_ctx->kernel_mul                = clCreateKernel(backend_ctx->program, "kernel_mul", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_row            = clCreateKernel(backend_ctx->program, "kernel_mul_row", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_flat           = clCreateKernel(backend_ctx->program, "kernel_mul_flat", &err), err));

    CL_CHECK((backend_ctx->kernel_scale              = clCreateKernel(backend_ctx->program, "kernel_scale", &err), err));
    CL_CHECK((backend_ctx->kernel_sqrt               = clCreateKernel(backend_ctx->program, "kernel_sqrt", &err), err));
    CL_CHECK((backend_ctx->kernel_sqr                = clCreateKernel(backend_ctx->program, "kernel_op_sqr", &err), err));

//  CL_CHECK((backend_ctx->kernel_silu      = clCreateKernel(backend_ctx->program, "kernel_silu", &err), err));
//  CL_CHECK((backend_ctx->kernel_silu_4    = clCreateKernel(backend_ctx->program, "kernel_silu_4", &err), err));
//  CL_CHECK((backend_ctx->kernel_gelu      = clCreateKernel(backend_ctx->program, "kernel_gelu", &err), err));
//  CL_CHECK((backend_ctx->kernel_gelu_4    = clCreateKernel(backend_ctx->program, "kernel_gelu_4", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_ABS ]       = clCreateKernel(backend_ctx->program, "kernel_fabs", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_NEG ]       = clCreateKernel(backend_ctx->program, "kernel_op_neg", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_ELU ]       = clCreateKernel(backend_ctx->program, "kernel_elu", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_GELU_QUICK] = clCreateKernel(backend_ctx->program, "kernel_gelu_quick", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_GELU]       = clCreateKernel(backend_ctx->program, "kernel_gelu", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_SILU]       = clCreateKernel(backend_ctx->program, "kernel_silu", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_RELU]       = clCreateKernel(backend_ctx->program, "kernel_relu", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_STEP]       = clCreateKernel(backend_ctx->program, "kernel_op_step", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_EXP ]       = clCreateKernel(backend_ctx->program, "kernel_exp", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_SGN ]       = clCreateKernel(backend_ctx->program, "kernel_op_sgn", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_TANH]       = clCreateKernel(backend_ctx->program, "kernel_tanh", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_SIGMOID]    = clCreateKernel(backend_ctx->program, "kernel_sigmoid", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_HARDSIGMOID]= clCreateKernel(backend_ctx->program, "kernel_hardsigmoid", &err), err));
    CL_CHECK((backend_ctx->kernel_unary[GGML_UNARY_OP_HARDSWISH]  = clCreateKernel(backend_ctx->program, "kernel_hardswish", &err), err));
    if (backend_ctx->extensions & EXT(khr_fp16)){
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_ABS ]       = clCreateKernel(backend_ctx->program, "kernel_fabs_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_NEG ]       = clCreateKernel(backend_ctx->program, "kernel_neg_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_ELU ]       = clCreateKernel(backend_ctx->program, "kernel_elu_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_GELU_QUICK] = clCreateKernel(backend_ctx->program, "kernel_gelu_quick_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_GELU]       = clCreateKernel(backend_ctx->program, "kernel_gelu_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_SILU]       = clCreateKernel(backend_ctx->program, "kernel_silu_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_RELU]       = clCreateKernel(backend_ctx->program, "kernel_relu_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_STEP]       = clCreateKernel(backend_ctx->program, "kernel_step_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_EXP ]       = clCreateKernel(backend_ctx->program, "kernel_exp_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_SGN ]       = clCreateKernel(backend_ctx->program, "kernel_sgn_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_TANH]       = clCreateKernel(backend_ctx->program, "kernel_tanh_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_SIGMOID]    = clCreateKernel(backend_ctx->program, "kernel_sigmoid_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_HARDSIGMOID]= clCreateKernel(backend_ctx->program, "kernel_hardsigmoid_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_unary_f16[GGML_UNARY_OP_HARDSWISH]  = clCreateKernel(backend_ctx->program, "kernel_hardswish_f16", &err), err));
    }
    if (backend_ctx->extensions & EXT(khr_fp64)){ // набор операций с двойной точностью
    }
    CL_CHECK((backend_ctx->kernel_clamp              = clCreateKernel(backend_ctx->program, "kernel_clamp", &err), err));
    CL_CHECK((backend_ctx->kernel_norm               = clCreateKernel(backend_ctx->program, "kernel_norm", &err), err));
    CL_CHECK((backend_ctx->kernel_rms_norm           = clCreateKernel(backend_ctx->program, "kernel_rms_norm", &err), err));
    CL_CHECK((backend_ctx->kernel_rms_norm_mul       = clCreateKernel(backend_ctx->program, "kernel_rms_norm_mul", &err), err));
    CL_CHECK((backend_ctx->kernel_diag_mask_inf      = clCreateKernel(backend_ctx->program, "kernel_diag_mask_inf", &err), err));
    CL_CHECK((backend_ctx->kernel_diag_mask_inf_8    = clCreateKernel(backend_ctx->program, "kernel_diag_mask_inf_8", &err), err));
    CL_CHECK((backend_ctx->kernel_soft_max           = clCreateKernel(backend_ctx->program, "kernel_soft_max", &err), err));
//    CL_CHECK((backend_ctx->kernel_soft_max_4         = clCreateKernel(backend_ctx->program, "kernel_soft_max_4", &err), err));
    CL_CHECK((backend_ctx->kernel_rope_norm_f32      = clCreateKernel(backend_ctx->program, "kernel_rope_norm_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_rope_neox_f32      = clCreateKernel(backend_ctx->program, "kernel_rope_neox_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_rope_multi_f32     = clCreateKernel(backend_ctx->program, "kernel_rope_multi_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_rope_vision_f32    = clCreateKernel(backend_ctx->program, "kernel_rope_vision_f32", &err), err));
    if (backend_ctx->extensions & EXT(khr_fp16)){
        CL_CHECK((backend_ctx->kernel_soft_max_f16       = clCreateKernel(backend_ctx->program, "kernel_soft_max_f16", &err), err));
//        CL_CHECK((backend_ctx->kernel_soft_max_4_f16     = clCreateKernel(backend_ctx->program, "kernel_soft_max_4_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_norm_f16      = clCreateKernel(backend_ctx->program, "kernel_rope_norm_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_neox_f16      = clCreateKernel(backend_ctx->program, "kernel_rope_neox_f16", &err), err));
    CL_CHECK((backend_ctx->kernel_rope_multi_f16     = clCreateKernel(backend_ctx->program, "kernel_rope_multi_f16", &err), err));
    CL_CHECK((backend_ctx->kernel_rope_vision_f16    = clCreateKernel(backend_ctx->program, "kernel_rope_vision_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_f16_f16        = clCreateKernel(backend_ctx->program, "kernel_cpy_f16_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_get_rows[GGML_TYPE_F16]       = clCreateKernel(backend_ctx->program, "kernel_get_rows_f16", &err), err));

        CL_CHECK((backend_ctx->kernel_cpy_f16_f32        = clCreateKernel(backend_ctx->program, "kernel_cpy_f16_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_f32_f16        = clCreateKernel(backend_ctx->program, "kernel_cpy_f32_f16", &err), err));
    }
    CL_CHECK((backend_ctx->kernel_cpy_f32_f32        = clCreateKernel(backend_ctx->program, "kernel_cpy_f32_f32", &err), err));

    CL_CHECK((backend_ctx->kernel_soft_max_bf16      = clCreateKernel(backend_ctx->program, "kernel_soft_max_bf16", &err), err));


    // Matmul kernels. 
    CL_CHECK((backend_ctx->kernel_mul_mat_f32_f32        = clCreateKernel(backend_ctx->program, "kernel_mul_mat_f32_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mat_f16_f16        = clCreateKernel(backend_ctx->program, "kernel_mul_mat_f16_f16", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mat_f16_f32_1row   = clCreateKernel(backend_ctx->program, "kernel_mul_mat_f16_f32_1row", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mat_f16_f32        = clCreateKernel(backend_ctx->program, "kernel_mul_mat_f16_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mat_f16_f32_l4     = clCreateKernel(backend_ctx->program, "kernel_mul_mat_f16_f32_l4", &err), err));
    if (backend_ctx->extensions & EXT(intel_bfloat16_conversions)){
        CL_CHECK((backend_ctx->kernel_mul_mat_bf16_f32       = clCreateKernel(backend_ctx->program, "kernel_mul_mat_bf16_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_mul_mat_bf16_f32_l4    = clCreateKernel(backend_ctx->program, "kernel_mul_mat_bf16_f32_l4", &err), err));
    }
    CL_CHECK((backend_ctx->kernel_mul_mat_q8_0_f32       = clCreateKernel(backend_ctx->program, "kernel_mul_mat_q8_0_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32       = clCreateKernel(backend_ctx->program, "kernel_mul_mat_q4_0_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mat_q4_1_f32       = clCreateKernel(backend_ctx->program, "kernel_mul_mat_q4_1_f32", &err), err));

    CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_flat  = clCreateKernel(backend_ctx->program, "kernel_mul_mat_q4_0_f32_flat", &err), err));
    CL_CHECK((backend_ctx->kernel_convert_block_q4_0     = clCreateKernel(backend_ctx->program, "kernel_convert_block_q4_0", &err), err));
    CL_CHECK((backend_ctx->kernel_restore_block_q4_0     = clCreateKernel(backend_ctx->program, "kernel_restore_block_q4_0", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_8x_flat = clCreateKernel(backend_ctx->program, "kernel_mul_mat_q4_0_f32_8x_flat", &err), err));


// vector Matmul _mv_qn 
    CL_CHECK((backend_ctx->kernel_ffn[GGML_TYPE_Q8_0]              = clCreateKernel(backend_ctx->program_mv,"ffn_q8_0_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_ffn[GGML_TYPE_F16 ]              = clCreateKernel(backend_ctx->program_mv,"ffn_f16_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_ffn[GGML_TYPE_Q4_K]              = clCreateKernel(backend_ctx->program_1, "kernel_ffn_q4_K_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_ffn[GGML_TYPE_Q5_K]              = clCreateKernel(backend_ctx->program_1, "kernel_ffn_q5_K_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_ffn[GGML_TYPE_Q6_K]              = clCreateKernel(backend_ctx->program_1, "kernel_ffn_q6_K_f32", &err), err));
// matrix vector matmul is_contiguous is_mastrix(a) !is_permuted
    CL_CHECK((backend_ctx->kernel_mv [GGML_TYPE_Q8_0]              = clCreateKernel(backend_ctx->program_mv, "mv_q8_0_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mv [GGML_TYPE_BF16]              = clCreateKernel(backend_ctx->program_mv, "mv_bf16_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mv [GGML_TYPE_F16 ]              = clCreateKernel(backend_ctx->program_mv, "mv_f16_f32",  &err), err));
    CL_CHECK((backend_ctx->kernel_mv [GGML_TYPE_F32 ]              = clCreateKernel(backend_ctx->program_mv, "mv_f32_f32",  &err), err));

//    CL_CHECK((backend_ctx->kernel_mva[GGML_TYPE_F32 ]              = clCreateKernel(backend_ctx->program, "mva_f32_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mva[GGML_TYPE_F16 ]              = clCreateKernel(backend_ctx->program_mv, "mva_f16_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mva[GGML_TYPE_BF16]              = clCreateKernel(backend_ctx->program_mv, "mva_bf16_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mva[GGML_TYPE_Q8_0]              = clCreateKernel(backend_ctx->program_mv, "mva_q8_0_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mva[GGML_TYPE_Q4_K]              = clCreateKernel(backend_ctx->program_1, "kernel_mul_mva_q4_K_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mva[GGML_TYPE_Q5_K]              = clCreateKernel(backend_ctx->program_1, "kernel_mul_mva_q5_K_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mva[GGML_TYPE_Q6_K]              = clCreateKernel(backend_ctx->program_1, "kernel_mul_mva_q6_K_f32", &err), err));

//    CL_CHECK((backend_ctx->kernel_mva[GGML_TYPE_Q6_K]              = clCreateKernel(backend_ctx->program_mv, "mva_q6_k_f32", &err), err));
//    CL_CHECK((backend_ctx->kernel_mul_mv_q8_0_f32                  = clCreateKernel(backend_ctx->program_1, "kernel_mul_mv_q8_0K_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mv_q4_0_f32                  = clCreateKernel(backend_ctx->program_1, "kernel_mul_mv_q4_0_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mv_q4_K_f32                  = clCreateKernel(backend_ctx->program_1, "kernel_mul_mv_q4_K_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mv_q5_K_f32                  = clCreateKernel(backend_ctx->program_1, "kernel_mul_mv_q5_K_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mv_q6_K_f32                  = clCreateKernel(backend_ctx->program_1, "kernel_mul_mv_q6_K_f32", &err), err));

    CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_1d_8x_flat      = clCreateKernel(backend_ctx->program_1, "kernel_mul_mat_q4_0_f32_1d_8x_flat", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_1d_16x_flat     = clCreateKernel(backend_ctx->program_1, "kernel_mul_mat_q4_0_f32_1d_16x_flat", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_flat_v0         = clCreateKernel(backend_ctx->program_1, "kernel_mul_mat_q4_0_f32_flat_v0", &err), err));
    CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_flat_img_v0     = clCreateKernel(backend_ctx->program_1, "kernel_mul_mat_q4_0_f32_flat_img_v0", &err), err));

    // Load additional data conversion kernels.
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src_2 {
        #include "ggml-opencl_cvt.cl.h"
    };
#else
    const std::string kernel_src_2 = read_file("ggml-opencl_cvt.cl");
#endif
    backend_ctx->program_2 = build_program_from_source(context, device, kernel_src_2.c_str(), compile_opts);

    CL_CHECK((backend_ctx->kernel_convert_block_q4_0_noshuffle     = clCreateKernel(backend_ctx->program_2, "kernel_convert_block_q4_0_noshuffle", &err), err));

    // im2col kernels
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src_im2col {
        #include "ggml-opencl_im2col.cl.h"
    };
#else
    const std::string kernel_src_im2col = read_file("ggml-opencl_im2col.cl");
#endif
    backend_ctx->program_im2col = build_program_from_source(context, device, kernel_src_im2col.c_str(), compile_opts);

    CL_CHECK((backend_ctx->kernel_im2col_f32 = clCreateKernel(backend_ctx->program_im2col, "kernel_im2col_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_im2col_f16 = clCreateKernel(backend_ctx->program_im2col, "kernel_im2col_f16", &err), err));

    // Kernels for Adreno
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string transpose_32_src {
        #include "ggml-opencl_transpose_32.cl.h"
    };
#else
    const std::string transpose_32_src = read_file("ggml-opencl_transpose_32.cl");
#endif
    backend_ctx->program_transpose_32 = build_program_from_source(context, device, transpose_32_src.c_str(), compile_opts);
    CL_CHECK((backend_ctx->kernel_transpose_32 = clCreateKernel(backend_ctx->program_transpose_32, "kernel_transpose_32", &err), err));

#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string transpose_32_16_src {
        #include "ggml-opencl_transpose_32_16.cl.h"
    };
#else
    const std::string transpose_32_16_src = read_file("ggml-opencl_transpose_32_16.cl");
#endif
    backend_ctx->program_transpose_32_16 = build_program_from_source(context, device, transpose_32_16_src.c_str(), compile_opts);
    CL_CHECK((backend_ctx->kernel_transpose_32_16 = clCreateKernel(backend_ctx->program_transpose_32_16, "kernel_transpose_32_16", &err), err));

#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string transpose_16_src {
        #include "ggml-opencl_transpose_16.cl.h"
    };
#else
    const std::string transpose_16_src = read_file("ggml-opencl_transpose_16.cl");
#endif
    backend_ctx->program_transpose_16 = build_program_from_source(context, device, transpose_16_src.c_str(), compile_opts);
    CL_CHECK((backend_ctx->kernel_transpose_16 = clCreateKernel(backend_ctx->program_transpose_16, "kernel_transpose_16", &err), err));

    // Gemv general
    std::string CL_gemv_compile_opts =
        " -cl-std=CL2.0 "
        " -cl-mad-enable "
        " -DSIMDGROUP_WIDTH=" + std::to_string(backend_ctx->adreno_wave_size);
    if (has_vector_subgroup_broadcast) {
        CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
    }
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src_CL_gemv_general {
        #include "ggml-opencl_gemv_noshuffle_general.cl.h"
    };
#else
    const std::string kernel_src_CL_gemv_general = read_file("ggml-opencl_gemv_noshuffle_general.cl");
#endif

    backend_ctx->program_CL_gemv_general = build_program_from_source(
        context, device, kernel_src_CL_gemv_general.c_str(), CL_gemv_compile_opts);
    CL_CHECK((backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_general = clCreateKernel(backend_ctx->program_CL_gemv_general, "kernel_gemv_noshuffle", &err), err));

    // Gemv 2048, 16384
    CL_gemv_compile_opts =
        " -cl-std=CL2.0 "
        " -cl-mad-enable "
        " -DLINE_STRIDE_A=2048 "
        " -DBLOCK_STRIDE_A=16384 "
        " -DSIMDGROUP_WIDTH=" + std::to_string(backend_ctx->adreno_wave_size);
    if (has_vector_subgroup_broadcast) {
        CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
    }
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src_CL_gemv {
        #include "ggml-opencl_gemv_noshuffle.cl.h"
    };
#else
    const std::string kernel_src_CL_gemv = read_file("ggml-opencl_gemv_noshuffle.cl");
#endif

    backend_ctx->program_CL_gemv_4096_1_4096 = build_program_from_source(
        context, device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
    CL_CHECK((backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_4096 = clCreateKernel(backend_ctx->program_CL_gemv_4096_1_4096, "kernel_gemv_noshuffle", &err), err));

    // Gemv 2048, 16384
    CL_gemv_compile_opts =
        " -cl-std=CL2.0 "
        " -cl-mad-enable "
        " -DLINE_STRIDE_A=2048 "
        " -DBLOCK_STRIDE_A=16384 "
        " -DSIMDGROUP_WIDTH=" + std::to_string(backend_ctx->adreno_wave_size);
    if (has_vector_subgroup_broadcast) {
        CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
    }

    backend_ctx->program_CL_gemv_4096_1_11008 = build_program_from_source(
        context, device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
    CL_CHECK((backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_11008 = clCreateKernel(backend_ctx->program_CL_gemv_4096_1_11008, "kernel_gemv_noshuffle", &err), err));

    // Gemv 5504, 44032
    CL_gemv_compile_opts =
        " -cl-std=CL2.0 "
        " -cl-mad-enable "
        " -DLINE_STRIDE_A=5504 "
        " -DBLOCK_STRIDE_A=44032 "
        " -DSIMDGROUP_WIDTH=" + std::to_string(backend_ctx->adreno_wave_size);
    if (has_vector_subgroup_broadcast) {
        CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
    }

    backend_ctx->program_CL_gemv_11008_1_4096 = build_program_from_source(
        context, device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
    CL_CHECK((backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_11008_1_4096 = clCreateKernel(backend_ctx->program_CL_gemv_11008_1_4096, "kernel_gemv_noshuffle", &err), err));

    // Gemv 16000, 128000
    CL_gemv_compile_opts =
        " -cl-std=CL2.0 "
        " -cl-mad-enable "
        " -DLINE_STRIDE_A=16000 "
        " -DBLOCK_STRIDE_A=128000 "
        " -DSIMDGROUP_WIDTH=" + std::to_string(backend_ctx->adreno_wave_size);
    if (has_vector_subgroup_broadcast) {
        CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
    }

    backend_ctx->program_CL_gemv_32000_1_4096 = build_program_from_source(context, device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
    CL_CHECK((backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_32000_1_4096 = clCreateKernel(backend_ctx->program_CL_gemv_32000_1_4096, "kernel_gemv_noshuffle", &err), err));

    // Gemm
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src_CL_gemm {
        #include "ggml-opencl_mul_mat_Ab_Bi_8x4.cl.h"
    };
#else
    const std::string kernel_src_CL_gemm = read_file("ggml-opencl_mul_mat_Ab_Bi_8x4.cl");
#endif
    backend_ctx->program_CL_gemm = build_program_from_source(context, device, kernel_src_CL_gemm.c_str(), compile_opts);
    CL_CHECK((backend_ctx->CL_mul_mat_Ab_Bi_8x4 = clCreateKernel(backend_ctx->program_CL_gemm, "kernel_mul_mat_Ab_Bi_8x4", &err), err));

    // Allocate intermediate buffers and images
    size_t max_A_q_d_bytes = 311164928;
    size_t max_A_s_d_bytes = 38895616;
    size_t max_B_d_bytes = 45088768;

    CL_CHECK((backend_ctx->A_q_d_max = clCreateBuffer(context, 0, max_A_q_d_bytes, NULL, &err), err));
    CL_CHECK((backend_ctx->A_s_d_max = clCreateBuffer(context, 0, max_A_s_d_bytes, NULL, &err), err));
    CL_CHECK((backend_ctx->B_d_max   = clCreateBuffer(context, 0, max_B_d_bytes,   NULL, &err), err));
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    // For now we support a single devices
    ggml_backend_opencl_n_devices = 1;

    return backend_ctx;
}

static void __attribute__((destructor)) ggml_cl2_free(void) {
#ifdef GGML_OPENCL_PROFILING
    FILE * fperf = fopen("cl_profiling.csv", "w");
    if (!fperf) {
        GGML_LOG_ERROR("Failed to open cl_profiling.csv\n");
        return;
    }

    float total_kernel_time = 0;
    float total_delays_time = 0;
    fprintf(fperf, "op name, kernel name, duration (ms), global size, local size, output size\n");
    for (const ProfilingInfo & info : g_profiling_info) {
        total_delays_time += info.delay_ns;
        total_kernel_time += info.duration_ns/1.e6f;
        fprintf(fperf, "%s,%s,%f,%zux%zux%zu,%zux%zux%zu,%zux%zux%zux%zu\n",
            info.op_name.c_str(), info.kernel_name.c_str(), info.duration_ns/1.e6f,
            info.global_size[0], info.global_size[1], info.global_size[2],
            info.local_size[0], info.local_size[2], info.local_size[2],
            info.output_size[0], info.output_size[1], info.output_size[2], info.output_size[3]);
    }
    fclose(fperf);

    GGML_LOG_INFO("ggml_opencl: total kernel  time: %f\n", total_kernel_time);
    GGML_LOG_INFO("ggml_opencl: total waiting time: %f\n", total_delays_time);
#endif
}

//------------------------------------------------------------------------------
// Tensor extra management
//------------------------------------------------------------------------------
struct ggml_tensor_extra_cl {
    cl_event evt;// событие по которому можно синхронизовать элементы в очереди команд
    // The buffer object that holds the data.
    cl_mem data_device;
    // The offset into the buffer object. This is primarily for scratch buffer
    // and view operation.
    // NB: this offset no longer includes view offset (view_offs). Whenever this
    // offset is used, view_offs should be considered.
    cl_ulong offset;
    // The actual size of the cl_mem object. This is needed when returning the
    // block to the pool.
    size_t actual_size;

    void reset() {
        evt = nullptr;
        data_device = nullptr;
        offset = 0;
        actual_size = 0;
    }
};

// Additional tensor extra structs for quantized tensors.
// These tensors are loaded from files and should not be allocated in scratch --
// they should always be allocated from the pool. Hence, they do not have an
// `offset`, which indicate their locations in the scratch buffer.
struct ggml_tensor_extra_cl_q4_0 {
    cl_event evt;// событие по которому можно синхронизовать элементы в очереди команд
    // Quantized values.
    cl_mem q = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem q_img = nullptr;
    // Scales.
    cl_mem d = nullptr;
    // Scales in image1d_buffer_t.
    cl_mem d_img = nullptr;
    // Size of quantized values.
    size_t size_q = 0;
    // Size of scales.
    size_t size_d = 0;

    ~ggml_tensor_extra_cl_q4_0() {
        reset();
    }

    void reset() {
        evt = nullptr;
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        // Currently, q_img and d_img are only initialized when SMALL_ALLOC is
        // enabled. They point to the images in ggml_backend_opencl_buffer_context.
        // So, there is no need to release them here.
        // TODO: initialize them for non SMALL_PATH path, or remove them.
        q_img = nullptr;
        d_img = nullptr;
        size_q = 0;
        size_d = 0;
    }
};

//------------------------------------------------------------------------------
// Backend API
//------------------------------------------------------------------------------

//
// backend
//
static const char * ggml_backend_opencl_name(ggml_backend_t backend) {
    return "OpenCL";

    UNUSED(backend);
}

static void ggml_backend_opencl_free(ggml_backend_t backend) {
    ggml_cl2_free();

    GGML_UNUSED(backend);
}
// todo
static void ggml_backend_opencl_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
    GGML_ASSERT(extra);
    CL_CHECK(clEnqueueWriteBuffer(
        queue, extra->data_device, CL_NON_BLOCKING, extra->offset + offset,
        size, data, 0, NULL, &extra->evt));

printf("ASYNC SET %s\n", tensor->name);
}
// todo
static void ggml_backend_opencl_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static bool ggml_backend_opencl_cpy_tensor_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(backend);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}
// todo
static void ggml_backend_opencl_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
}
/*! \brief НАходит источник в графе, для отображения зависимостей
    \param arg_idx для аргумента с индексом 
    \return relative position 1..i
 */
static int _get_offset(ggml_cgraph * cgraph, int i, int arg_idx){
    int offs=0;
    int N = MIN(128,i);
    ggml_tensor * node = cgraph->nodes[i];
    if (node->src[arg_idx]!=nullptr){
        for (offs=1;offs<=N; offs++)
            if (node->src[arg_idx] ==cgraph->nodes[i-offs]) break;
        if(offs>N) offs=0;//(cgraph->nodes[0] - node->src[arg_idx])/GGML_TENSOR_SIZE;
    }
    return offs;
}
/*! \brief добавить событие связаное с готовностью тензора в список ожидания
 */
static cl_int _event_get(const ggml_tensor * node, cl_event *event_list){
    cl_int event_list_len = 0;
    ggml_tensor_extra_cl* extra = (ggml_tensor_extra_cl*)node->extra;
    GGML_ASSERT(extra!=nullptr);
    if (extra->evt!=nullptr)
        event_list[event_list_len++] = extra->evt;//_get_event(node->src[k]);
    return event_list_len;
}
/*! \brief создает список ожидания по аргументам тензора */
static cl_int _event_list(const ggml_tensor * node, cl_event *event_list){
    cl_int event_list_len = 0;
    for (int k=0;k<GGML_MAX_SRC && node->src[k]!=nullptr;++k){
        ggml_tensor_extra_cl* extra = (ggml_tensor_extra_cl*)node->src[k]->extra;
        GGML_ASSERT(extra!=nullptr);
        if (extra->evt!=nullptr)
            event_list[event_list_len++] = extra->evt;//_get_event(node->src[k]);
    }
    return event_list_len;
}

static cl_long _event_measure(cl_event evt, int64_t &delay_ns, int64_t &duration, uint64_t &start_ns)
{
    if (evt==nullptr) return 0;
    cl_ulong submit;
    cl_ulong queued;
    cl_ulong start;
    cl_ulong end;
    cl_ulong complete;
    (clWaitForEvents(1, &evt));

    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_QUEUED,  sizeof(cl_ulong), &submit, NULL));
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_QUEUED,  sizeof(cl_ulong), &queued, NULL));
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,   sizeof(cl_ulong), &start, NULL));
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,     sizeof(cl_ulong), &end, NULL));
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_COMPLETE,sizeof(cl_ulong), &complete, NULL));

    start_ns = start;
//    CL_CHECK(clReleaseEvent(evt));
    delay_ns = start-queued;   // разница возникает, когда применяетcя ожидание события
//    delay_ns = queued-submit;   // разница возникает, когда применяетcя ожидание события
    duration = end - start; // время на исполнение
    return duration;
}
static ggml_status ggml_backend_opencl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {

static int once = 0;
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
if (!once){
    // if (cgraph->n_leafs) ggml_build_forward_expand(cgraph, outp);
    //ggml_graph_print(cgraph); 
    once = 1;
}
    if(0)  printf("%d/%d:-----------------------------------------------------------------\n", cgraph->n_nodes, cgraph->n_leafs);
    uint32_t optimizations[4]={0};
    uint64_t total_min=__LONG_LONG_MAX__;
    uint64_t total_max=0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
if (0) {// optimizations
        if (i+3<cgraph->n_nodes)
            if (node->op==GGML_OP_MUL_MAT && cgraph->nodes[i+1]->op==GGML_OP_UNARY && cgraph->nodes[i+2]->op==GGML_OP_MUL_MAT && cgraph->nodes[i+3]->op==GGML_OP_MUL
            && node->src[1]==cgraph->nodes[i+2]->src[1] && backend_ctx->kernel_ffn[node->src[0]->type]!=nullptr){
                //printf("OPTIMIZE FFn (MM|MM)x\n");
                GGML_ASSERT(node->src[1]->type==GGML_TYPE_F32);

                ggml_cl_ffn(backend, node->src[0], cgraph->nodes[i+2]->src[0], node->src[1], cgraph->nodes[i+3]);
                i+=3;
                optimizations[1]++;
                continue;
            }
        if (i+1<cgraph->n_nodes) // последовательность rms_norm(x)*y -- "нор-муль"
            if (node->op==GGML_OP_RMS_NORM && cgraph->nodes[i+1]->op== GGML_OP_MUL && (cgraph->nodes[i+1]->src[0]==node)){
                GGML_ASSERT(cgraph->nodes[i+1]->ne[0]==node->ne[0]);
                GGML_ASSERT(cgraph->nodes[i+1]->src[1]->ne[1]==node->ne[1] || node->ne[1]%cgraph->nodes[i+1]->src[1]->ne[1]==0); // -- может быть равно flat или картно
                GGML_ASSERT(node->src[0]->type==GGML_TYPE_F32);
                GGML_ASSERT(cgraph->nodes[i+1]->src[0]->type==GGML_TYPE_F32);
                GGML_ASSERT(cgraph->nodes[i+1]->src[1]->type==GGML_TYPE_F32);

                ggml_cl_rms_norm_mul(backend, node->src[0], cgraph->nodes[i+1]->src[1], cgraph->nodes[i+1]);
                i++;
                optimizations[2]++;
                continue;
            }
        if (i+2<cgraph->n_nodes) // norm(x)*y+b -- "нор-мул-а"
            if (node->op==GGML_OP_NORM && cgraph->nodes[i+1]->op== GGML_OP_MUL  && cgraph->nodes[i+2]->op== GGML_OP_ADD && (cgraph->nodes[i+1]->src[0]==node)){
            }

        if (i+1<cgraph->n_nodes) // последовательность Ax+y - аффинные преобразования
            if (node->op==GGML_OP_MUL_MAT && cgraph->nodes[i+1]->op== GGML_OP_ADD && (cgraph->nodes[i+1]->src[0]==node/* || cgraph->nodes[i+1]->src[1]==node*/) 
            &&  node->ne[1]==node->src[1]->ne[1]
            &&  backend_ctx->kernel_mva[node->src[0]->type]!=nullptr
            ){
                GGML_ASSERT(cgraph->nodes[i+1]->ne[0]==node->ne[0]);
                GGML_ASSERT(cgraph->nodes[i+1]->src[1]->ne[1]==node->ne[1] || node->ne[1]%cgraph->nodes[i+1]->src[1]->ne[1]==0); // -- может быть равно flat или картно
                GGML_ASSERT(node->src[1]->type==GGML_TYPE_F32);
                GGML_ASSERT(cgraph->nodes[i+1]->src[0]->type==GGML_TYPE_F32);
                GGML_ASSERT(cgraph->nodes[i+1]->src[1]->type==GGML_TYPE_F32);
                GGML_ASSERT(cgraph->nodes[i+1]->src[1]->ne[0] == cgraph->nodes[i+1]->src[0]->ne[0]);
                // добавить параметр scale factor = 1
                _tensor_set_param_f32(cgraph->nodes[i+1], 0, 1.f);
                ggml_cl_mva(backend, node->src[0], node->src[1], cgraph->nodes[i+1]->src[1], cgraph->nodes[i+1]);
                i++;
                optimizations[0]++;
                continue;
            }
        
        if (i+2<cgraph->n_nodes) // последовательность \alpha Ax+y -- аффинные преобразования
            if (node->op==GGML_OP_MUL_MAT && cgraph->nodes[i+1]->op== GGML_OP_SCALE && cgraph->nodes[i+2]->op== GGML_OP_ADD 
            && (cgraph->nodes[i+1]->src[0]==node && cgraph->nodes[i+2]->src[0]==cgraph->nodes[i+1]) 
            ){
#if 0
                float scale = _tensor_get_param_f32(cgraph->nodes[i+2], 0);
                _tensor_set_param_f32(cgraph->nodes[i+3], 0, scale);
                ggml_cl_lora(backend, node->src[0], node->src[1], cgraph->nodes[i+2]->src[1], cgraph->nodes[i+3]);
                i+=3;
                continue;
#endif
            }
        if (i+1<cgraph->n_nodes) // последовательность y = \alpha BAx+y -- LoRa
            if (node->op==GGML_OP_MUL_MAT && cgraph->nodes[i+1]->op== GGML_OP_MUL_MAT && cgraph->nodes[i+2]->op== GGML_OP_SCALE  && cgraph->nodes[i+3]->op== GGML_OP_ADD) {
                i+=3;
                continue;
            }
        if (i+2<cgraph->n_nodes && 0) // операция с кешем надо чтобы работало с множеством строк. может некорректно работаеть если не скаляр
            if (cgraph->nodes[i+0]->op==GGML_OP_GET_ROWS &&  cgraph->nodes[i+1]->op==GGML_OP_GET_ROWS && cgraph->nodes[i+2]->op==GGML_OP_ADD
            &&  cgraph->nodes[i+2]->src[0] == cgraph->nodes[i+0]
            &&  cgraph->nodes[i+2]->src[1] == cgraph->nodes[i+1] 
            &&  ggml_is_scalar(cgraph->nodes[i+0]->src[1]) && ggml_is_scalar(cgraph->nodes[i+1]->src[1])
           // &&  cgraph->nodes[i+0]->src[1] == cgraph->nodes[i+1]->src[1] // общий аргумент ?
            ){// если одна строка, то копировать не надо? надо считать offset
                GGML_ASSERT(ggml_is_vector(cgraph->nodes[i+0]));
                GGML_ASSERT(ggml_is_vector(cgraph->nodes[i+1]));
                _tensor_set_param_f32(cgraph->nodes[i+2], 0, 1.f);
                ggml_cl_gga(backend, cgraph->nodes[i+2]->src[0], cgraph->nodes[i+2]->src[1], cgraph->nodes[i+2]);
                optimizations[3]++;
                i+=2;
                continue;
            }
        if (i+2<cgraph->n_nodes && 1) // операция с кешем
            if (cgraph->nodes[i+0]->op==GGML_OP_MUL_MAT &&  cgraph->nodes[i+1]->op==GGML_OP_SOFT_MAX && cgraph->nodes[i+2]->op==GGML_OP_MUL_MAT
            &&  cgraph->nodes[i+0]->src[0]->op==GGML_OP_VIEW && cgraph->nodes[i+2]->src[0]->op==GGML_OP_VIEW
            ){// матрицы могут считаться параллельно
#if 0
                struct ggml_tensor* t0 = cgraph->nodes[i+0];
                struct ggml_tensor* t2 = cgraph->nodes[i+2];
                struct ggml_tensor* s0 = cgraph->nodes[i+0]->src[0];
                struct ggml_tensor* s2 = cgraph->nodes[i+0]->src[1];
if (1) printf("MM %s(%zu,%zu,%zu)(%zu,%zu,%zu)%c %s(%zu,%zu,%zu)(%zu,%zu,%zu)%c\n", 
    s0->name, s0->ne[0], s0->ne[1], s0->ne[2], s0->nb[0], s0->nb[1], s0->nb[2], ggml_is_contiguous(s0)?'c':'n',
    s2->name, s2->ne[0], s2->ne[1], s2->ne[2], s2->nb[0], s2->nb[1], s2->nb[2], ggml_is_contiguous(s2)?'c':'n'
    );
    s0 = cgraph->nodes[i+2]->src[0];
    s2 = cgraph->nodes[i+2]->src[1];
if (1) printf("MM %s(%zu,%zu,%zu)(%zu,%zu,%zu)%c %s(%zu,%zu,%zu)(%zu,%zu,%zu)%c\n", 
    s0->name, s0->ne[0], s0->ne[1], s0->ne[2], s0->nb[0], s0->nb[1], s0->nb[2], ggml_is_contiguous(s0)?'c':'n',
    s2->name, s2->ne[0], s2->ne[1], s2->ne[2], s2->nb[0], s2->nb[1], s2->nb[2], ggml_is_contiguous(s2)?'c':'n'
    );

if (1) printf("OPTIMIZE KV-cache %s(%zu,%zu,%zu)(%zu,%zu,%zu)%c %s(%zu,%zu,%zu)(%zu,%zu,%zu)%c\n", 
    cgraph->nodes[i+0]->name, t0->ne[0], t0->ne[1], t0->ne[2],t0->nb[0], t0->nb[1], t0->nb[2], ggml_is_contiguous(t0)?'c':'n',
    cgraph->nodes[i+2]->name, t2->ne[0], t2->ne[1], t2->ne[2],t2->nb[0], t2->nb[1], t2->nb[2], ggml_is_contiguous(t2)?'c':'n'
    );
#endif
            }
}// optimizations
        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            // GGML_ASSERT(node->src[0]!=nullptr);

            if (node->op == GGML_OP_VIEW && 0 )
                printf("\tOP VIEW %-15.15s base=%p offs=%6zu size=(%zu,%zu,%zu) ld=(%zu,%zu,%zu) %s\n", node->name, 
                    (char*)node->data-node->view_offs,
                    node->view_offs, node->ne[0],node->ne[1],node->ne[2], node->nb[0],node->nb[1],node->nb[2], ggml_is_transposed(node)?"(transposed)":"" );

            if (node->src[0]!=nullptr) {
                ggml_tensor_extra_cl* extra = (ggml_tensor_extra_cl*)node->extra;
                ggml_tensor_extra_cl* extra0 = (ggml_tensor_extra_cl*)node->src[0]->extra;
                // todo разрезание модели приводит к ошибке extra==null
                GGML_ASSERT(extra!=nullptr);
                if (extra->evt!=nullptr){
                    extra->evt = extra0->evt;
                    CL_CHECK(clRetainEvent(extra->evt));// The event reference count is incremented. 
                }
            }
            continue;
        }
        if (node->op == GGML_OP_CPY && 0)
            printf("\tOP COPY %-15.15s base=%p offs=%6zu size=(%zu,%zu,%zu)\n", node->name, 
                        (char*)node->data-node->view_offs,
                        node->view_offs, node->src[1]->ne[0],node->src[1]->ne[1],node->src[1]->ne[2]);
/*  Команда VIEW должна ставить буфер из кеша в очередь 
    Команда CPY  должна 
 */
        bool ok = ggml_cl_compute_forward(backend, node);// &events[idx++]
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

// освободить все события и завершить все 
//    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    cl_event event_list[GGML_MAX_SRC]; UNUSED(event_list);
    int seq =0; UNUSED(seq);
    int64_t kernel_time=0;
    for (int i = 0; i < cgraph->n_nodes; i++){
//        if ((i&0xFF)==0x00)  CL_CHECK(clFinish(backend_ctx->queue));
        
        ggml_tensor * node = cgraph->nodes[i];
        ggml_tensor_extra_cl* extra = (ggml_tensor_extra_cl*)node->extra;
        int64_t delay=0, duration=0;
        uint64_t start=0;
        if (extra->evt!=nullptr) {
            int res = (clWaitForEvents(1, &extra->evt));
            if (res!=CL_SUCCESS){
                char buf[128];
                sprintf(buf, "EVT %s %s %d", ggml_op_name(node->op), node->name, res);
                puts(buf);
                clReleaseEvent(extra->evt); extra->evt=0;
            }
        }
        kernel_time += _event_measure(extra->evt, delay, duration, start);
        if (start!=0){
            if(start<total_min) total_min = start;
            if(start>total_max) total_max = start;
        }
#if defined(TENSOR_FLOW_TEST) && _TEST>1// отладка 
        int k=0;// число аргументов
        cl_int event_list_len = 0;
        for (k=0;k<GGML_MAX_SRC && node->src[k]!=nullptr;++k){
            int offs = _get_offset(cgraph, i, k);
            if(offs!=0) {
                ggml_tensor_extra_cl* extra = (ggml_tensor_extra_cl*)node->src[k]->extra;
                GGML_ASSERT(extra!=nullptr);
                if (extra->evt!=nullptr)
                    event_list[event_list_len++] = extra->evt;//_get_event(node->src[k]);
            }
        }
        int arg0 = _get_offset(cgraph, i, 0);
        int arg1 = _get_offset(cgraph, i, 1);
        // раскраска последовательностей команд
        if (((arg0==0||node->src[0]->op==GGML_OP_VIEW) && arg1==1) 
        ||  ((arg1==0||node->src[1]->op==GGML_OP_VIEW) && arg0==1)){
            seq++;
        } else {
            if (node->op==GGML_OP_GET_ROWS)    printf("\033[%dm",31);// один аргумент
            else if ((arg0==0) && node->src[1]!=NULL && arg1>1) printf("\e[%dm",arg1<=seq?34:35);// два агрумента out-of-order execution
            else if (arg0 >1 && node->src[1]==NULL) printf("\033[%dm",32);// один агрумент out-of-order execution
            else if (arg0==0 && node->src[1]==NULL) printf("\033[%dm",36);// один аргумент команда типа ViEW - работа с кешем
            else if (node->op==GGML_OP_CPY && node->src[1]->op==GGML_OP_VIEW)    printf("\033[%dm",33);// Запись в кеш
            else {
                printf("\033[%dm", 37);
            }
            seq=0;
        }
        printf ("%3d:[%6.3f,%6.3f] ", i, delay/1e6f, duration/1e6f);
        printf ("%-9.9s ",/*"%4s ",  node->name, */
            node->op==GGML_OP_UNARY? ggml_unary_op_name(ggml_get_unary_op(node)):ggml_op_name(node->op)/* , ggml_type_name(node->type) */);
        for (int argc=0; argc<k; argc++){
            int offs = _get_offset(cgraph, i, argc);
            printf ("%c%d.%-4s(%4zux%zux%zu),\t",offs!=0?'r':'m', offs, ggml_type_name(node->src[argc]->type),
            node->src[argc]->ne[0], node->src[argc]->ne[1], node->src[argc]->ne[2] ); 
        }
        cl_uint refs=0;
        if (extra->evt!=nullptr)
            CL_CHECK(clGetEventInfo(extra->evt, CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &refs, NULL));
        printf("\t%4zux%zux%-3zu\t;[%d]%-23.23s", node->ne[0], node->ne[1], node->ne[2],  refs, node->name);


        for (int j=0;j<event_list_len; ++j)
            printf(":%d", event_list[j]);
        printf("\n");
#endif



        if (extra!=nullptr && extra->evt!=nullptr){
            CL_CHECK(clReleaseEvent(extra->evt)); extra->evt= nullptr;
        }
    }
#if defined(_TEST) && _TEST>0
    printf("\e[0m kernel_time=%4.1f ms, total = %4.1f ms\n", kernel_time/(1.e6f), (total_max-total_min)*1.e-6f);
#endif

    return GGML_STATUS_SUCCESS;
}

static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (0) printf(" OP %s %s %s\n", ggml_op_name(op->op), op->src[0]?ggml_type_name(op->src[0]->type):"", op->src[1]?ggml_type_name(op->src[1]->type):"");

    switch (op->op) {
        case GGML_OP_NONE:
            return true;
        case GGML_OP_GET_ROWS:
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_BF16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q8_1:
                case GGML_TYPE_Q8_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q4_0:
					return ggml_is_matrix(op->src[0]) && ggml_is_contiguous(op->src[0]);
#ifdef GGML_OPENCL_SOA_Q
                    // We do not support flattened Q4_0 (and possibly other Q's)
                    return false;
#else // GGML_OPENCL_SOA_Q
                    return true;
#endif // GGML_OPENCL_SOA_Q
                default:
                    return false;
            }
            return true;
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                    switch (op->type) {
                        case GGML_TYPE_F16:
                        case GGML_TYPE_F32:
                        case GGML_TYPE_BF16:
                        case GGML_TYPE_Q8_0:
                        case GGML_TYPE_Q8_1:
                        case GGML_TYPE_Q8_K:
                            return true;
                        default:
                            return false;
                    }
                case GGML_TYPE_F16:
                    switch (op->type) {
                        case GGML_TYPE_F16:
                        case GGML_TYPE_F32:
                            return true;
                        default:
                            return false;
                    }
                case GGML_TYPE_BF16:
                    switch (op->type) {
                        case GGML_TYPE_BF16:
                        case GGML_TYPE_F32:
                            return true;
                        default:
                            return false;
                    }
                default:
                    return false;
            }
        case GGML_OP_ADD:
        case GGML_OP_SCALE:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
            return true;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_CLAMP:
        case GGML_OP_UPSCALE:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return op->type == GGML_TYPE_F32 && true;
        case GGML_OP_MUL_MAT:
            if (op->src[0]->type == GGML_TYPE_F32 ||
                op->src[0]->type == GGML_TYPE_F16 ||
                op->src[0]->type == GGML_TYPE_BF16) {
                return true;
            } else if (
                       op->src[0]->type == GGML_TYPE_Q4_0 ||
                       op->src[0]->type == GGML_TYPE_Q4_1 ||
                       op->src[0]->type == GGML_TYPE_Q5_0 ||
                       op->src[0]->type == GGML_TYPE_Q5_1 ||
                       op->src[0]->type == GGML_TYPE_Q8_0 ||
					   op->src[0]->type == GGML_TYPE_Q4_K ||
                       op->src[0]->type == GGML_TYPE_Q5_K ||
                       op->src[0]->type == GGML_TYPE_Q6_K) {
                return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            }
            return false;
        case GGML_OP_MUL_MAT_ID:
            return false;
        case GGML_OP_OUT_PROD: // todo \see blas
            return true;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return op->ne[3] == 1;
        case GGML_OP_ROPE: {
            const int mode = ((const int32_t *) op->op_params)[2];
            if (mode & GGML_ROPE_TYPE_MROPE) {
                return true;
            }
            if (mode & GGML_ROPE_TYPE_VISION) {
                printf ("rope vision\n");
                return false;
            }
            return true;
        }
        case GGML_OP_CONCAT:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_FLASH_ATTN_EXT: // TODO
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
            return false;
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_POOL_2D_BACK:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_SILU_BACK:
        case GGML_OP_SOFT_MAX_BACK:
            return false;
        default:
			if (1) printf(" UNSUPPORTED OP %s %s %s\n", ggml_op_name(op->op), op->src[0]?ggml_type_name(op->src[0]->type):"", op->src[1]?ggml_type_name(op->src[1]->type):"");
_Exit(1);
            return false;
    }
}

// Forward declaration - implementation appears later in the file.
static const char * ggml_backend_opencl_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type);

static ggml_guid_t ggml_backend_opencl_guid() {
    static ggml_guid guid = { 0xde, 0xe0, 0x70, 0xa2, 0x73, 0x4e, 0x4d, 0xbc, 0xb0, 0xc7, 0x4f, 0xd4, 0x6d, 0x4e, 0x90, 0xfe };
    return &guid;
}

static ggml_backend_i ggml_backend_opencl_i = {
    /* .get_name                = */ ggml_backend_opencl_name,
    /* .free                    = */ ggml_backend_opencl_free,
    /* .set_tensor_async        = */ ggml_backend_opencl_set_tensor_async,  /* ggml_backend_opencl_set_tensor_async */
    /* .get_tensor_async        = */ NULL,  /* ggml_backend_opencl_get_tensor_async */
    /* .cpy_tensor_async        = */ NULL,  /* ggml_backend_opencl_cpy_tensor_async */
    /* .synchronize             = */ ggml_backend_opencl_synchronize,  /* ggml_backend_opencl_synchronize */
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_opencl_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

ggml_backend_t ggml_backend_opencl_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_opencl_reg(), 0);
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(dev);

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_opencl_guid(),
        /* .interface = */ ggml_backend_opencl_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx
    };

    return backend;
}

bool ggml_backend_is_opencl(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_opencl_name;
}

//
// buffer
//
struct ggml_backend_opencl_buffer_context {
    // A buffer context can hold multiple cl_mem objects. This is for flattening
    // quantized weights and should be used with GGML_OPENCL_SMALL_ALLOC where
    // each tensor is allocated a separate buffer. When flattening is enabled
    // with small allocation, each tensor is backed by two cl_mem objects (for
    // quants and scales) packed into a backend_opencl_buffer.
    ggml_backend_opencl_buffer_context(cl_mem buf)
        : name("OpenCL") {
        buffer.push_back(buf);
    }

    ~ggml_backend_opencl_buffer_context() {
        for (cl_mem buf : buffer) {
            CL_CHECK(clReleaseMemObject(buf));
        }
        for (cl_mem im : img) {
            CL_CHECK(clReleaseMemObject(im));
        }

        // Delete all extras to trigger their destructors
        for (ggml_tensor_extra_cl * e : temp_tensor_extras) {
            delete e;
        }
        for (ggml_tensor_extra_cl * e : temp_tensor_extras_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0_in_use) {
            delete e;
        }
    }

    ggml_tensor_extra_cl * ggml_opencl_alloc_temp_tensor_extra() {
        ggml_tensor_extra_cl * extra;
        if (temp_tensor_extras.empty()) {
            extra = new ggml_tensor_extra_cl();
        } else {
            extra = temp_tensor_extras.back();
            temp_tensor_extras.pop_back();
        }

        temp_tensor_extras_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q4_0 * ggml_opencl_alloc_temp_tensor_extra_q4_0() {
        ggml_tensor_extra_cl_q4_0 * extra;
        if (temp_tensor_extras_q4_0.empty()) {
            extra = new ggml_tensor_extra_cl_q4_0();
        } else {
            extra = temp_tensor_extras_q4_0.back();
            temp_tensor_extras_q4_0.pop_back();
        }

        temp_tensor_extras_q4_0_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    void reset() {
        for (ggml_tensor_extra_cl * e : temp_tensor_extras_in_use) {
            temp_tensor_extras.push_back(e);
        }
        temp_tensor_extras_in_use.clear();

        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0_in_use) {
            temp_tensor_extras_q4_0.push_back(e);
        }
        temp_tensor_extras_q4_0_in_use.clear();
    }

    // Pools for extras. Available extras are in `temp_tensor_extras`. Extras
    // being used are in `temp_tensor_extras_in_use`. At the first run, new
    // extras get created and put in `in_use`. When the buffer is reset via
    // the `reset` callback, all extras in `in_use` get moved to available extras
    // for reuse.
    std::vector<ggml_tensor_extra_cl *> temp_tensor_extras;
    std::vector<ggml_tensor_extra_cl *> temp_tensor_extras_in_use;
    std::vector<ggml_tensor_extra_cl_q4_0 *> temp_tensor_extras_q4_0;
    std::vector<ggml_tensor_extra_cl_q4_0 *> temp_tensor_extras_q4_0_in_use;

    // The buffer_context is initially created by ggml_backend_buft_alloc_buffer
    // before any tensor is initialized (at the beginning of alloc_tensor_range).
    // Hence, there is alway a buffer object in this vector. When each tensor is
    // being initialized, this original buffer object will be released if both
    // flattening and small allocation are enabled, and additional buffer
    // objects will be created in init_tensor to represent flattened quantized
    // weights.
    std::vector<cl_mem> buffer;
    // These are image1d_buffer_t objects that wrap around the quants and scales.
    // For Q4_0 quantization, there should be two of them - one for quants and
    // one for scales. They should be populated only when flattening and small
    // allocation are enabled.
    std::vector<cl_mem> img;
    std::string name;
};

static void * const cl_ptr_base = (void *)(uintptr_t) 0x1000;

static void ggml_backend_opencl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_opencl_buffer_get_base(ggml_backend_buffer_t buffer) {
    return cl_ptr_base;

    GGML_UNUSED(buffer);
}

static enum ggml_status ggml_backend_opencl_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;

    ggml_cl2_init(buffer->buft->device);

    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);

        ggml_tensor_extra_cl * view_extra = (ggml_tensor_extra_cl *) tensor->view_src->extra;
        GGML_ASSERT(view_extra && "view_extra is nullptr?");

        // Reuse extra of the parent tensor. The offset of this view tensor
        // becomes `extra->offset + view_offs` and needs to be calculated when
        // it is used. This changes is needed because of the change to
        // ggml_alloc.c in https://github.com/ggerganov/llama.cpp/pull/7640.
        // `buffer` passed in here will always be `tensor->buffer`. It is OK
        // to allocate extras from the same buffer context for ordinary
        // intermediate tensors. But for views into kv cache tensors, doing so
        // would mess up the extras used by kv cache.
        // Before #7640, `buffer` is for intermediate tensors, which is always
        // different from that of kv cache tensors.
        //
        // NB: now extra->offset no longer accounts for view_offs.
        // NB: this should not apply to weight tensors (for end-to-end runs, but
        //     may apply for test-backend-ops).
        // FIXME: if any unexpected results are seen, double check the offset -
        // there could be other places that need fix.
        tensor->extra = view_extra;
#if defined(_TEST) && _TEST>3
        //size_t offset = (char *)tensor->data - (char *)cl_ptr_base;
        printf("VIEW  %-9.9s %-32.32s %8zx %8zx (%zu,%zu,%zu)\n", ggml_op_name(tensor->op), tensor->name, view_extra->offset, view_extra->actual_size,
        tensor->ne[0],tensor->ne[1],tensor->ne[2] );
#endif
    } else {
        size_t offset = (char *)tensor->data - (char *)cl_ptr_base;
#if defined(_TEST) && _TEST>3
        printf("ALLOC %-9.9s %-32.32s %8zx %8zx (%zu,%zu,%zu)\n", ggml_op_name(tensor->op), tensor->name, offset, ggml_nbytes(tensor),
        tensor->ne[0],tensor->ne[1],tensor->ne[2] );
#endif
        ggml_tensor_extra_cl * extra = ctx->ggml_opencl_alloc_temp_tensor_extra();
        GGML_ASSERT(extra->evt == nullptr);// все события освобождаются когда завершается цикл расчета графа
        extra->evt = nullptr;
        extra->offset = offset;
        extra->data_device = ctx->buffer[0];// почему 0?
        extra->actual_size = ggml_nbytes(tensor);

        tensor->extra = extra;
    }
    return GGML_STATUS_SUCCESS;
}

// The optimized gemm and gemv kernels are used for large matrices without batch.
// tensor is the quantized weights matrix.
inline bool use_adreno_kernels(const ggml_tensor *tensor) {
    return tensor->ne[0] >= 512 && tensor->ne[1] >= 512 &&
            tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static void ggml_backend_opencl_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(buffer->buft->device);

    cl_context context = backend_ctx->context;
    cl_command_queue queue = backend_ctx->queue;

#ifdef GGML_OPENCL_SOA_Q
    // We separate the quantized bits and scale from block_q4_0 by using an
    // additional kernel, where each thread handles a block. We first read the
    // original weights into a temporary buffer, then create two separate
    // buffers for quantized bits and scales, which are then populated by the
    // conversion kernel.
    if (tensor->type == GGML_TYPE_Q4_0) {
        // Tensors should have been preallocated, therefore they should
        // already have ggml_tensor_extra_cl as extra.
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q4_0 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q4_0();

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(
            queue, data_device, CL_TRUE, 0,
            ggml_nbytes(tensor), data, 0, NULL, &extra->evt));

        // We consider the specified offset arg as always, although For weights
        // the offset arg should be 0 (we do not assert this).
        //GGML_ASSERT(offset == 0);

        // We create subbuffers from the original tensor buffer for scales and
        // quants - i.e., scales and quants are aliases into the buffer obejct
        // that backs the original tensor. This is a cleaner way to adapt to the
        // new memory management.
        // In the old code, we allocate new buffers for scales and quants
        // respectively, which could still be done but would result in double
        // allocation; properly deallocating the preallocated buffer that backs
        // the tensors is tricky and would leak the backend specific information
        // into the general backend code.
        // Does this create misaligned subbuffers (alignment is 1024) in certain
        // cases ?
        cl_buffer_region region;

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        // Create subbuffer for scales.
        region.origin = extra_orig->offset + tensor->view_offs + offset;
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

        // Create subbuffer for quants.
        region.origin = extra_orig->offset + tensor->view_offs + offset + size_d;
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

        //cl_kernel kernel = backend_ctx->kernel_convert_block_q4_0;
    #ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        cl_kernel kernel = backend_ctx->kernel_convert_block_q4_0;

        // The optimized kernels need weights in natural order, so unshuffle.
        if (use_adreno_kernels(tensor)) {
            kernel = backend_ctx->kernel_convert_block_q4_0_noshuffle;
        }
    #else
        cl_kernel kernel = backend_ctx->kernel_convert_block_q4_0;
    #endif // GGML_OPENCL_USE_ADRENO_KERNELS
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra = extra;

        // transpose the weights and scales
    #ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Only do transpose for large, non batched matrix
        // TODO: use preallocated images instead of sub-buffer then image
        if (use_adreno_kernels(tensor)) {
        // <----------------------------------------------------------------------------------> //
        // start transpose
        // <----------------------------------------------------------------------------------> //
        int M = tensor->ne[1];   // ne01
        int K = tensor->ne[0];   // ne00

        //For matrix-vector multiplication kernel, we assume K is a multiple of 32
        GGML_ASSERT(K % 32 == 0);
        //For transpose kernels, we assume K is a multiple of 4 (satisfied by prior assert), and M is a multiple of 4
        GGML_ASSERT(M % 4 == 0);

        // transpose is out of place, so we need to allocate transposed buffers
        // <----------------------------------------------------------------------------------> //
        // use sub_buffer of max buffer size instead

        size_t q_size_bytes = K * M / 8 * sizeof(float);
        cl_buffer_region region;
        region.origin = 0;
        region.size = q_size_bytes;
        cl_mem qT_d = clCreateSubBuffer(
            backend_ctx->A_q_d_max,
            0,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &err);
        // cl_mem qT_d = clCreateBuffer(context, CL_MEM_READ_WRITE, q_size_bytes, NULL, &err);
        CL_CHECK(err);

        // size_t d_size_bytes = M * (K / 32) / 2 * sizeof(float);
        size_t d_size_bytes = M * (K / 32) * 2;
        region.origin = 0;
        region.size = d_size_bytes;
        cl_mem dT_d = clCreateSubBuffer(
            backend_ctx->A_s_d_max,
            0,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &err);
        // cl_mem dT_d = clCreateBuffer(context, CL_MEM_READ_WRITE, d_size_bytes, NULL, &err);
        CL_CHECK(err);

        // <----------------------------------------------------------------------------------> //


        // create images from the buffers
        // <----------------------------------------------------------------------------------> //
        cl_mem q_d_image1D;
        cl_mem d_d_image1D;
        cl_mem qT_d_image1D;
        cl_mem dT_d_image1D;

        cl_image_format img_fmt_1d = { CL_RGBA, CL_HALF_FLOAT };
        cl_image_desc img_desc_1d;

        memset(&img_desc_1d, 0, sizeof(img_desc_1d));
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = M * K / 4 / 4;
        img_desc_1d.buffer = extra->q;
        q_d_image1D = clCreateImage(context, 0, &img_fmt_1d, &img_desc_1d, NULL, &err);
        CL_CHECK(err);

        img_fmt_1d = { CL_RGBA, CL_HALF_FLOAT };
        memset(&img_desc_1d, 0, sizeof(img_desc_1d));
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = M * K / 4 / 4;
        img_desc_1d.buffer = qT_d;
        qT_d_image1D = clCreateImage(context, 0, &img_fmt_1d, &img_desc_1d, NULL, &err);
        CL_CHECK(err);

        img_fmt_1d = { CL_RGBA, CL_HALF_FLOAT };
        memset(&img_desc_1d, 0, sizeof(img_desc_1d));
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = M * K / 32 / 4;
        img_desc_1d.buffer = extra->d;
        d_d_image1D = clCreateImage(context, 0, &img_fmt_1d, &img_desc_1d, NULL, &err);
        CL_CHECK(err);

        img_fmt_1d = { CL_RGBA, CL_HALF_FLOAT };
        memset(&img_desc_1d, 0, sizeof(img_desc_1d));
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = M * K / 32 / 4;
        img_desc_1d.buffer = dT_d;
        dT_d_image1D = clCreateImage(context, 0, &img_fmt_1d, &img_desc_1d, NULL, &err);
        CL_CHECK(err);
        // <----------------------------------------------------------------------------------> //

        // set up and call the transpose kernels
        // <----------------------------------------------------------------------------------> //
        // weights
        int height_q = M / 4;
        int width_q = K / 4 / 4;
        kernel = backend_ctx->kernel_transpose_16;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &q_d_image1D));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &qT_d_image1D));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_q));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_q));

        size_t local_size_q[3] = {4, 16, 1};
        size_t global_size_q[3] = {static_cast<size_t>(width_q), static_cast<size_t>(height_q), 1};
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size_q, local_size_q, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));

        // scales
        int height_s = M / 4;
        int width_s = K / 32 / 4;

        kernel = backend_ctx->kernel_transpose_16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_d_image1D));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dT_d_image1D));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &height_s));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &width_s));

        size_t local_size_s[3] = {4, 16, 1};
        size_t global_size_s[3] = {static_cast<size_t>(width_s), static_cast<size_t>(height_s), 1};
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size_s, local_size_s, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        // <----------------------------------------------------------------------------------> //

        // copy transposed buffer contents to original buffers
        // <----------------------------------------------------------------------------------> //
        // weights
        CL_CHECK(clEnqueueCopyBuffer(queue, qT_d, extra->q, 0, 0, q_size_bytes, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));

        // scales
        CL_CHECK(clEnqueueCopyBuffer(queue, dT_d, extra->d, 0, 0, d_size_bytes, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        // <----------------------------------------------------------------------------------> //

        // deallocate transpose buffers
        // <----------------------------------------------------------------------------------> //
        CL_CHECK(clReleaseMemObject(qT_d));
        CL_CHECK(clReleaseMemObject(dT_d));

        // deallocate temporary images
        CL_CHECK(clReleaseMemObject(q_d_image1D));
        CL_CHECK(clReleaseMemObject(d_d_image1D));
        CL_CHECK(clReleaseMemObject(qT_d_image1D));
        CL_CHECK(clReleaseMemObject(dT_d_image1D));
        // <----------------------------------------------------------------------------------> //
        // end transpose
        // <----------------------------------------------------------------------------------> //
        }
    #endif // GGML_OPENCL_USE_ADRENO_KERNELS

        return;
    }
#endif // GGML_OPENCL_SOA_Q

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
    GGML_ASSERT(extra);
#if defined(_TEST) && _TEST>2
    const char* units="";
    uint64_t scale = 1;
    if (size<8*1024) { units=" B"; scale =1; }
    else if (size<8*1024*1024) { units="kB"; scale =1024; }
    else { units="Mb"; scale = 1024*1024; }

printf("SET %-4s %5zd%s\t; %s\t\t\n", ggml_type_name(tensor->type), size/scale,units,  tensor->name);
    if (tensor->type == GGML_TYPE_I32){ // оптимизация, не надо заводить буфер ради этого
        for (int i=0;i<GGML_MAX_OP_PARAMS/4 && i<size/4; ++i){
            tensor->op_params[i] = ((int32_t*)(data+offset))[i];
            //printf("%d:VAL %d\n", i, tensor->op_params[i], tensor->name);
        }
    }
#endif

// Буферы бывают с квантизованными типами и нет, буферы бывают READ_ONLY, READ_WRITE (SHARED)
/*    if (size>64*1024*1024){
        extra->evt = nullptr;
        CL_CHECK(clEnqueueWriteBuffer(
            queue, extra->data_device, CL_BLOCKING, extra->offset + offset,
            size, data, 0, NULL, NULL));
    } else */
        CL_CHECK(clEnqueueWriteBuffer(
        queue, extra->data_device, CL_NON_BLOCKING, extra->offset + offset,
        size, data, 0, NULL, &extra->evt));

    GGML_UNUSED(buffer);
}

static void ggml_backend_opencl_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->extra);

    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(buffer->buft->device);

    cl_context context = backend_ctx->context;
    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
    // Make sure all previously submitted commands are finished.
    if (extra->evt==nullptr) 
        CL_CHECK(clFinish(queue));

#ifdef GGML_OPENCL_SOA_Q
    // In end-to-end runs, get_tensor is usually used to get back the logits,
    // where we can simply do clEnqueueReadBuffer since they are f32.
    // However, in test-backend-ops, the GPU graph is copied to the CPU backend,
    // which requires reading back quantized weight tensors.
    // To properly support this, we need to restore block_q4_0 struct arrays
    // from the flattened buffers.
    if (tensor->type == GGML_TYPE_Q4_0) {
        ggml_tensor_extra_cl_q4_0 * extra = (ggml_tensor_extra_cl_q4_0 *)tensor->extra;

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->kernel_restore_block_q4_0;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseEvent(evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));// это связанные события надо ждать второе
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
#endif // GGML_OPENCL_SOA_Q
    cl_event evt;
    CL_CHECK(clEnqueueReadBuffer(
        queue, extra->data_device, CL_BLOCKING, extra->offset + tensor->view_offs + offset,
        size, data, extra->evt==nullptr?0:1, &extra->evt, &evt));
    // CL_CHECK(clWaitForEvents(1, &evt)); -- блокирующий вызов не требует ожидания
#if defined(_TEST) && _TEST>1
    const char* units="";
    uint64_t scale = 1;
    if (size<8*1024) { units=" B"; scale =1; }
    else if (size<8*1024*1024) { units="kB"; scale =1024; }
    else { units="Mb"; scale = 1024*1024; }
    int64_t delay=0, duration=0;
    uint64_t start=0;
    cl_long total_time=0;
    total_time += _event_measure(evt, delay, duration, start);
    printf ("%3d:[%6.3f,%6.3f] ", 0, delay/1e6f, duration/1e6f);
// анализировать формат данных

    printf("GET %-4s(%zu,%zu,%zu) %5zd%s\t; %s\n", ggml_type_name(tensor->type), 
        tensor->ne[0],tensor->ne[1],tensor->ne[2],
        size/scale,units,  
        tensor->name);
    if(0)for (int k = 0; k< tensor->ne[0] && k<16; k++){
        printf ("%f ", ((float*)data)[k]);
    }
    printf ("\n");
#endif

    CL_CHECK(clReleaseEvent(evt));

    GGML_UNUSED(buffer);
}

static void ggml_backend_opencl_buffer_memset    (ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {

    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(buffer->buft->device);

    cl_command_queue queue = backend_ctx->queue;
    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;

    GGML_ASSERT(extra);
    CL_CHECK(clEnqueueFillBuffer(queue, extra->data_device, &value, sizeof(value), 
        extra->offset + tensor->view_offs + offset, size, extra->evt==nullptr?0:1, &extra->evt, &extra->evt));

}
static void ggml_backend_opencl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_dev_t dev = buffer->buft->device;
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(dev);
    cl_command_queue queue = backend_ctx->queue;

    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    for (const cl_mem buf : ctx->buffer) {
        CL_CHECK(clEnqueueFillBuffer(queue, buf, &value, sizeof(value), 0, buffer->size, 0, NULL, NULL));
    }
    CL_CHECK(clFinish(queue));
}

static void ggml_backend_opencl_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    ctx->reset();
}

static ggml_backend_buffer_i ggml_backend_opencl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_opencl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_opencl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_opencl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_opencl_buffer_memset,
    /* .set_tensor      = */ ggml_backend_opencl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_opencl_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_opencl_buffer_clear,
    /* .reset           = */ ggml_backend_opencl_buffer_reset,
};

//
// buffer type
//

static const char * ggml_backend_opencl_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type) {
    return "OpenCL";

    GGML_UNUSED(buffer_type);
}
/*
Функиця вызывается несколько раз для каждого типа буфера (сегмента), например:
model   4955 MB = может быть разбит на несколько буферов
KV cache 224 MB -- размер зависит от окна контекста (4096), при окна удвоении удваивается размер кэша
Compute  304 MB -- размер зависит от окна контекста, при удвоении 492 (размер увеличивается почти линейно) 
*/
static ggml_backend_buffer_t ggml_backend_opencl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buffer_type, size_t size) {
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(buffer_type->device);

    // clCreateBuffer returns -61 for size 0
    size = std::max(size, (size_t)0x1000);

    char buf[128];
    sprintf(buf, "ALLOC_BUF %.2f MB", size / 1024.0 / 1024.0);// выделяется буфер три сегмента: под константы - матрицы, переменные и kv-кеш. 
    puts(buf);
    cl_int err;

    cl_mem mem = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_INFO("%s: failed to allocate %.2f MiB\n", __func__, size / 1024.0 / 1024.0);
        return nullptr;
    }

    ggml_backend_opencl_buffer_context * ctx = new ggml_backend_opencl_buffer_context(mem);

    return ggml_backend_buffer_init(buffer_type, ggml_backend_opencl_buffer_interface, ctx, size);
}

static size_t ggml_backend_opencl_buffer_type_get_alignment(ggml_backend_buffer_type_t buffer_type) {
    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer_type->device);
    return backend_ctx->alignment;
}

static size_t ggml_backend_opencl_buffer_type_get_max_size(ggml_backend_buffer_type_t buffer_type) {
    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer_type->device);
    return backend_ctx->max_alloc_size;
}
static size_t ggml_backend_opencl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buffer_type, const struct ggml_tensor * tensor) {
    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer_type->device);
    size_t size = ggml_nbytes(tensor);
    if (ggml_is_quantized(tensor->type)){
        // для квантованных типов рассчитать вместе с выравниванием строк 
    }
    return GGML_PAD(size, backend_ctx->alignment);
}

static bool ggml_backend_opencl_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_opencl(backend);

    UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_opencl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_opencl_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_opencl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_opencl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_opencl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_opencl_buffer_type_get_alloc_size,// (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
    /* .is_host          = */ NULL,// (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
};

ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type() {
    static ggml_backend_buffer_type buffer_type = {
        /* .iface   = */ ggml_backend_opencl_buffer_type_interface,
        /* .device  = */ &g_ggml_backend_opencl_device,
        /* .context = */ nullptr,
    };

    return &buffer_type;
}

//
// backend device
//

static const char * ggml_backend_opencl_device_get_name(ggml_backend_dev_t dev) {
    return "GPU OpenCL";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_opencl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_opencl_device_context *dev_ctx = (ggml_backend_opencl_device_context *) dev->context;
    return dev_ctx->device_name.c_str();
}

static void ggml_backend_opencl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
//    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(dev);

    *free = 1;
    *total = 1;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_opencl_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_opencl_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_opencl_device_get_name(dev);
    props->description = ggml_backend_opencl_device_get_description(dev);
    props->type        = ggml_backend_opencl_device_get_type(dev);
    ggml_backend_opencl_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = ggml_backend_dev_caps {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_opencl_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(dev);

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_opencl_guid(),
        /* .interface = */ ggml_backend_opencl_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_opencl_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_opencl_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_opencl_device_buffer_from_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool ggml_backend_opencl_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    bool res = ggml_opencl_supports_op(dev, op);
	if (!res) 
        if(1) printf(" UNSUPPORTED OP %s %s %s\n", ggml_op_name(op->op), op->src[0]?ggml_type_name(op->src[0]->type):"", op->src[1]?ggml_type_name(op->src[1]->type):"");

	return res;
}

static bool ggml_backend_opencl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_opencl_buffer_type_get_name;

    GGML_UNUSED(dev);
}
static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:// этот запрос отвечает за embedded layer выборка строки из матрицы, вызывает задержку, видимо нужна синхронизация
            return (ggml_is_quantized(op->src[0]->type)/* ||  op->ne[1]>1 || op->ne[2]>1 */)?0:op->ne[0];
//            return (!ggml_is_contiguous(op->src[0]) || !ggml_is_scalar(op->src[1])/* ||  op->ne[1]>1 || op->ne[2]>1 */)?0:op->ne[0];
//            return op->ne[0];
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_opencl_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;
    
    bool res = get_op_batch_size(op) >= min_batch_size;
#if defined(_TEST) && _TEST>0
    if(!res) {
        char buf[128];
        int len = sprintf(buf, " OFFLOAD REQUESTED %s.%s %s %s (%zu x %zu x %zu) (%zu x %zu x %zu) (%zu x %zu x %zu)", 
        ggml_op_name(op->op), ggml_type_name(op->type),
        op->src[0]?ggml_type_name(op->src[0]->type):"", 
        op->src[1]?ggml_type_name(op->src[1]->type):"", 
        op->ne[0], op->ne[1], op->ne[2],
        op->src[0]->ne[0], op->src[0]->ne[1], op->src[0]->ne[2],
        op->src[1]->ne[0], op->src[1]->ne[1], op->src[1]->ne[2]
       );
       puts(buf);
    }
#endif
    return res;

    GGML_UNUSED(dev);
}
static struct ggml_backend_device_i ggml_backend_opencl_device_i = {
    /* .get_name             = */ ggml_backend_opencl_device_get_name,
    /* .get_description      = */ ggml_backend_opencl_device_get_description,
    /* .get_memory           = */ ggml_backend_opencl_device_get_memory,
    /* .get_type             = */ ggml_backend_opencl_device_get_type,
    /* .get_props            = */ ggml_backend_opencl_device_get_props,
    /* .init_backend         = */ ggml_backend_opencl_device_init,
    /* .get_buffer_type      = */ ggml_backend_opencl_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_opencl_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_opencl_device_supports_op,
    /* .supports_buft        = */ ggml_backend_opencl_device_supports_buft,
    /* .offload_op           = */ ggml_backend_opencl_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// Backend registry

static const char * ggml_backend_opencl_reg_get_name(ggml_backend_reg_t reg) {
    return "OpenCL";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_opencl_reg_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_opencl_n_devices;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_opencl_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    return &g_ggml_backend_opencl_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static struct ggml_backend_reg_i ggml_backend_opencl_reg_i = {
    /* .get_name         = */ ggml_backend_opencl_reg_get_name,
    /* .device_count     = */ ggml_backend_opencl_reg_device_count,
    /* .device_get       = */ ggml_backend_opencl_reg_device_get,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_opencl_reg(void) {
    // TODO: make this thread-safe somehow?
    static ggml_backend_reg reg;
    static bool initialized = false;

    if (!initialized) {
        reg = ggml_backend_reg {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface   = */ ggml_backend_opencl_reg_i,
            /* .context = */ NULL,
        };

        g_ggml_backend_opencl_device = ggml_backend_device {
            /* .iface   = */ ggml_backend_opencl_device_i,
            /* .reg     = */ &reg,
            /* .context = */ &g_ggml_ctx_dev_main,
        };

        ggml_cl2_init(&g_ggml_backend_opencl_device);

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_opencl_reg)

//------------------------------------------------------------------------------
// Debugging utils
//------------------------------------------------------------------------------
#if 0
#define QK4_0 32
typedef struct {
    ggml_fp16_t d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2,
    "wrong q4_0 block size/padding");

#include <math.h>
#ifdef __cplusplus
#include "half.hpp"
#endif

static void dump_tensor(ggml_backend_t backend, const struct ggml_tensor * tensor) {
    void * buf = malloc(ggml_nbytes(tensor));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;
#ifdef GGML_OPENCL_SOA_Q
    void * buf_q;
    void * buf_d;
#endif

    // Make sure everything is done.
    CL_CHECK(clFinish(queue));

#ifdef GGML_OPENCL_SOA_Q
    if (tensor->type == GGML_TYPE_Q4_0) {
        ggml_tensor_extra_cl_q4_0 * extra = (ggml_tensor_extra_cl_q4_0 *) tensor->extra;
        GGML_ASSERT(extra);

        size_t size_q = ggml_nelements(tensor)/QK4_0 * QK4_0/2;
        size_t size_d = ggml_nelements(tensor)/QK4_0 * sizeof(ggml_fp16_t);
        GGML_ASSERT(size_q + size_d == ggml_nbytes(tensor));
        buf_q = malloc(size_q);
        buf_d = malloc(size_d);

        CL_CHECK(clEnqueueReadBuffer(queue, extra->q, CL_TRUE, 0, size_q, buf_q, 0, NULL, NULL));
        CL_CHECK(clEnqueueReadBuffer(queue, extra->d, CL_TRUE, 0, size_d, buf_d, 0, NULL, NULL));
        CL_CHECK(clFinish(queue));
    } else {
        // Read out the tensor from GPU memory.
        ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
        GGML_ASSERT(extra);

        CL_CHECK(clEnqueueReadBuffer(queue, extra->data_device, CL_TRUE,
        extra->offset, ggml_nbytes(tensor), buf, 0, NULL, NULL));
        CL_CHECK(clFinish(queue));
    }
#else
    // Read out the tensor from GPU memory.
    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
    GGML_ASSERT(extra);

    CL_CHECK(clEnqueueReadBuffer(queue, extra->data_device, CL_TRUE,
        extra->offset, ggml_nbytes(tensor), buf, 0, NULL, NULL));
    CL_CHECK(clFinish(queue));
#endif // GGML_OPENCL_SOA_Q

    // Open file and dump.
    char fname[512];
    sprintf(fname, "./tensor-dumps/%s.txt", tensor->name);
    FILE * f = fopen(fname, "w");
    if (!f) {
        printf("Failed to open %s\n", fname);
        return;
    }

    if (tensor->type == GGML_TYPE_F32) {
        float * data = (float *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%f\n", data[i]);
        }
    } else if (tensor->type == GGML_TYPE_I32) {
        int * data = (int *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%d\n", data[i]);
        }
    } else if (tensor->type == GGML_TYPE_F16) {
#ifdef __cplusplus
        half_float::half * data = (half_float::half *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (std::isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%f\n", float(data[i]));
        }
#endif
    } else if (tensor->type == GGML_TYPE_Q4_0) {
#ifdef GGML_OPENCL_SOA_Q
        ggml_fp16_t * data_d = (ggml_fp16_t *)buf_d;
        unsigned char * data_q = (unsigned char *)buf_q;

        for (int i = 0; i < ggml_nelements(tensor)/QK4_0; ++i) {
            fprintf(f, "%04x, ", data_d[i]);
            for (int k = 0; k < QK4_0/2; ++k) {
                fprintf(f, "%02x, ", data_q[k]);
            }
            fprintf(f, "\n");
            data_q += QK4_0/2;
        }
        free(buf_d);
        free(buf_q);
#else
        block_q4_0 * data = (block_q4_0 *) buf;
        for (int i = 0; i < ggml_nelements(tensor)/QK4_0; ++i) {
            fprintf(f, "%04x, ", data[i].d);
            for (int k = 0; k < QK4_0/2; ++k) {
                fprintf(f, "%02x, ", data[i].qs[k]);
            }
            fprintf(f, "\n");
        }
#endif // GGML_OPENCL_SOA_Q
    }
    free(buf);
    fflush(f);
    fclose(f);
}
#else
#define dump_tensor(tensor)
#endif

//------------------------------------------------------------------------------
// Profiling utility
//------------------------------------------------------------------------------
#ifdef GGML_OPENCL_PROFILING 

static uint64_t profi_lines=0;
void populateProfilingInfo(
        ProfilingInfo& info, cl_event evt, cl_kernel kernel,
        size_t global_size[3], size_t local_size[3], const ggml_tensor * tensor, const ggml_tensor * tensor0) {
    if (profi_lines==MAX_PROFI_SIZE) {
        float total_kernel_time = 0;
        int event_idx=0;
        printf("op name, kernel name, delay (ms) duration (ms), global size, local size, output size\n");
        for (const ProfilingInfo & info : g_profiling_info) {
            event_idx++;
            if (event_idx>MAX_PROFI_SIZE-1024) {
                total_kernel_time += info.duration_ns/1.e6f;
                //if (info.duration_ns/1.e6f >0.3)
                int color =37;
                if (info.duration_ns<info.delay_ns) color=31;
                printf("%-24.24s %-27.27s \e[%dm%5.3f %7.3f\e[0m\t%3zux%2zux%zu %6zu.%2zu.%zu\t%5zux%-3zu %5zux%zux%zu\n",
                    info.op_name.c_str(), info.kernel_name.c_str(), 
                    color, info.delay_ns/1.e6f, info.duration_ns/1.e6f,
                    info.local_size [0], info.local_size [1], info.local_size [2],
                    info.global_size[0], info.global_size[1], info.global_size[2],
                    info.src1_size[0], info.src1_size[1],
                    info.output_size[0], info.output_size[1], info.output_size[2]/*, info.output_size[3]*/);
            }
        }
    }
    if (profi_lines++>MAX_PROFI_SIZE) {
//        CL_CHECK(clReleaseEvent(evt));
        return ;
    }
    cl_ulong submit;
    cl_ulong queued;
    cl_ulong start;
    cl_ulong end;
    cl_ulong complete;
    CL_CHECK(clWaitForEvents(1, &evt));

    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_QUEUED,  sizeof(cl_ulong), &submit, NULL));
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_QUEUED,  sizeof(cl_ulong), &queued, NULL));
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,   sizeof(cl_ulong), &start, NULL));
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,     sizeof(cl_ulong), &end, NULL));
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_COMPLETE,sizeof(cl_ulong), &complete, NULL));

//    CL_CHECK(clReleaseEvent(evt));
    info.delay_ns = start-queued;   // разница возникает, когда применяетcя ожидание события
    info.duration_ns = end - start; // время на исполнение
//    info.evt = evt;
    char kernel_name[64];
    CL_CHECK(clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME,
        sizeof(kernel_name), kernel_name, NULL));
//    info.duration_ns = 0;
    info.op_name = tensor->name;
    info.kernel_name = kernel_name;
    info.local_size[0]  = local_size[0];
    info.local_size[1]  = local_size[1];
    info.local_size[2]  = local_size[2];
    info.global_size[0] = global_size[0];
    info.global_size[1] = global_size[1];
    info.global_size[2] = global_size[2];

    info.output_size[0] = tensor->ne[0];
    info.output_size[1] = tensor->ne[1];
    info.output_size[2] = tensor->ne[2];
    info.output_size[3] = tensor->ne[3];
    if (tensor0!=NULL){
        info.src1_size[0] = tensor0->ne[0];
        info.src1_size[1] = tensor0->ne[1];
        info.src1_size[2] = tensor0->ne[2];
        info.src1_size[3] = tensor0->ne[3];
    } else {
        info.src1_size[0] = 0;
        info.src1_size[1] = 0;
        info.src1_size[2] = 0;
        info.src1_size[3] = 0;
    }
}
#endif

//------------------------------------------------------------------------------
// Ops
//------------------------------------------------------------------------------

static bool ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type)) &&
            src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32  &&
            (ne0 >= 32 && ne1 >= 32 && ne10 >= 32);
}

static void ggml_cl_nop(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(backend);
    UNUSED(src0);
    UNUSED(src1);
    UNUSED(dst);
}
/*! \brief Кеширование результата */
static void ggml_cl_get_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const cl_uint ne00 = src0->ne[0];
    const cl_uint nb01 = src0->nb[1];
    const cl_uint nb02 = src0->nb[2];
    const cl_uint ne10 = src1->ne[0];
    const cl_uint nb10 = src1->nb[0];
    const cl_uint ne11 = src1->ne[1];
    const cl_uint nb11 = src1->nb[1];
    const cl_uint nb1  =  dst->nb[1];
    const cl_uint nb2  =  dst->nb[2];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_kernel kernel = backend_ctx->kernel_get_rows[src0->type];
    GGML_ASSERT( dst->type==GGML_TYPE_F32);
    GGML_ASSERT(src1->type==GGML_TYPE_I32 && ggml_is_vector(src1) && ne11==1);
    GGML_ASSERT(ggml_is_matrix(src0) && ggml_is_contiguous(src0));
    GGML_ASSERT(kernel!=nullptr && "not implemented");

    
    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);
    int i=0;
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_mem),   &A_d));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_mem),   &B_d));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_mem),   &C_d));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint),  &ne00));// число элементов в строке
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint),  &nb01));// число блоков
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint),  &nb02));
//  CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint),  &ne10));// число элементов в векторе - не используется
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint), &nb10));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint), &nb11));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint), &nb1));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint), &nb2));

//    size_t n = ggml_nelements();
    size_t global_work_size[] = {(size_t)ne00/32, (size_t)ne10, 1};// N x 1 1
// \todo Выбрать число 128 64 32 или спросить выбор suggested local work size for kernel
/* size_t sgs;
    CL_CHECK(clGetKernelSubGroupInfo(kernel, dev_ctx->device,
    CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
    sizeof(local_work_size), local_work_size,
    sizeof(size_t), &sgs, NULL)); */
    //size_t local_work_size[] = {MIN(64,ne00), (size_t)ne11, 1};
    size_t local_work_size[] = {32, 1, 1};
    //CL_CHECK(SuggestedLocalWorkSize(queue, kernel,3,NULL, global_work_size, local_work_size));// можно договориться
    

    cl_event event_list[GGML_MAX_SRC];
    cl_int event_list_len = _event_list(dst, event_list);
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));

#if defined(_TEST) && _TEST>0
        ggml_tensor * op = dst;
        printf(" OP %s %4s %4s %s (%zu x %zu x %zu) work.gr (%zu x %zu x %zu)\t; %s, %s %s(%d)\n", 
        ggml_op_name(op->op), 
        op->src[0]?ggml_type_name(op->src[0]->type):"", 
        op->src[1]?ggml_type_name(op->src[1]->type):"", 
        ggml_type_name(op->type), 
        op->ne[0],
        op->ne[1],
        op->ne[2],
        global_work_size[0],
        global_work_size[1],
        global_work_size[2],
        op->name, op->src[0]->name, op->src[1]->name, event_list_len
        );
#endif

#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, src0, src1);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(B_d));
    CL_CHECK(clReleaseMemObject(C_d));

}

static void ggml_cl_add(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;
    const int  ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb00 = src0 ? src0->nb[0] : 0;
    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    const int  ne10 = src1 ? src1->ne[0] : 0;
    const int  ne11 = src1 ? src1->ne[1] : 0;
    const int  ne12 = src1 ? src1->ne[2] : 0;
    const int  ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

    const cl_ulong nb10 = src1 ? src1->nb[0] : 0;
    const cl_ulong nb11 = src1 ? src1->nb[1] : 0;
    const cl_ulong nb12 = src1 ? src1->nb[2] : 0;
    const cl_ulong nb13 = src1 ? src1->nb[3] : 0; UNUSED(nb13);

    const int  ne0  = dst ? dst->ne[0] : 0;
    const int  ne1  = dst ? dst->ne[1] : 0;
    const int  ne2  = dst ? dst->ne[2] : 0;
    const int  ne3  = dst ? dst->ne[3] : 0;

    const cl_ulong nb0  = dst ? dst->nb[0] : 0;
    const cl_ulong nb1  = dst ? dst->nb[1] : 0;
    const cl_ulong nb2  = dst ? dst->nb[2] : 0;
    const cl_ulong nb3  = dst ? dst->nb[3] : 0;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    bool bcast_row = false;
    cl_kernel kernel;
    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    GGML_ASSERT(ggml_n_dims(src0)<=2);
    if ((ggml_nelements(src1) == ggml_nelements(src0) || ne11==1) && ne10==ne00) {
        GGML_ASSERT(ne00 % 32 == 0 && ne10 % 8 == 0);
        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ggml_is_contiguous(src1));

        // src1 is a row
        // GGML_ASSERT(ne11 == 1);

        bcast_row = true;
        int ne = ne10*ne11 / 8;
        if (ne11==1) 
            kernel = backend_ctx->kernel_add_row;
        else
            kernel = backend_ctx->kernel_add_flat;
    
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &A_d));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &B_d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &C_d));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint),  &ne));
    } else {
        kernel = backend_ctx->kernel_add;
        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_d));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &B_d));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &C_d));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne02));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne03));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb00));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne13));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb10));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb11));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb12));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb13));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne1));
        CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne2));
        CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne3));
        CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &nb0));
        CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong), &nb1));
        CL_CHECK(clSetKernelArg(kernel, 25, sizeof(cl_ulong), &nb2));
        CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb3));
    }

    if (bcast_row) {
        int n = ggml_nelements(dst)/8;
        size_t global_work_size[] = {(size_t)n, 1, 1};
        size_t  local_work_size[] = {(size_t)MIN(64,n), 1, 1};// 
        CL_CHECK(SuggestedLocalWorkSize(queue, kernel, 3, NULL, global_work_size, local_work_size)); 
        cl_event event_list[GGML_MAX_SRC];
        cl_uint  event_list_len = _event_list(dst, event_list);
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
        g_profiling_info.emplace_back();
        populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src1);
#endif
    } else {
        unsigned int nth = MIN(64, ne0);
        size_t global_work_size[] = {ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t  local_work_size[] = {nth, 1, 1};

        cl_event event_list[GGML_MAX_SRC];
        cl_uint  event_list_len = _event_list(dst, event_list);

        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
        g_profiling_info.emplace_back();
        populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src1);
#endif
    }
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(B_d));
    CL_CHECK(clReleaseMemObject(C_d));
}
// параметры TF32 загружаются как f32
static inline _Float32 _tensor_get_param_f32(const ggml_tensor* tensor, int idx ){
    _Float32 v;
    __builtin_memcpy(&v, tensor->op_params+idx, sizeof(float));
    return v;
}
static inline void _tensor_set_param_f32(ggml_tensor* tensor, int idx, float value ){
    __builtin_memcpy(tensor->op_params+idx, &value, sizeof(float));
}
static inline int32_t _tensor_get_param_i32(const ggml_tensor* tensor, int idx ){
    return tensor->op_params[idx];
}
static void ggml_cl_gga(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

//    const int  ne10 = src1->ne[0];
//    const int  ne11 = src1->ne[1];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    GGML_ASSERT(src0->op==GGML_OP_GET_ROWS);
    GGML_ASSERT(src1->op==GGML_OP_GET_ROWS);
    GGML_ASSERT(src0->src[1]->type==GGML_TYPE_I32);
    GGML_ASSERT(src1->src[1]->type==GGML_TYPE_I32);
    int32_t row0 = _tensor_get_param_i32(src0->src[1], 0);
    int32_t row1 = _tensor_get_param_i32(src1->src[1], 0);
    //printf("row0=%d row1=%d\n", row0, row1);
    cl_mem A_d =_get_cl_buffer_row(src0->src[0], row0, CL_MEM_READ_ONLY );
    cl_mem B_d =_get_cl_buffer_row(src1->src[0], row1, CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    const int ne = ggml_nelements(dst)/8;
    cl_kernel kernel;
    kernel = backend_ctx->kernel_add_flat;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &A_d));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &B_d));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &C_d));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint),  &ne));

    int n = ggml_nelements(dst)/8;
    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t  local_work_size[] = {(size_t)MIN(64,n), 1, 1};// 
    CL_CHECK(SuggestedLocalWorkSize(queue, kernel, 3, NULL, global_work_size, local_work_size)); 
    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = 0;//_event_list(dst, event_list);
    event_list_len += _event_get(src0->src[0], event_list+event_list_len);
    event_list_len += _event_get(src1->src[0], event_list+event_list_len);
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src1);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(B_d));
    CL_CHECK(clReleaseMemObject(C_d));
}

static void ggml_cl_mul(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const int ne00 = src0 ? src0->ne[0] : 0;
    const int ne01 = src0 ? src0->ne[1] : 0;
    const int ne02 = src0 ? src0->ne[2] : 0;
    const int ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb00 = src0 ? src0->nb[0] : 0;
    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    const int ne10 = src1 ? src1->ne[0] : 0;
    const int ne11 = src1 ? src1->ne[1] : 0;
    const int ne12 = src1 ? src1->ne[2] : 0;
    const int ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

    const cl_ulong nb10 = src1 ? src1->nb[0] : 0;
    const cl_ulong nb11 = src1 ? src1->nb[1] : 0;
    const cl_ulong nb12 = src1 ? src1->nb[2] : 0;
    const cl_ulong nb13 = src1 ? src1->nb[3] : 0; UNUSED(nb13);

    const int ne0  = dst ? dst->ne[0] : 0;
    const int ne1  = dst ? dst->ne[1] : 0;
    const int ne2  = dst ? dst->ne[2] : 0;
    const int ne3  = dst ? dst->ne[3] : 0;

    const cl_ulong nb0  = dst ? dst->nb[0] : 0;
    const cl_ulong nb1  = dst ? dst->nb[1] : 0;
    const cl_ulong nb2  = dst ? dst->nb[2] : 0;
    const cl_ulong nb3  = dst ? dst->nb[3] : 0;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    bool bcast_row = false;
    cl_kernel kernel;
    cl_mem              A_d = nullptr;
    cl_mem              C_d = nullptr;
    cl_mem              B_d = nullptr;
    GGML_ASSERT(ggml_n_dims(src0)<=2);
    if ((ggml_nelements(src1) == ggml_nelements(src0) || ne11==1) && ne10==ne00) {
        GGML_ASSERT(ne00 % 32 == 0 && ne10 % 8 == 0);
        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ggml_is_contiguous(src1));

        // src1 is a row
        // GGML_ASSERT(ne11 == 1);

        bcast_row = true;
        int ne = ne10*ne11 / 8;
        if (ne11==1) 
            kernel = backend_ctx->kernel_mul_row;
        else
            kernel = backend_ctx->kernel_mul_flat;
        cl_int status;
        cl_buffer_region region;
        int N = ggml_nelements(dst);
        region.origin = offset0; region.size = ggml_nelements(src0)/*M * N */* sizeof(float);
        A_d = clCreateSubBuffer(extra0->data_device, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
        region.origin = offset1; region.size = ne10*ne11/*M * N */* sizeof(float);
        B_d = clCreateSubBuffer(extra1->data_device, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
        region.origin = offsetd; region.size = N/*M * N */* sizeof(float);
        C_d = clCreateSubBuffer(extrad->data_device, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
        CL_CHECK(status);
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &A_d));//extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &B_d));//extra1->data_device
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &C_d));//extrad->data_device
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint),  &ne));
    } else {
        kernel = backend_ctx->kernel_mul;

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne03));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb00));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne11));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne13));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb10));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
        CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
        CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &ne1));
        CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &ne2));
        CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &ne3));
        CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb0));
        CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb1));
        CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb2));
        CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nb3));
    }

    if (bcast_row) {
        size_t n = ggml_nelements(dst)/8;
        size_t global_work_size[] = {n, 1, 1};
        size_t  local_work_size[] = {MIN(64,n), 1, 1};

        CL_CHECK(SuggestedLocalWorkSize(queue, kernel,3,NULL, global_work_size, local_work_size));// можно договориться
        cl_event event_list[GGML_MAX_SRC];
        cl_uint  event_list_len = _event_list(dst, event_list);
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
        g_profiling_info.emplace_back();
        populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src1);
#endif
        CL_CHECK(clReleaseMemObject(A_d)); A_d = nullptr;
        CL_CHECK(clReleaseMemObject(B_d)); B_d = nullptr;
        CL_CHECK(clReleaseMemObject(C_d)); C_d = nullptr;// надо ли дожидаться завершения?
    } else {
        unsigned int nth = MIN(64, ne0);
        size_t global_work_size[] = {ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t  local_work_size[] = {nth, 1, 1};

        cl_event event_list[GGML_MAX_SRC];
        cl_uint  event_list_len = _event_list(dst, event_list);

        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
        g_profiling_info.emplace_back();
        populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src1);
#endif
    }
}
static void _cl_unary_op(ggml_backend_opencl_context* backend_ctx, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(dst->op==GGML_OP_UNARY);
    cl_kernel kernel = backend_ctx->kernel_unary[ggml_get_unary_op(dst)];
    if (dst->type==GGML_TYPE_F32) 
        kernel = backend_ctx->kernel_unary[ggml_get_unary_op(dst)];
    else if(dst->type==GGML_TYPE_F16)
        kernel = backend_ctx->kernel_unary_f16[ggml_get_unary_op(dst)];
    else if(dst->type==GGML_TYPE_BF16)
        kernel = backend_ctx->kernel_unary_bf16[ggml_get_unary_op(dst)];
    else
        GGML_ASSERT(0 && "UNARY type");


    cl_command_queue queue = backend_ctx->queue;
    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &A_d));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &C_d));

    size_t n = ggml_nelements(dst);
    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t  local_work_size[] = {(size_t)MIN(128,n), 1, 1};
    CL_CHECK(SuggestedLocalWorkSize(queue, kernel,3,NULL, global_work_size, local_work_size));// можно договориться

    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = _event_list(dst, event_list);

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt);
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src0);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(C_d));
}
static void ggml_cl_op_abs (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
}
static void ggml_cl_op_neg (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
}
static void ggml_cl_op_sgn (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
}
static void ggml_cl_op_step (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
}
static void ggml_cl_tanh (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
}
static void ggml_cl_sigmoid (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
}
static void ggml_cl_hardsigmoid (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
}
static void ggml_cl_hardswish (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
}
static void ggml_cl_gelu_quick (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
}
static void ggml_cl_exp (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
}
static void ggml_cl_elu (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
}
static void ggml_cl_gelu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
#if 0
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    cl_kernel kernel;

    int n = ggml_nelements(dst);

    kernel = backend_ctx->kernel_gelu;

    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &A_d));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &C_d));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t  local_work_size[] = {(size_t)MIN(128,n), 1, 1};
    CL_CHECK(SuggestedLocalWorkSize(queue, kernel,3,NULL, global_work_size, local_work_size));// можно договориться
    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = _event_list(dst, event_list);
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt);
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src0);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(C_d));
#endif
}

static void ggml_cl_silu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
#if 0
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    cl_kernel kernel;

    int n = ggml_nelements(dst);
    if (n % 8 == 0) {
        kernel = backend_ctx->kernel_silu_4;
        n /= 8;
    } else {
        kernel = backend_ctx->kernel_silu;
    }

    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &A_d));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &C_d));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t  local_work_size[] = {(size_t)MIN(128,n), 1, 1};
    CL_CHECK(SuggestedLocalWorkSize(queue, kernel,3,NULL, global_work_size, local_work_size));// можно договориться
    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = _event_list(dst, event_list);

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src0);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(C_d));
#endif
}

static void ggml_cl_relu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    _cl_unary_op(backend_ctx, src0, dst);
#if 0
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    cl_kernel kernel = nullptr;
    if (dst->type==GGML_TYPE_F32)
        kernel = backend_ctx->kernel_relu;
    else 
        kernel = backend_ctx->kernel_relu_f16;


    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &A_d));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &C_d));

    int n = ggml_nelements(dst);
    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t  local_work_size[] = {(size_t)MIN(128,n), 1, 1};
    CL_CHECK(SuggestedLocalWorkSize(queue, kernel,3,NULL, global_work_size, local_work_size));// можно договориться
    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = _event_list(dst, event_list);

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src0);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(C_d));
#endif
}

static void ggml_cl_clamp(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    float min, max;
    __builtin_memcpy(&min, ((int32_t *) dst->op_params) + 0, sizeof(float));
    __builtin_memcpy(&max, ((int32_t *) dst->op_params) + 1, sizeof(float));

    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    cl_kernel kernel = backend_ctx->kernel_clamp;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &A_d));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &C_d));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_float), &min));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_float), &max));

    const size_t n = ggml_nelements(dst);

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t  local_work_size[] = {(size_t)MIN(64,n), 1, 1};

    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = _event_list(dst, event_list);

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    CL_CHECK(clEnqueueNDRangeKernel(backend_ctx->queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src0);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(C_d));
}

static void ggml_cl_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    float eps;
    __builtin_memcpy(&eps, dst->op_params, sizeof(float));

    const cl_uint  ne00 = src0->ne[0];
    const cl_ulong nb01 = src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));

    const int nth = MIN(64, ne00);

    cl_kernel kernel = backend_ctx->kernel_norm;
    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &A_d));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),    &C_d));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_uint),   &ne00));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &nb01));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_float),     &eps));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_float)*nth, NULL));

    const int64_t nrows = ggml_nrows(src0);

    size_t global_work_size[] = {(size_t)nrows*nth, 1, 1};
    size_t  local_work_size[] = {(size_t)nth, 1, 1};

    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = _event_list(dst, event_list);

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src0);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(C_d));
}

// × вычисление выборки из матрицы (подматрицы) с учетом перестановок
static void _view(struct ggml_tensor * t, size_t *origin, const size_t *pitch){
    t->view_offs = origin[2]* pitch[2] + origin[1] * pitch[1] + origin[0]*pitch[0];
}


/*!
    \see OpenCL Specification
    Appendix D: Checking for Memory Copy Overlap


    Рассматриваются два случая для блочных(квантизованных) типов и для типов нормальных. 
    способ расчета (actual_size) с пересатновкой или пропуском подстрок, см. ggml_nbytes:
```c
    nbytes = ggml_type_size(tensor->type);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
    } 
    region.size = nbytes;
```
*/
unsigned int check_copy_overlap(const size_t src_origin[],
    const size_t dst_origin[],
    const size_t region[],
    const size_t row_pitch,
    const size_t slice_pitch )
{
    const size_t slice_size = (region[1] - 1) *   row_pitch + region[0];
    const size_t block_size = (region[2] - 1) * slice_pitch + slice_size;
    const size_t src_start = src_origin[2] * slice_pitch + src_origin[1] * row_pitch + src_origin[0];
    const size_t src_end   = src_start + block_size;
    const size_t dst_start = dst_origin[2] * slice_pitch + dst_origin[1] * row_pitch + dst_origin[0];
    const size_t dst_end   = dst_start + block_size;
    /*  No overlap if dst ends before src starts or if src ends before dst starts. */
    if( (dst_end <= src_start) || (src_end <= dst_start) )  return 0;

    {/* No overlap if region[0] for dst or src fits in the gap
    between region[0] and row_pitch. */
        const size_t src_dx = src_origin[0] % row_pitch;
        const size_t dst_dx = dst_origin[0] % row_pitch;
        if( ((dst_dx >= src_dx + region[0]) && (dst_dx + region[0] <= src_dx + row_pitch)) 
        ||  ((src_dx >= dst_dx + region[0]) && (src_dx + region[0] <= dst_dx + row_pitch)) )
            return 0;
    }
    {/* No overlap if region[1] for dst or src fits in the gap
        between region[1] and slice_pitch. */
        const size_t src_dy = (src_origin[1] * row_pitch + src_origin[0]) % slice_pitch;
        const size_t dst_dy = (dst_origin[1] * row_pitch + dst_origin[0]) % slice_pitch;
        if( ((dst_dy >= src_dy + slice_size) && (dst_dy + slice_size <= src_dy + slice_pitch)) 
        ||  ((src_dy >= dst_dy + slice_size) && (src_dy + slice_size <= dst_dy + slice_pitch)) ) 
            return 0;
    }
/* Otherwise src and dst overlap. */
    return 1;
}
/* функция работает только для тензоров для которых строки уложены подряд is_contiguous и для permuted, в том числе для sliced 
 */
static cl_mem _get_cl_buffer(const ggml_tensor * tensor, cl_mem_flags mem_flags){
    cl_int status;
    cl_buffer_region region;

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *)tensor->extra;
GGML_ASSERT(tensor->view_offs==0);
    region.origin = extra->offset;// + tensor->view_offs; 
    region.size = extra->actual_size;
    cl_mem buffer = clCreateSubBuffer(extra->data_device, mem_flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    if (status){
        char buf[128];
        sprintf(buf, "ERR:CreateSubBuffer %s(%zu,%zu,%zu) status=%d\n", tensor->name, tensor->ne[0],tensor->ne[1],tensor->ne[2], status);
        puts(buf);
    }
    CL_CHECK(status);
    return buffer;
}
/*! \brief выделить одну строку из матрицы и создать буфер для загрузки 

заметим, операции по выделению блока могут быть: _row, _column, _subrow, _subcolumn, _submatrix
матрицы могут быть _is_permuted - перестановка блоков, при этом 
    nb1 = ne2*nb2 -- по строке, ne2 - число голов
 */

static cl_mem _get_cl_buffer_row(const ggml_tensor * tensor, int row, cl_mem_flags mem_flags){
    cl_int status;
    cl_buffer_region region;

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *)tensor->extra;
    const size_t blck_size = ggml_blck_size(tensor->type);
    GGML_ASSERT(blck_size==1);// только для типов tf32 f16 bf16 и т.п.
    const size_t nbytes = ggml_type_size(tensor->type);
    size_t offset = extra->offset + tensor->view_offs + row*tensor->ne[0]*nbytes;
    region.origin = offset; 
    region.size   = tensor->ne[0]*nbytes;
    cl_mem buffer = clCreateSubBuffer(extra->data_device, mem_flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    if (status){
        printf("ERR:CreateSubBuffer %s(%zu,%zu,%zu) status=%d\n", tensor->name, tensor->ne[0],tensor->ne[1],tensor->ne[2], status);
    }
    CL_CHECK(status);
    return buffer;
}

static void ggml_cl_rms_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;


    cl_float eps;// = *(float*)dst->op_params;
    static_assert(sizeof(float)==sizeof(cl_float));
    __builtin_memcpy(&eps, dst->op_params, sizeof(float));

    const cl_uint ne00 = src0->ne[0];
    const cl_uint nb01 = src0->nb[1];

    GGML_ASSERT(ne00 % 8 == 0);
    GGML_ASSERT(ggml_is_contiguous_1(src0));

    const size_t nth = MIN(64, ne00/4);

    const int64_t nrows = ggml_nrows(src0);

    size_t global_work_size[] = {(size_t)nrows*nth, 1, 1};
    size_t  local_work_size[] = {(size_t)nth, 1, 1};
    cl_kernel kernel = backend_ctx->kernel_rms_norm;// traitis(rms_norm, DST_T)

    // Note, this kernel declares local memory in kernel args and the size
    // depends on subgroup size.
    // Retrieve subgroup size.
    // Note, this requires OpenCL 2.1 and above
/*
    size_t sgs;
    CL_CHECK(clGetKernelSubGroupInfo(kernel, dev_ctx->device,
        CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
        sizeof(local_work_size), local_work_size,
        sizeof(size_t), &sgs, NULL));
*/
    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);
    
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &A_d));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),    &C_d));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_uint),   &ne00));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint),   &nb01));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_float),  &eps));

    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = _event_list(dst, event_list);

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));

#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src0);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(C_d));
}
static void ggml_cl_rms_norm_mul(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    cl_float eps;// = *(float*)dst->op_params;
    static_assert(sizeof(float)==sizeof(cl_float));
    __builtin_memcpy(&eps, dst->op_params, sizeof(float));

    const cl_uint ne00 = src0->ne[0];
    const cl_uint nb01 = src0->nb[1];
    const cl_uint ne11 = src1->ne[1];

    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ggml_is_contiguous_1(src0));

    const int nth = MIN(64, ne00/4);

    const int64_t nrows = ggml_nrows(src0);

    size_t global_work_size[] = {(size_t)nrows*nth, 1, 1};
    size_t  local_work_size[] = {(size_t)nth, 1, 1};
    cl_kernel kernel = backend_ctx->kernel_rms_norm_mul;// может быть в квантах f16 bf16 и 8_1

    //CL_CHECK(SuggestedLocalWorkSize(queue, kernel, 3, NULL, global_work_size, local_work_size));// 

    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);
    
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &A_d));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),    &B_d));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),    &C_d));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint),   &ne00));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint),   &nb01));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint),   &ne11));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_float),  &eps));

    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = 0;
    event_list_len+= _event_get(src0, event_list+0);
    event_list_len+= _event_get(src1, event_list+event_list_len);

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;// добыть событие, сделать метод
    CL_CHECK(clEnqueueNDRangeKernel(backend_ctx->queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src1);
#endif

    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(B_d));
    CL_CHECK(clReleaseMemObject(C_d));
}
static void ggml_cl_mva(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(src2);
    GGML_ASSERT(src2->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const enum ggml_type src0t = src0->type;
//    const enum ggml_type src1t = src1->type;
//    const enum ggml_type src2t = src2->type;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    const cl_uint  ne00 = src0->ne[0];
    const cl_uint  ne01 = src0->ne[1];
    const cl_uint  ne02 = src0->ne[2];
    const cl_uint  ne03 = src0->ne[3];
GGML_ASSERT(ne03==1);

//    const cl_ulong nb00 = src0->nb[0];
//    const cl_ulong nb01 = src0->nb[1];
//    const cl_ulong nb02 = src0->nb[2];
//    const cl_ulong nb03 = src0->nb[3];
//GGML_ASSERT(nb03==0);

    const cl_uint  ne10 = src1->ne[0];
    const cl_uint  ne11 = src1->ne[1];
    const cl_uint  ne12 = src1->ne[2];
    const cl_uint  ne13 = src1->ne[3];

    const cl_uint  ne21 = src2->ne[1];
//    const cl_ulong nb10 = src1->nb[0];
//    const cl_ulong nb11 = src1->nb[1];
//    const cl_ulong nb12 = src1->nb[2];
//    const cl_ulong nb13 = src1->nb[3];
// размерность результата DST
    const cl_uint  ne0 = dst->ne[0];
    const cl_uint  ne1 = dst->ne[1];
// кратность
//    int r2 = ne12/ne02;
//    int r3 = ne13/ne03;

    GGML_ASSERT(ne12*ne13==1);// batch
    GGML_ASSERT(ne02==1);   // 
    GGML_ASSERT(ne00==ne10);// K
    GGML_ASSERT(ne01==ne0); // M dst[MxN]
    GGML_ASSERT(ne11==ne1); // N

    int nth0 = 32;
    int nth1 = 1;
//    int nrows = 1;
    // The number of values produced by each subgroup
    int ndst = 1;

    cl_kernel kernel=nullptr;

    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(src2,CL_MEM_READ_ONLY );
    cl_mem D_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    // use custom matrix x vector kernel
    if (backend_ctx->gpu_family == INTEL) {
        switch (src0t) {
        case GGML_TYPE_Q6_K:
            nth0 = 32; nth1 = 1; ndst = 2;
            break;
        case GGML_TYPE_Q5_K:
            nth0 = 32; nth1 = 1; ndst = 1;
            break;
        case GGML_TYPE_Q4_K:
            nth0 = 32; nth1 = 1; ndst = 1;
            break;
        case GGML_TYPE_Q8_0:
            nth0 = 32; nth1 = 1; ndst = 1;
            break;
        case GGML_TYPE_F16:
            nth0 = 32; nth1 = 1; ndst = 1;
            break;
        case GGML_TYPE_BF16:
            nth0 = 32; nth1 = 1; ndst = 1;
            break;
        default:
            printf("MVA %s %s %s %s\n", ggml_type_name(src0t), ggml_op_name(dst->op), ggml_op_name(dst->src[0]->op),ggml_op_name(dst->src[1]->op));
            GGML_ASSERT(0 && "TODO MVA");
            break;
        }
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    kernel = backend_ctx->kernel_mva[src0t];
    //_get_max_subgroup_size(queue, kernel, );
    float scale = _tensor_get_param_f32(dst, 0);
    int i=0;
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_mem),   &A_d));//extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_mem),   &B_d));//extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_mem),   &C_d));//extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_mem),   &D_d));//extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint),      &ne10));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint),      &ne0));
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_uint),      &ne21));// сам придумал сюда кратность попадает
    CL_CHECK(clSetKernelArg(kernel,  i++, sizeof(cl_float),     &scale));
// \todo исправить формулу: чтобы произведение не зависило от nth0 при увеличении рабочей группы
// сделать определение рабочей группы с проверкой ASSERT(sgs==nth0) и увеличение группы до slice в 4-8 раз. путем увеличения ndst
    size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t  local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};// кратность, общее число должно равняться числу вычислительных блоков 512?

    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = 0;
    event_list_len+= _event_get(src0, event_list+0);
    event_list_len+= _event_get(src1, event_list+event_list_len);
    event_list_len+= _event_get(src2, event_list+event_list_len);

    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src1);
#endif

    if (A_d!=nullptr) CL_CHECK(clReleaseMemObject(A_d)); 
    CL_CHECK(clReleaseMemObject(B_d)); 
    CL_CHECK(clReleaseMemObject(C_d)); 
    CL_CHECK(clReleaseMemObject(D_d)); 
}
static void ggml_cl_ffn         (ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(src2);
    GGML_ASSERT(src2->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    const cl_uint  ne00 = src0->ne[0];
    const cl_uint  ne01 = src0->ne[1];
    const cl_uint  ne10 = src2->ne[0];
    const cl_uint  ne11 = src2->ne[1];
    const cl_uint  ne0  = dst ->ne[0];
    const cl_uint  ne1  = dst ->ne[1];

    int nth0 = 32;
    int nth1 = 1;
//    int nrows = 1;
    // The number of values produced by each subgroup
    int ndst = 1;

    cl_kernel kernel=backend_ctx->kernel_ffn[src0->type];
    switch (src0->type) {
        case GGML_TYPE_Q6_K:
            nth0 = 32; nth1 = 1; ndst = 2;
        break;
        case GGML_TYPE_Q5_K:
            nth0 = 32; nth1 = 1; ndst = 1;
        break;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
            nth0 = 32; nth1 = 1; ndst = 1;
        break;
    default:
        printf("FFN %s %s %s %s\n", ggml_type_name(src0->type), ggml_op_name(dst->op), ggml_op_name(dst->src[0]->op),ggml_op_name(dst->src[1]->op));
        GGML_ASSERT(0 && "TODO FFN");
        break;
    }

    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(src2,CL_MEM_READ_ONLY );
    cl_mem D_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &A_d));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),    &B_d));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),    &C_d));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),    &D_d));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint),   &ne00));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint),   &ne01));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uint),   &ne10));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint),   &ne0));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uint),   &ne1));


    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = 0;
    event_list_len+= _event_get(src0, event_list+0);
    event_list_len+= _event_get(src1, event_list+event_list_len);
    event_list_len+= _event_get(src2, event_list+event_list_len);

    size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, 1};
    size_t  local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};// кратность, общее число должно равняться числу вычислительных блоков 512?

    cl_command_queue queue = backend_ctx->queue;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src0);
#endif

    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(B_d));
    CL_CHECK(clReleaseMemObject(C_d));
    CL_CHECK(clReleaseMemObject(D_d));
}


static void ggml_cl_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const enum ggml_type src0t = src0->type;
    const enum ggml_type src1t = src1->type;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *)src0->extra;
#endif
    const cl_uint  ne00 = src0->ne[0];
    const cl_uint  ne01 = src0->ne[1];
    const cl_uint  ne02 = src0->ne[2];
    const cl_uint  ne03 = src0->ne[3];
GGML_ASSERT(ne03==1);

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    //GGML_ASSERT(nb03==0);

    const cl_uint  ne10 = src1->ne[0];
    const cl_uint  ne11 = src1->ne[1];
    const cl_uint  ne12 = src1->ne[2];
    const cl_uint  ne13 = src1->ne[3];

GGML_ASSERT(ne13==1);

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];
// размерность результата DST
    const cl_uint  ne0 = dst->ne[0];
    const cl_uint  ne1 = dst->ne[1];
// кратность
    cl_uint r2 = ne12/ne02;
    cl_uint r3 = ne13/ne03;

    GGML_ASSERT(r3 == 1);
    GGML_ASSERT(ne00 == ne10);

    int nth0 = 32;
    int nth1 = 1;
    int nrows = 1;
    // The number of values produced by each subgroup
    int ndst = 1;

    cl_kernel kernel=nullptr;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    cl_context context = backend_ctx->context;

    if (ne01 && ne1 && use_adreno_kernels(src0)) {

    // init CL objects
    // <--------------------------------------------> //
    cl_int              status;
    cl_image_format     img_fmt_1d;
    cl_image_desc       img_desc_1d;
    cl_buffer_region    region;
    cl_mem              A_image1d = nullptr;
    cl_mem              B_image1d = nullptr;
    cl_mem              B_sub_buffer = nullptr;
    cl_mem              C_d = nullptr;
    // for B transpose
    cl_mem B_d = nullptr;
    cl_mem B_d_input_image = nullptr;
    // <--------------------------------------------> //

    // define matrix dimensions
    // <--------------------------------------------> //
    int M = ne01;
    int N = ne1;
    int K = ne00;
    int padding;
    // <--------------------------------------------> //

    // q4_0 x fp32
    if(src0t == GGML_TYPE_Q4_0 && src1t == GGML_TYPE_F32) {
        // TODO: remove duplicate definitions of image description + format -- move to top

        // create an image for A
        // <--------------------------------------------> //
        if (N == 1) {
            img_fmt_1d = { CL_R, CL_UNSIGNED_INT32};
        } else {
            img_fmt_1d = { CL_R, CL_FLOAT};
        }
        memset(&img_desc_1d, 0, sizeof(img_desc_1d));
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = M * K / 2 / 4;    // Divide by 4 for char -> float
        img_desc_1d.buffer = extra0_q4_0->q;
        A_image1d = clCreateImage(
            context,
            CL_MEM_READ_ONLY,
            &img_fmt_1d,
            &img_desc_1d,
            NULL,
            &status);
        CL_CHECK(status);
        // <--------------------------------------------> //


        // create a sub_buffer for B
        // <--------------------------------------------> //
        region.origin = (extra1->offset);
        region.size = K * N * sizeof(float);
        B_sub_buffer = clCreateSubBuffer(
            extra1->data_device,
            0,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &status);
        CL_CHECK(status);
        // <--------------------------------------------> //

        // transpose activation for Skyler's gemm
        if (N != 1) {
            //how many extra elements beyond multiple of 8
            int extra_elements = N % 8;

            //how much padding to add
            padding = 0;
            if (extra_elements > 0){
                padding = 8 - extra_elements;
            }

            // Specify the starting offset (in bytes)
            region.origin = 0;
            // Specify the size of the sub-buffer (divide by 2 for FP16)
            region.size = K * (N + padding) * sizeof(float)/2;
            B_d = clCreateSubBuffer(
                backend_ctx->B_d_max,
                0,
                CL_BUFFER_CREATE_TYPE_REGION,
                &region,
                &status);
            CL_CHECK(status);

            cl_image_format image_format_B_d_input = { CL_RGBA, CL_FLOAT };
            cl_image_desc image_desc_B_d_input = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(K * N / 4),
                0, 0, 0, 0, 0, 0, 0, { B_sub_buffer }
            };
            B_d_input_image = clCreateImage(
                context,
                0,
                &image_format_B_d_input,
                &image_desc_B_d_input,
                NULL,
                &status);
            CL_CHECK(status);

            cl_image_format image_format_B_d_output = { CL_RGBA, CL_HALF_FLOAT }; //(CL_HALF_FLOAT for FP16)
            cl_image_desc image_desc_B_d_output = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(K * (N + padding)/4),
                0, 0, 0, 0, 0, 0, 0, { B_d }
            };
            B_image1d = clCreateImage(
                context,
                0,
                &image_format_B_d_output,
                &image_desc_B_d_output,
                NULL,
                &status);
            CL_CHECK(status);

            int height_B = N/4;
            if (height_B == 0) {
                height_B = 1;
            }
            int width_B = K/4;
            int padded_height_B = (N + padding)/4;

            kernel = backend_ctx->kernel_transpose_32_16;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &B_d_input_image));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_image1d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

            size_t local_size_t[2] = { 1, 16 };
            //WGS tuning
            if (ne0 == 4096 && ne1 == 128 && ne10 == 4096) {
                local_size_t[0]=4;
                local_size_t[1]=8;
            } else if (ne0 == 11008 && ne1 == 128 && ne10 == 4096) {
                local_size_t[0]=2;
                local_size_t[1]=8;
            } else if(ne0 == 4096 && ne1 == 128 && ne10 == 11008) {
                local_size_t[0]=1;
                local_size_t[1]=8;
            } else if(ne0 == 32000 && ne1 == 128 && ne10 == 4096) {
                local_size_t[0]=2;
                local_size_t[1]=8;
            }

            size_t global_size_t[2] = {
                static_cast<size_t>(width_B),
                static_cast<size_t>(padded_height_B)
            };

            #ifdef GGML_OPENCL_PROFILING
                cl_event evt;
                CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size_t, local_size_t, 0, NULL, &evt));

                g_profiling_info.emplace_back();
                populateProfilingInfo(g_profiling_info.back(), evt, kernel, global_size_t, local_size_t, dst, src1);
            #else
                CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size_t, local_size_t, 0, NULL, NULL));
            #endif
        } else {
            // no need to transpose B in other cases
            // create an image for B from sub_buffer
            // <--------------------------------------------> //
            img_fmt_1d = {CL_RGBA, CL_FLOAT};

            memset(&img_desc_1d, 0, sizeof(img_desc_1d));
            img_desc_1d.image_width = K * N / 4;
            img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc_1d.buffer = B_sub_buffer;
            B_image1d = clCreateImage(
                context,
                CL_MEM_READ_ONLY,
                &img_fmt_1d,
                &img_desc_1d,
                NULL,
                &status);
            CL_CHECK(status);
            // <--------------------------------------------> //
        }

        // choose gemm or gemv kernel
        // <--------------------------------------------> //
        if (N == 1) {
            kernel = backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_general;
            if (M == 4096 && K == 4096) {
                kernel = backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_4096;
            } else if (M == 4096 && K == 11008) {
                kernel = backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_11008;
            } else if (M == 11008 && K == 4096) {
                kernel = backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_11008_1_4096;
            } else if (M == 32000 && K == 4096) {
                kernel = backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_32000_1_4096;
            }
        } else {
            kernel = backend_ctx->CL_mul_mat_Ab_Bi_8x4;
        }
        // <--------------------------------------------> //

        // set kernel args
        // <--------------------------------------------> //
        cl_uint k_arg = 0;

        if (N == 1) {
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &A_image1d));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &B_image1d));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_ulong), &extra1->offset));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_ulong), &extrad->offset));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &r3));
        } else {
            region.origin = extrad->offset; // Specify the starting offset (in bytes)
            region.size = M * N * sizeof(float); // Specify the size of the sub-buffer
            C_d = clCreateSubBuffer(extrad->data_device, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
            CL_CHECK(status);

            int padded_N = ne1 + padding;

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra0_q4_0->q)); //A_q_dextra0_q4_0->q
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra0_q4_0->d)); //A_s_d
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &B_image1d)); //B_d
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &C_d)); //C_d
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &ne01)); //M
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),    &padded_N)); //N with padding
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),    &ne00)); //K
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),    &ne1)); //N without padding
        }
        // <--------------------------------------------> //

        // choose workgroup size
        // <--------------------------------------------> //
        size_t global_work_size[3] = {
            64, static_cast<size_t>((M+63)/64), static_cast<size_t>((N+31)/32)};
        size_t local_work_size[3] = {64, 2, 4};

        global_work_size[0] = (size_t)(ceil((float)ne1/8));// это вы в своем уме?
        global_work_size[1] = (size_t)(ne01/4);
        global_work_size[2] = (size_t)(1);

        local_work_size[0]  = (size_t)(1); //4x32 for FP32
        local_work_size[1]  = (size_t)(128);
        local_work_size[2]  = (size_t)(1);

        //WGS tuning
        if (ne0 == 4096 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 1;
            local_work_size[1] = 128;
        } else if (ne0 == 11008 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        } else if (ne0 == 4096 && ne1 == 128 && ne10 == 11008) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        } else if (ne0 == 32000 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        }

        if (N == 1) {
            size_t wavesize = backend_ctx->adreno_wave_size;
            local_work_size[0] = wavesize; // localsize
            local_work_size[1] = 4; // reduce factor
            local_work_size[2] = 1;

            global_work_size[0] = (((M / 2) + wavesize - 1) / wavesize) * wavesize;
            global_work_size[1] = 4; // reduce factor
            global_work_size[2] = 1;
        }
        // <--------------------------------------------> //

        // enqueue kernel with profiling
        // <--------------------------------------------> //
    #ifdef GGML_OPENCL_PROFILING
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));

        g_profiling_info.emplace_back();
        populateProfilingInfo(g_profiling_info.back(), evt, kernel, global_work_size, local_work_size, dst, src1);
        // enqueue kernel without profiling
    #else
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    #endif
        // <--------------------------------------------> //

        // deallocate sub buffers and images
        // <--------------------------------------------> //
        CL_CHECK(clReleaseMemObject(A_image1d));
        CL_CHECK(clReleaseMemObject(B_sub_buffer));
        CL_CHECK(clReleaseMemObject(B_image1d));

        if (N != 1) {
            CL_CHECK(clReleaseMemObject(B_d));
            CL_CHECK(clReleaseMemObject(B_d_input_image));
            CL_CHECK(clReleaseMemObject(C_d));
        }
        // <--------------------------------------------> //

        return;
    }
    } // if (ne01 && ne1)
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
// классификация матриц
GGML_ASSERT(!ggml_is_transposed(src0) && "src0 transposed"); // не встречается совсем
GGML_ASSERT(!ggml_is_transposed(src1) && "src1 transposed"); //
GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1);// ggml_is_3d(src0) && ggml_is_3d(src1)); --  размерность не превышает 3х
GGML_ASSERT(src0->ne[0] == src1->ne[0] && src0->ne[0]%32==0);// операции всегда выполняются по строкам, строки кратны 32

if (ggml_is_quantized(src0->type)){
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_matrix(src0) && "src0 quantized tensor"); //
} else 
if (ggml_is_contiguous(src0)) {
} else 
if (src0->ne[0]*src0->nb[0]<src0->nb[1] && src0->nb[2]==src0->ne[1]*src0->nb[1]) {// submatrx
    // `nb[0]==_type_size()` AND `ne[0]*nb[0]< nb[1]` AND `ne[1]*nb[1]==nb[2]`
    GGML_ASSERT(src0->nb[0]==ggml_type_size(src0->type));
    GGML_ASSERT(src0->nb[1]> src0->ne[0]*src0->nb[0]);// шаг длинее строки
    GGML_ASSERT(src0->nb[2]==src0->ne[1]*src0->nb[1]);// ggml_is_contiguous_1
    GGML_ASSERT(src0->nb[3]==src0->ne[2]*src0->nb[2]);
    GGML_ASSERT(src0->ne[3]==1);// is 3d
// range = src0->nb[3];
} else if (ggml_is_permuted(src0)) {// `nb[0]==_type_size()` AND `nb[1]==ne[2]*nb[2]`
    GGML_ASSERT(src0->nb[0]==ggml_type_size(src0->type));
    GGML_ASSERT(src0->nb[2]==src0->ne[0]*src0->nb[0]);// contiguouse && permuted
    GGML_ASSERT(src0->nb[1]==src0->ne[2]*src0->nb[2]);
//    GGML_ASSERT(src0->nb[3]==src0->ne[1]*src0->nb[1]);
    GGML_ASSERT(ggml_is_3d(src1));
//    GGML_ASSERT(0 && "permuted");
} else {
    if (1) printf("MM %s(%zu,%zu,%zu,%zu)(%zu,%zu,%zu,%zu)%c %s(%zu,%zu,%zu,%zu)(%zu,%zu,%zu)%c->%s(%zu,%zu,%zu,%zu)(%zu,%zu,%zu)%c\n", 
        src0->name, src0->ne[0], src0->ne[1], src0->ne[2],src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2],src0->nb[3], ggml_is_contiguous(src0)?'c': ggml_is_permuted(src0)?'p':'n',
        src1->name, src1->ne[0], src1->ne[1], src1->ne[2],src1->ne[3], src1->nb[0], src1->nb[1], src1->nb[2], ggml_is_contiguous(src1)?'c': ggml_is_permuted(src1)?'p':'n',
        dst->name, dst->ne[0], dst->ne[1], dst->ne[2],dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], ggml_is_contiguous(src1)?'c':'n'
        ); 
    GGML_ASSERT(0 && "not defined class");
}
if (ggml_is_quantized(src1->type)){// матрицы весовых коэффициентов
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_matrix(src1) && "src1 quantized tensor"); //
} else 
if (src1->ne[0]*src1->nb[0]==src1->nb[1] && src1->nb[2]==src1->ne[1]*src1->nb[1]){// ggml_is_contiguous(src1)
    GGML_ASSERT(src1->nb[1]==src1->ne[0]*src1->nb[0]);// contiguouse && permuted
    GGML_ASSERT(src1->nb[2]==src1->ne[1]*src1->nb[1]);
    GGML_ASSERT(src1->nb[3]==src1->ne[2]*src1->nb[2]);
    GGML_ASSERT(src1->ne[3]==1);// is 3d
} else 
if (src1->ne[0]*src1->nb[0]< src1->nb[1] && src1->nb[2]==src1->ne[1]*src1->nb[1]) {// submatrx - содержит подстроки
    GGML_ASSERT(src1->nb[0]==ggml_type_size(src1->type));
    GGML_ASSERT(src1->nb[2]==src1->ne[1]*src1->nb[1]);
    GGML_ASSERT(src1->ne[3]==1);// is 3d
} else if (ggml_is_permuted(src1)) {
    GGML_ASSERT(src1->nb[0]==ggml_type_size(src1->type));
    GGML_ASSERT(src1->nb[1]==src1->ne[2]*src1->nb[2]);
    GGML_ASSERT(src1->ne[3]==1);// is 3d
    // GGML_ASSERT(0 && "permuted");
    // размер матрицы? =nb03
} else {
    GGML_ASSERT(0 && "not defined class");
}
//! классификация матриц

    if (!ggml_is_transposed(src0) &&
        !ggml_is_transposed(src1) &&
        src1t == GGML_TYPE_F32 &&
        ne00%32 == 0 &&
        ne11 > 2) {
#ifdef GGML_OPENCL_SOA_Q
        // Set up kernel.
        switch(src0t) {
            case GGML_TYPE_Q4_0:
                // This should have been satisfied.
                GGML_ASSERT(ne11 == ne1);
                GGML_ASSERT(ne01 == ne0);

                if (backend_ctx->gpu_family == INTEL) {
/*                    nth0 = 16; nth1 = 1; ndst = 8;
                    kernel = backend_ctx->kernel_mul_mat_q4_0_f32_1d_8x_flat; */
                  nth0 = 16; nth1 = 1; ndst = 16;
                    kernel = backend_ctx->kernel_mul_mat_q4_0_f32_1d_16x_flat; 
                } else if (backend_ctx->gpu_family == ADRENO) {
                    nth0 = 64; nth1 = 1; ndst = 8;
                    kernel = backend_ctx->kernel_mul_mat_q4_0_f32_1d_8x_flat;
                } else {
                    GGML_ASSERT(false && "TODO: Unknown GPU");
                }

                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->q));
                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_0->d));
                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
                CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
                CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
                CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
                CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
                CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
                CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
                CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
                break;
            default:
                break;
        }

        // Launch kernel.
        if (src0t == GGML_TYPE_Q4_0) {
            size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
            size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

            #ifdef GGML_OPENCL_PROFILING
            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));

            g_profiling_info.emplace_back();
            populateProfilingInfo(g_profiling_info.back(), evt, kernel, global_work_size, local_work_size, dst, src1);
#else
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
#endif
            return;
        }
#else  // GGML_OPENCL_SOA_Q
       // TODO: add block_q4_0 variant.
#endif // GGML_OPENCL_SOA_Q
    }

    cl_mem A_d = nullptr;
    cl_mem B_d = nullptr;
    cl_mem C_d = nullptr;
    // use custom matrix x vector kernel
    switch (src0t) {
        case GGML_TYPE_F32:{//
//            GGML_ASSERT(ne02 == ne12);
            // матрица MxKxN
            GGML_ASSERT(ne03 == ne13 && ne03==1);
            // batch
            GGML_ASSERT(ne00==ne10);// K
            GGML_ASSERT(ne01==ne0); // M dst[MxN]
            GGML_ASSERT(ne11==ne1); // N

            GGML_ASSERT(src1t == GGML_TYPE_F32);
            kernel = backend_ctx->kernel_mul_mat_f32_f32;
            nrows = 4;

            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 32;
                nth1 = 1;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }
            A_d = _get_cl_buffer(src0, CL_MEM_READ_ONLY);
            B_d = _get_cl_buffer(src1, CL_MEM_READ_ONLY);
            C_d = _get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_d));//extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &B_d));//extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &C_d));//extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
		} break;
        case GGML_TYPE_F16:{
//            GGML_ASSERT(ne02 == ne12);
            // матрица MxKxN
            GGML_ASSERT(ne03 == ne13 && ne03==1);
            // batch
            GGML_ASSERT(ne00==ne10);// K
            GGML_ASSERT(ne01==ne0); // M dst[MxN]
            GGML_ASSERT(ne11==ne1); // N
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 32;
                nth1 = 1;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            if (src1t == GGML_TYPE_F32) {
                if (ne11 * ne12 < 4) {
                    kernel = backend_ctx->kernel_mul_mat_f16_f32_1row;
                    nrows = 1;
                } else
                if (ne00 >= 128 && ne01 >= 8 && ne00%128 == 0) {
                    if (backend_ctx->gpu_family == INTEL) nth0 = 16;
                    kernel = backend_ctx->kernel_mul_mat_f16_f32_l4;
                    nrows = ne11;
                } else 
				{
                    nth0 = 16;
                    kernel = backend_ctx->kernel_mul_mat_f16_f32;
                    nrows = 4;
                }
            } else {
                kernel = backend_ctx->kernel_mul_mat_f16_f16;
                nrows = 4;
            }
/*
            region.origin = offset0; region.size =ggml_nelements(src0) * sizeof(cl_half);// bloat16_as_ushort
            A_d = clCreateSubBuffer(extra0->data_device, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
            CL_CHECK(status);
            region.origin = offset1; region.size = ggml_nelements(src1) * (src1t == GGML_TYPE_F32?sizeof(cl_float):sizeof(cl_half));
            B_d = clCreateSubBuffer(extra1->data_device, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
            CL_CHECK(status);
            region.origin = offsetd; region.size = ggml_nelements(dst) * sizeof(cl_float);
            C_d = clCreateSubBuffer(extrad->data_device, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
            CL_CHECK(status);*/
            A_d = _get_cl_buffer(src0, CL_MEM_READ_ONLY);//_get_cl_buffer_sliced
            B_d = _get_cl_buffer(src1, CL_MEM_READ_ONLY);
            C_d = _get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_d));//extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &B_d));//extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &C_d));//extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
		} break;
        case GGML_TYPE_BF16:{
            // GGML_ASSERT(ne02 == ne12);
            // матрица MxKxN
            GGML_ASSERT(ne03 == ne13 && ne03==1);
            // batch
            GGML_ASSERT(ne00==ne10);// K
            GGML_ASSERT(ne01==ne0); // M dst[MxN]
            GGML_ASSERT(ne11==ne1); // N
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 32;ndst=1;
                nth1 = 1;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }
/*            if (ne00 >= 128 && ne01 >= 8 && ne00%8 == 0) {
                kernel = backend_ctx->kernel_mul_mat_bf16_f32_l4;// работает медленно и с ошибками
                nrows = ne11;
            } else */
#if 0
            if (src0->ne[0]==src1->ne[0] && src0->ne[1]*src0->nb[1]!=src0->nb[2] && src0->nb[1]==src0->ne[2]*src0->nb[2]) {
                printf("select kernel bf16 row permute %s\n", src0->name);
            }
            if (src0->ne[0]==src1->ne[0] && src1->ne[1]*src1->nb[1]!=src1->nb[2] && src1->nb[1]==src1->ne[2]*src1->nb[2]) {
                printf("select kernel bf16 row permute %s\n", src1->name);
            }
            if (src0->ne[0]==src1->ne[0] && src0->ne[0]*src0->nb[0]< src0->nb[1] && src0->ne[1]*src0->nb[1]==src0->nb[2]) {
                printf("select kernel bf16 padded %s\n", src0->name);
                 
            }
// todo матрицы с повторами

            if (src0->ne[0]==src1->ne[0] && src0->ne[2]*src0->nb[2]!=src0->nb[3] && src0->nb[2]==src0->ne[3]*src0->nb[3]) {
                printf("select kernel bf16 submatrix permute %s\n", src0->name);
            }
            if (src0->ne[0]==src1->ne[0] && src1->ne[2]*src1->nb[2]!=src1->nb[3] && src1->nb[2]==src1->ne[3]*src1->nb[3]) {
                printf("select kernel bf16 submatrix permute %s\n", src1->name);
            }
            if (ggml_is_contiguous(src0) && ggml_is_matrix(src0) && ggml_is_vector(src1)) {
                printf("select kernel mv bf16 flat\n");
                // kernel = backend_ctx->kernel_mv[GGML_TYPE_Q8_0];
            } else
            if (!ggml_is_contiguous(src0)) {
                printf("select kernel mul_mat bf16 permuted\n");
                // kernel = backend_ctx->kernel_mv[GGML_TYPE_Q8_0];
                if (1) printf("MM %s %zu(%zu,%zu,%zu,%zu)(%zu,%zu,%zu,%zu)%c %s %zu(%zu,%zu,%zu,%zu)(%zu,%zu,%zu)%c->%s %zu(%zu,%zu,%zu,%zu)(%zu,%zu,%zu)%c\n", 
                    src0->name, offset0, src0->ne[0], src0->ne[1], src0->ne[2],src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2],src0->nb[3], ggml_is_contiguous(src0)?'c': ggml_is_permuted(src0)?'p':'n',
                    src1->name, offset1, src1->ne[0], src1->ne[1], src1->ne[2],src1->ne[3], src1->nb[0], src1->nb[1], src1->nb[2], ggml_is_contiguous(src1)?'c': ggml_is_permuted(src1)?'p':'n',
                     dst->name, offsetd,  dst->ne[0],  dst->ne[1],  dst->ne[2], dst->ne[3],  dst->nb[0],  dst->nb[1],  dst->nb[2], ggml_is_contiguous(src1)?'c':'n'
                    ); 
            } 
#endif
            {
                kernel = backend_ctx->kernel_mul_mat_bf16_f32;
                nrows = 1;
                GGML_ASSERT(ne11%nrows==0);
                GGML_ASSERT(ne01%nrows==0);
            }
            A_d = _get_cl_buffer(src0, CL_MEM_READ_ONLY);//_get_cl_buffer_sliced
            B_d = _get_cl_buffer(src1, CL_MEM_READ_ONLY);
            C_d = _get_cl_buffer(dst, CL_MEM_WRITE_ONLY);
            int i=0;
            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &A_d));//extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &B_d));//extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &C_d));//extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne00));
//            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne01));
//            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne02));
//            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint), &nb00));
            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint), &nb01));
            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint), &nb02));
//            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint), &nb03));
//            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne10));
//            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne11));
//            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne12));
//            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint), &nb10));
            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint), &nb11));
            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint), &nb12));
//            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint), &nb13));
            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &r2));
//            CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &r3));

		} break;
        case GGML_TYPE_Q8_0:{
            GGML_ASSERT(ne12*ne13==1);// batch
            GGML_ASSERT(ne02==1);   // 
            GGML_ASSERT(ne00==ne10);// K
            GGML_ASSERT(ne01==ne0); // M dst[MxN]
            GGML_ASSERT(ne11==ne1); // N
			if (backend_ctx->gpu_family == INTEL) {
                nth0 = 32; nth1 = 1; ndst = 1;
			}// INTEL
			else {
				GGML_ASSERT(false && "TODO: Unknown GPU");
			}

            A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
            B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
            C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

            if (ggml_is_contiguous(src0) && ggml_is_matrix(src0) 
            &&  ggml_is_contiguous(src1) && ggml_is_vector(src1)) {
                kernel = backend_ctx->kernel_mv[src0->type];

                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_d));
                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &B_d));
                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &C_d));
                CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_uint),  &ne00));
                CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_uint),  &nb01));
            } else {
                kernel = backend_ctx->kernel_mul_mat_q8_0_f32;
            
                int i=0;
                CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &A_d));
                CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &B_d));
                CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &C_d));
                CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne00));
                CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne01));
                CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne02));//==1
                CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne10));
                CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne12));//==1
                CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne0));
                CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),      &ne1));
            }
        } break;
        case GGML_TYPE_Q4_0:// ggml_is_matrix(src1);
            GGML_ASSERT(ne12*ne13==1);// batch - для квантов не используется
            GGML_ASSERT(ne02==1);   // 
            GGML_ASSERT(ne00==ne10);// K
            GGML_ASSERT(ne01==ne0); // M dst[MxN]
            GGML_ASSERT(ne11==ne1); // N

#ifdef GGML_OPENCL_SOA_Q
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16; nth1 = 1; ndst = 1;
                kernel = backend_ctx->kernel_mul_mat_q4_0_f32_flat;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64; nth1 = 1; ndst =8;
                kernel = backend_ctx->kernel_mul_mat_q4_0_f32_8x_flat;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
#else // GGML_OPENCL_SOA_Q
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16; nth1 = 1; ndst = 1;
                kernel = backend_ctx->kernel_mul_mat_q4_0_f32;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
            B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
            C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_d));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &B_d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &C_d));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_uint),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_uint),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_uint),      &ne02));//==1
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_uint),      &ne10));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_uint),      &ne12));//==1
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_uint),      &ne0));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_uint),      &ne1));
#endif
            break;

        case GGML_TYPE_Q4_1:
			kernel = backend_ctx->kernel_mul_mat_q4_1_f32;
            GGML_ASSERT(ne12*ne13==1);// batch
            GGML_ASSERT(ne02==1);   // 
            GGML_ASSERT(ne00==ne10);// K
            GGML_ASSERT(ne01==ne0); // M dst[MxN]
            GGML_ASSERT(ne11==ne1); // N
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 32; nth1 = 1; ndst = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }
            A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
            B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
            C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_d));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &B_d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &C_d));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_uint),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_uint),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_uint),      &ne02));//==1
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_uint),      &ne10));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_uint),      &ne12));//==1
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_uint),      &ne0));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_uint),      &ne1));
			break;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
			GGML_ASSERT(false && "not implemented");
			break;
        case GGML_TYPE_Q4_K:
            kernel = backend_ctx->kernel_mul_mv_q4_K_f32;
            GGML_ASSERT(ne12*ne13==1);// batch
            GGML_ASSERT(ne02==1);   // 
            GGML_ASSERT(ne00==ne10);// K
            GGML_ASSERT(ne01==ne0); // M dst[MxN]
            GGML_ASSERT(ne11==ne1); // N
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 1; nth1 = 32; ndst = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }
            A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
            B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
            C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_d));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &B_d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &C_d));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_uint),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_uint),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_uint),      &ne02));//== 1
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_uint),      &ne10));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_uint),      &ne12));//== 1
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_uint),      &ne0));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_uint),      &ne1));
            break;
        case GGML_TYPE_Q5_K:
            kernel = backend_ctx->kernel_mul_mv_q5_K_f32;
            GGML_ASSERT(ne12*ne13==1);// batch
            GGML_ASSERT(ne02==1);   // 
            GGML_ASSERT(ne00==ne10);// K
            GGML_ASSERT(ne01==ne0); // M dst[MxN]
            GGML_ASSERT(ne11==ne1); // N
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 1; nth1 = 32; ndst = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }
            A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
            B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
            C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_d));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &B_d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &C_d));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_uint),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_uint),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_uint),      &ne02));//== 1
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_uint),      &ne10));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_uint),      &ne12));//== 1
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_uint),      &ne0));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_uint),      &ne1));
			break;
        case GGML_TYPE_Q6_K:
            kernel = backend_ctx->kernel_mul_mv_q6_K_f32;
            GGML_ASSERT(ne12*ne13==1);// batch
            GGML_ASSERT(ne00==ne10);// K
            GGML_ASSERT(ne01==ne0); // M
            GGML_ASSERT(ne02==1);   // 1
            GGML_ASSERT(ne11==ne1); // N
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 1; nth1 = 16; ndst = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }
            A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
            B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
            C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_d));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &B_d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &C_d));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_uint),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_uint),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_uint),      &ne02));//== 1
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_uint),      &ne10));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_uint),      &ne12));//== 1
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_uint),      &ne0));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_uint),      &ne1));
            break;
        default:
            GGML_ASSERT(false && "not implemented");
    }

    if (src0t == GGML_TYPE_Q4_0 ||
        src0t == GGML_TYPE_Q4_1 ||
        src0t == GGML_TYPE_Q8_0 ||
        src0t == GGML_TYPE_Q4_K ||
        src0t == GGML_TYPE_Q5_K ||
        src0t == GGML_TYPE_Q6_K ||
        src0t == GGML_TYPE_Q3_K ||
        src0t == GGML_TYPE_Q2_K) {
        // Each SIMD group produces N_DST values in the result. Assuming each
        // workgroup has N_SIMDGROUP SIMD groups, then each workgroup will
        // produce N_DST*N_SIMDGROUP values in the result. Hence, the grid size
        // (number of workgroups) will be a nearest multiple of
        // N_DST*N_SIMDGROUP to cover the size of the dimension. Below, 4 is
        // N_DST*N_SIMDGROUP (see the kernel for Q4_0 matmul).

        // Размер расчета (число строк)/n_dst* (число_подгрупп)
        size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
        size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};// кратность, общее число должно равняться числу вычислительных блоков 512?

        cl_event event_list[GGML_MAX_SRC];
        cl_uint event_list_len = _event_list(dst, event_list);

        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
        g_profiling_info.emplace_back();
        populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, src0, src1);
#endif
    } else {
        int64_t ny = (ne11 + nrows - 1)/nrows;

        size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst *nth0, (size_t)ny*nth1, (size_t)ne12*ne13};
        size_t  local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

        cl_event event_list[GGML_MAX_SRC];
        cl_uint  event_list_len = _event_list(dst, event_list);
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
        g_profiling_info.emplace_back();
        populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, src0, src1);
#endif
    }
    if (A_d!=nullptr) CL_CHECK(clReleaseMemObject(A_d)); 
    if (B_d!=nullptr) CL_CHECK(clReleaseMemObject(B_d)); 
    if (C_d!=nullptr) CL_CHECK(clReleaseMemObject(C_d)); 
}

static void ggml_cl_scale(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_UNUSED(src1);

    GGML_ASSERT(ggml_is_contiguous(src0));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    float scale = _tensor_get_param_f32(dst, 0);
    // __builtin_memcpy(&scale, dst->op_params, sizeof(scale));


    cl_kernel kernel = backend_ctx->kernel_scale;
    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &A_d));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &C_d));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_float), &scale));

    size_t n = ggml_nelements(dst)/8;

    size_t global_work_size[] = {n, 1, 1};
    size_t  local_work_size[] = {MIN(64,n), 1, 1};
    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = _event_list(dst, event_list);

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src0);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(C_d));
}

static void ggml_cl_cpy(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);

    // GGML_OP_CPY happens between src0 and src1.
    // GGML_OP_DUP and GGML_OP_CONT happen between src0 and dst.
    UNUSED(dst);

    const int ne00 = src0 ? src0->ne[0] : 0;
    const int ne01 = src0 ? src0->ne[1] : 0;
    const int ne02 = src0 ? src0->ne[2] : 0;
    const int ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb00 = src0 ? src0->nb[0] : 0;
    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    const int ne10 = src1 ? src1->ne[0] : 0;
    const int ne11 = src1 ? src1->ne[1] : 0;
    const int ne12 = src1 ? src1->ne[2] : 0;
    const int ne13 = src1 ? src1->ne[3] : 0;

    const cl_ulong nb10 = src1 ? src1->nb[0] : 0;
    const cl_ulong nb11 = src1 ? src1->nb[1] : 0;
    const cl_ulong nb12 = src1 ? src1->nb[2] : 0;
    const cl_ulong nb13 = src1 ? src1->nb[3] : 0;

    const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
    const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;

    cl_kernel kernel;
    size_t ndst=1;
    switch (src0t) {
        case GGML_TYPE_F32:
            switch (src1t) {
                case GGML_TYPE_BF16:
                    kernel = backend_ctx->kernel_cpy_f32_bf16;
                    break;
                case GGML_TYPE_F16:
                    kernel = backend_ctx->kernel_cpy_f32_f16;
                    break;
                case GGML_TYPE_F32:
                    kernel = backend_ctx->kernel_cpy_f32_f32;
                    break;
                default:
                    GGML_ASSERT(false && "not implemented");
            }
            break;
        case GGML_TYPE_F16:
            switch (src1t) {
                case GGML_TYPE_F16:
                    kernel = backend_ctx->kernel_cpy_f16_f16;
                    break;
                case GGML_TYPE_F32:
                    kernel = backend_ctx->kernel_cpy_f16_f32;
                    break;
                default:
                    GGML_ASSERT(false && "not implemented");
            }
            break;
        case GGML_TYPE_BF16:
            switch (src1t) {
                case GGML_TYPE_BF16:
                    kernel = backend_ctx->kernel_cpy_f16_f16;
                    break;
                case GGML_TYPE_F32:
                    kernel = backend_ctx->kernel_cpy_bf16_f32;
                    break;
                default:
                    GGML_ASSERT(false && "not implemented");
            }
            break;
        default:
            GGML_ASSERT(false && "not implemented");
    }
//    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
//    cl_mem B_d =_get_cl_buffer(src1,CL_MEM_WRITE_ONLY);
 
    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne13));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));

    const int nth = MIN(64, ne00);

    size_t global_work_size[] = {(size_t)(ne01+ndst-1)/ndst*nth, (size_t)ne02, (size_t)ne03};
    size_t  local_work_size[] = {(size_t)nth, 1, 1};

    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = 0;
    event_list_len += _event_get(src0, event_list);// добавить одно событие
//    event_list_len += _event_get(src1, event_list);// добавить одно событие
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extra1->evt));

#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extra1->evt, kernel, global_work_size, local_work_size, src1, src0);
#endif
//    CL_CHECK(clReleaseMemObject(A_d));
//    CL_CHECK(clReleaseMemObject(B_d));
}

static void ggml_cl_dup(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cl_cpy(backend, src0, dst, nullptr);
    UNUSED(src1);
}

static void ggml_cl_diag_mask_inf(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    int n_past = ((int32_t *)(dst->op_params))[0];

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    if (ne00%8 == 0) {
        kernel = backend_ctx->kernel_diag_mask_inf_8;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_past));

        size_t global_work_size[] = {(size_t)ne00*ne01*ne02/8, 1, 1};
        size_t  local_work_size[] = {64, 1, 1};
// todo добавить _evt_list

        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &extrad->evt));

#ifdef GGML_OPENCL_PROFILING
        g_profiling_info.emplace_back();
        populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src1);
#endif
    } else {
        kernel = backend_ctx->kernel_diag_mask_inf;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_past));

        size_t global_work_size[] = {(size_t)ne00, (size_t)ne01, (size_t)ne02};
        size_t  local_work_size[] = {64, 1, 1};


        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
        g_profiling_info.emplace_back();
        populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src1);
#endif
    }
}

static void ggml_cl_soft_max(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    // Softmax can now fuse KQ mask and KQ scale, which used to be two additional
    // ops before softmax. It now also fuses alibi if `max_bias > 0`. For llama,
    // alibi is not used; however, for some other models, it is used.
    // KQ_mask

    if (src1) // -- пока отключил
    {
        GGML_ASSERT(src1);
        GGML_ASSERT(src1->extra);
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    const cl_uint  ne00 = src0->ne[0];
    const cl_uint  ne01 = src0->ne[1];
    const cl_uint  ne02 = src0->ne[2];
    const cl_uint  ne03 = src0->ne[3];

//    const cl_uint  nb01 = src0->nb[1];
//    const cl_uint  nb02 = src0->nb[2];

    float scale     = _tensor_get_param_f32(dst, 0);
    float max_bias  = _tensor_get_param_f32(dst, 1);
//    __builtin_memcpy(&scale,    dst->op_params + 0, sizeof(float));
//    __builtin_memcpy(&max_bias, dst->op_params + 1, sizeof(float));

    GGML_ASSERT(max_bias==0);// в нашем случае так, от этого звисит выбор варианта функции softmax

    const int nrows_x = ggml_nrows(src0);
    const int nrows_y = src0->ne[1];

    const int n_head      = nrows_x/nrows_y;// ne[2]?
    const int n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    float slope = 1.f;
    if (max_bias > 0.0f) {
        int h = ne02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // Local size must be wave size. Each workgroup is a wave, working on a row,
    // where a row corresponds to leading dimension.
    int nth = MIN(32, ne00);

    if (backend_ctx->gpu_family == INTEL) {
        // This is the same as the initial value.
        nth = 32;//MIN(32, ne00); \todo исправить чтобы параллельно запускалось по числу строк
    }
    else if (backend_ctx->gpu_family == ADRENO) {
        nth = 64;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    cl_kernel kernel;
    if (src1->type == GGML_TYPE_BF16) {
        kernel = backend_ctx->kernel_soft_max_bf16;
    } else
    if (src1->type == GGML_TYPE_F16) {
        kernel = backend_ctx->kernel_soft_max_f16;
    } else {
        kernel = backend_ctx->kernel_soft_max;
    }
    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem B_d =src1!=nullptr? _get_cl_buffer(src1,CL_MEM_READ_ONLY ): nullptr;
    cl_mem C_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_d));//extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   B_d!=nullptr? &B_d: &A_d));//extra1 ? &extra1->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &C_d));//extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_uint),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_uint),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_uint),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_float),    &scale));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_float),    &slope));
//    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(float),    &max_bias));
//    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(float),    &m0));
//    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(float),    &m1));
//    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &n_head_log2));

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t  local_work_size[] = {(size_t)nth, 1, 1};

    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = _event_list(dst, event_list);

    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));

#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src0);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    if (B_d!=nullptr) CL_CHECK(clReleaseMemObject(B_d));
    CL_CHECK(clReleaseMemObject(C_d));
}

static void ggml_cl_rope(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    ggml_tensor * src2 = dst->src[2];
    ggml_tensor_extra_cl * extra2 = src2 ? (ggml_tensor_extra_cl *)src2->extra : nullptr;

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;
    const int  ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong  nb00 = src0 ? src0->nb[0] : 0;
    const cl_ulong  nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong  nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong  nb03 = src0 ? src0->nb[3] : 0;

    const int ne10 = src1 ? src1->ne[0] : 0;
    const int ne11 = src1 ? src1->ne[1] : 0; UNUSED(ne11);
    const int ne12 = src1 ? src1->ne[2] : 0; UNUSED(ne12);
    const int ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

    const int  ne0 = dst ? dst->ne[0] : 0;
    const int  ne1 = dst ? dst->ne[1] : 0;
    const int  ne2 = dst ? dst->ne[2] : 0;
    const int  ne3 = dst ? dst->ne[3] : 0;

    const cl_ulong  nb0 = dst ? dst->nb[0] : 0;
    const cl_ulong  nb1 = dst ? dst->nb[1] : 0;
    const cl_ulong  nb2 = dst ? dst->nb[2] : 0;
    const cl_ulong  nb3 = dst ? dst->nb[3] : 0;

    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    int nth = MIN(64, ne00);

    const int n_past     = ((int *) dst->op_params)[0];
    const int n_dims     = ((int *) dst->op_params)[1];
    const int mode       = ((int *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    int32_t sections[4]; // multi 
    // переделать на _tensor_get_param_f32()
    __builtin_memcpy(&freq_base,   (int32_t *) dst->op_params + 5, sizeof(float));
    __builtin_memcpy(&freq_scale,  (int32_t *) dst->op_params + 6, sizeof(float));
    __builtin_memcpy(&ext_factor,  (int32_t *) dst->op_params + 7, sizeof(float));
    __builtin_memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    __builtin_memcpy(&beta_fast,   (int32_t *) dst->op_params + 9, sizeof(float));
    __builtin_memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    __builtin_memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int32_t)*4);

    const bool is_neox = mode & 2;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    cl_kernel kernel;

    if (is_neox)  {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_neox_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_neox_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    } else
    if (is_mrope && !is_vision) {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_multi_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_multi_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    } else 
    if (is_vision) {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_vision_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_vision_f16;
                break;
            default:
                GGML_ASSERT(false);
        }
    } else {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_norm_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_norm_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    }
    //GGML_ASSERT(extra2!=nullptr);
    cl_mem A_d =_get_cl_buffer(src0,CL_MEM_READ_ONLY );
    cl_mem B_d =_get_cl_buffer(src1,CL_MEM_READ_ONLY );
    cl_mem C_d = extra2? _get_cl_buffer(src2,CL_MEM_READ_ONLY ): nullptr;
    cl_mem D_d =_get_cl_buffer(dst, CL_MEM_WRITE_ONLY);

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_d));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &B_d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   C_d!=nullptr?&C_d:&A_d));//extra2 ? &extra2->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &D_d));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb3));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &n_past));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &n_dims));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &n_ctx_orig));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(float),    &freq_base));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(float),    &freq_scale));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(float),    &ext_factor));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(float),    &attn_factor));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(float),    &beta_fast));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(float),    &beta_slow));
    if (is_mrope || is_vision) {
        CL_CHECK(clSetKernelArg(kernel, 29, sizeof(int32_t)*4, &sections));
    }

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t  local_work_size[] = {(size_t)nth, 1, 1};

    cl_event event_list[GGML_MAX_SRC];
    cl_uint  event_list_len = _event_list(dst, event_list);

    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, event_list_len, event_list, &extrad->evt));
#ifdef GGML_OPENCL_PROFILING
    g_profiling_info.emplace_back();
    populateProfilingInfo(g_profiling_info.back(), extrad->evt, kernel, global_work_size, local_work_size, dst, src1);
#endif
    CL_CHECK(clReleaseMemObject(A_d));
    CL_CHECK(clReleaseMemObject(B_d));
    if (C_d!=nullptr) CL_CHECK(clReleaseMemObject(C_d));
    CL_CHECK(clReleaseMemObject(D_d));

}

//------------------------------------------------------------------------------
// Op offloading
//------------------------------------------------------------------------------

typedef void (*ggml_cl_func_t)(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

static bool ggml_cl_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor) {
    ggml_cl_func_t func = nullptr;

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    const bool any_on_device = tensor->extra
        || (src0 != nullptr && src0->extra)
        || (src1 != nullptr && src1->extra);

    switch (tensor->op) {
        case GGML_OP_GET_ROWS:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_get_rows;
            break;
        case GGML_OP_CPY:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_cpy;
            break;
        case GGML_OP_DUP:
            GGML_ASSERT(0 && "исключить операцию");
            break;
        case GGML_OP_CONT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_dup;
            break;
        case GGML_OP_ADD:
            if (!any_on_device) {
                return false;
            }
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ggml_is_contiguous(src1));

            func = ggml_cl_add;
            break;
        case GGML_OP_MUL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_mul;
            break;
        case GGML_OP_UNARY:// заменить на выборку по массиву
            switch (ggml_get_unary_op(tensor)) {
                case GGML_UNARY_OP_ABS:
                    func = ggml_cl_op_abs;
                    break;
                case GGML_UNARY_OP_SGN:
                    func = ggml_cl_op_sgn;
                    break;
                case GGML_UNARY_OP_NEG:
                    func = ggml_cl_op_neg;
                    break;
                case GGML_UNARY_OP_STEP:
                    func = ggml_cl_op_step;
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    func = ggml_cl_gelu_quick;
                    break;
                case GGML_UNARY_OP_TANH:
                    func = ggml_cl_tanh;
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    func = ggml_cl_sigmoid;
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    func = ggml_cl_hardsigmoid;
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    func = ggml_cl_hardswish;
                    break;
                case GGML_UNARY_OP_EXP:
                    func = ggml_cl_exp;
                    break;
                case GGML_UNARY_OP_ELU:
                    func = ggml_cl_elu;
                    break;
                case GGML_UNARY_OP_GELU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_gelu;
                    break;
                case GGML_UNARY_OP_SILU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_silu;
                    break;
                case GGML_UNARY_OP_RELU:
                    if (!any_on_device) return false;
                    func = ggml_cl_relu;
                    break;
                default:
                    return false;
            } break;
        case GGML_OP_CLAMP:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_clamp;
            break;
        case GGML_OP_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_norm;
            break;
        case GGML_OP_RMS_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_rms_norm;
            break;
        case GGML_OP_MUL_MAT:
            if (!any_on_device && !ggml_cl_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_cl_mul_mat;
            break;
        case GGML_OP_SCALE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_scale;
            break;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_nop;
            break;
        case GGML_OP_DIAG_MASK_INF:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_diag_mask_inf;
            break;
        case GGML_OP_SOFT_MAX:
            func = ggml_cl_soft_max;
            break;
        case GGML_OP_ROPE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_rope;
            break;
        default:
            return false;
    }

    func(backend, tensor->src[0], tensor->src[1], tensor);
    return true;
}
