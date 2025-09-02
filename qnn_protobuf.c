/*!
qnn_protobuf.c

 */
#include <stdint.h>
#include <stdio.h>

#include "qnn_protobuf.h"
#include <glib.h>


#define _OPT     0  //!< опциональный
#define _REP     1  //!< последовательность
#define _REQ     2  //!< обязательный
#define _ONEOF   4  //!< один из вариантов
#define _PACKED  8  //!< упакованный
enum { // упорядочить как в iot_*.h
    _NULL,
    _UINT32,
    _INT32,
    _FLOAT,
    _UINT64,
    _INT64,
    _DOUBLE,
    _ENUM,
    _STRING,
    _STRUCT,
    _SEQUENCE,
    _CHOICE
};

#define proto_new(typ)  ({Protobuf_t * pb = g_slice_alloc0(sizeof(Protobuf_t)); pb->type = typ; pb;})
static 
const message_t* qnn_proto_oneof(const message_t* m, uint32_t  id){
    if (m) while (m->name!=NULL) {
        if (m->field_num == id) break;
        m++;
    }
    return m;
}
static
const message_t* qnn_proto_type(const message_t* m, uint32_t id){
    if (m) while (m->name!=NULL){
        if (m->field_num == id) break;
        m++;
    }
    return m;
}
/*! \brief декодирование сообщения в структуру Protobuf
    \param buf - указатель на буфер с данными
    \param size - размер буфера
    \param tail - указатель конец разбора с остатком данных 
    \param msg - указатель на структуру сообщения в кодировке protobuf
    \param level - уровень вложенности
    \return указатель на структуру Protobuf (список элементов)
 */
GSList * qnn_proto_decode( uint8_t *buf, size_t size, uint8_t ** tail, 
        const message_t* msg,  int level)
{
    uint8_t *s = buf;
    GSList* list = NULL; 
    const message_t* m= msg;
    while ((buf - s) < size){
        uint64_t v=0;
        uint64_t len=0;
        buf = proto_varint(buf, &v);
        uint8_t  wire_type = PROTO_WIRE_TYPE(v);
        unsigned field_num = PROTO_FIELD_NUMBER(v);
        Protobuf_t *node = proto_new(PB_NULL);
        node->id = field_num;// позиционное кодирование
        m = qnn_proto_type(msg, field_num);
        const char* name = m!=NULL && m->name? m->name : "";
        int offs = level*2+2;
        switch(wire_type){
        case PROTO_WIRE_TYPE_VARINT: 
            buf = proto_varint(buf, &v);
            node->type = PB_UINT;
            node->value.u = v;
            if (m->type == _ENUM){
                const message_t * ref = qnn_proto_oneof(m->ref, v);
                printf("%*d:var %lld '%s': %s\n", offs, node->id, node->value.u, name, ref->name);
            } else
                printf("%*d:var %lld '%s'\n", offs, node->id, node->value.u, name);
            break;
        case PROTO_WIRE_TYPE_LEN: 
            buf = proto_varint(buf, &len);
            printf("%*d:len=%d '%s':", offs, node->id, (int)len, name);
            if (m==NULL) {
                printf("\n");
            } else
            if (m->type == _STRING) {
                node->type = PB_STRING;
                node->value.s = buf; 
                printf("'%-.*s'\n", (int)len, node->value.s);
            } else
            if (m->type == _CHOICE) {
                node->type = PB_OBJECT;
                uint64_t choice_tag= 1;
                uint64_t len2= 0;
                uint8_t *buf2 = proto_varint(buf, &choice_tag);
                buf2 = proto_varint(buf2, &len2);
                const message_t * ref = qnn_proto_oneof(m->ref, PROTO_FIELD_NUMBER(choice_tag));
                printf(" choice:%d'%s'\n", choice_tag, (ref!=NULL && ref->name)?ref->name : "");
                node->value.list = qnn_proto_decode(buf2, len2, tail, ref->ref, level+1);
                buf2+=len2;
            } else
            if (m->type == _STRUCT) {
                node->type = PB_OBJECT;
                printf("{\n");
                node->value.list = qnn_proto_decode(buf, len, tail, m->ref, level+1);
                printf("%*s}\n", offs, "");
            }
            buf+=len;
            // копировать данные не будем
            break;
        case PROTO_WIRE_TYPE_I32:
            node->type = PB_UINT;
            node->value.u = *(uint32_t*)buf; buf+=4;
            printf("..%d:i32 %lld\n", node->id, node->value.u);
            break;
        case PROTO_WIRE_TYPE_I64:
            node->type = PB_UINT;
            node->value.u = *(uint64_t*)buf; buf+=8;
            printf("..%d:i64 %lld\n", node->id, node->value.u);
            break;
        case PROTO_WIRE_TYPE_SGROUP:// deprecated
            node->type = PB_OBJECT;
            node->value.list = qnn_proto_decode(buf, size, &buf, m->ref, level+1);
            printf("..%d: {}\n", node->id);
            break;
        case PROTO_WIRE_TYPE_EGROUP:
            *tail = buf;
            return list;
        default: // ошибка
            printf("..fail\n");
            break;
        }
        list = g_slist_append(list, node);
    }
    *tail = buf;
    return list;
}

// сообщения ONNX
enum _AttributeType {
    UNDEFINED   = 0,
    FLOAT       = 1,
    INT         = 2,
    STRING      = 3,
    TENSOR      = 4,
    GRAPH       = 5,
    FLOATS      = 6,
    INTS        = 7,
    STRINGS     = 8,
    TENSORS     = 9,
    GRAPHS      = 10,
    SPARSE_TENSOR = 11,
    SPARSE_TENSORS = 12,
    TYPE_PROTO  = 13,
    TYPE_PROTOS = 14,
};

// forward declarations
static message_t TypeProto[];
static message_t NodeProto[];
static message_t GraphProto[];
static message_t TensorProto[];
static message_t OperatorProto[];
static message_t FunctionProto[];
static message_t ValueInfoProto[];
static message_t AttributeProto[];
static message_t OperatorSetIdProto[];
static message_t StringStringEntryProto[];
static message_t NodeDeviceConfigurationProto[];
static message_t DeviceConfigurationProto[];
static message_t TrainingInfoProto[];
static message_t SparseTensorProto[];
static message_t TensorAnnotation [];//?
static message_t ShardingSpecProto[];
static message_t ShardedDimProto  [];
static message_t SimpleShardedDimProto[];

static message_t OperatorProto[] = {
  {"op_type", 1, _STRING, _OPT},
  {"since_version", 2, _INT64, _OPT},
  {"status", 3, _STRING, _OPT},
  {"doc_string", 10, _STRING, _OPT},
  {NULL}
}; 
static message_t OperatorSetProto [] = {
  {"magic", 1, _STRING},
  {"ir_version", 2, _INT64},
  {"ir_version_prerelease", 3, _STRING},
  {"ir_build_metadata", 7, _STRING},
  {"domain", 4, _STRING},
  {"opset_version", 5, _INT64},
  {"doc_string", 6, _STRING},
  {"operator",  8, _STRUCT, _REP, .ref = OperatorProto},
  {"functions", 9, _STRUCT, _REP, .ref = FunctionProto},
  {NULL}
};
static message_t NodeProto[] = {
  {"input", 1, _STRING, _REP},   // namespace Value
  {"output",2, _STRING, _REP},   // namespace Value
  {"name",  3, _STRING, _OPT},   // namespace Node
  {"op_type",4, _STRING, _OPT},
  {"domain",  7, _STRING, _OPT},  // namespace OperatorSet
  {"overload", 8, _STRING, _OPT},
  {"attribute", 5, _STRUCT, _REP, .ref = AttributeProto},
  {"doc_string", 6, _STRING, _OPT},
  {"metadata_props", 9, _STRUCT, _REP, .ref = StringStringEntryProto},
  {"device_configurations", 10, _STRUCT, _REP, .ref = NodeDeviceConfigurationProto},
  {NULL}
};
static
message_t Segment[] = {
  {"begin", 1, _INT64, _OPT},
  {"end", 2, _INT64, _OPT},
  {NULL}
};
static 
message_t DataLocation[] = {
  {"DEFAULT", 0},
  {"EXTERNAL", 1},
  {NULL}
};
static 
message_t AttributeType[] = {
  {"UNDEFINED", 0},
  {"FLOAT", 1},
  {"INT", 2},
  {"STRING", 3},
  {"TENSOR", 4},
  {"GRAPH", 5},
  {"FLOATS", 6},
  {"INTS", 7},
  {"STRINGS", 8},
  {"TENSORS", 9},
  {"GRAPHS", 10},
  {"SPARSE_TENSOR", 11},
  {"SPARSE_TENSORS", 12},
  {"TYPE_PROTO", 13},
  {"TYPE_PROTOS", 14},
  {NULL}
};
static
message_t TensorProto[] = {
  {"dims", 1, _INT64, _REP},
  {"data_type", 2, _INT32, _OPT},
  {"segment",   3, _STRUCT, _OPT, .ref = Segment},
  {"float_data", 4, _FLOAT, _REP|_PACKED},
  {"int32_data", 5, _INT32, _REP|_PACKED},
  {"string_data", 6, _STRING, _REP},
  {"int64_data", 7, _INT64, _REP|_PACKED},
  {"name", 8, _STRING, _OPT}, // namespace Value
  {"doc_string", 12, _STRING, _OPT},
  {"raw_data", 9, _STRING, _OPT},
  {"external_data", 13, _STRUCT, _REP, .ref = StringStringEntryProto},
  {"data_location", 14, _ENUM, _OPT, .ref = DataLocation},
  {"double_data", 10, _DOUBLE, _REP|_PACKED},
  {"uint64_data", 11, _UINT64, _REP|_PACKED},
  {"metadata_props", 16, _STRUCT, _REP, .ref = StringStringEntryProto},
  {NULL}
};
static
message_t GraphProto[] = {
  {"node", 1, _STRUCT, _REP, .ref = NodeProto},
  {"name", 2, _STRING, _OPT},   // namespace Graph
  {"initializer", 5, _STRUCT, _REP, .ref = TensorProto},
  {"sparse_initializer", 15, _STRUCT, _REP, .ref = SparseTensorProto},
  {"doc_string", 10, _STRING, _OPT},
  {"input", 11, _STRUCT, _REP, .ref = ValueInfoProto},
  {"output", 12, _STRUCT, _REP, .ref = ValueInfoProto},
  {"value_info", 13, _STRUCT, _REP, .ref = ValueInfoProto},
  {"quantization_annotation", 14, _STRUCT, _REP, .ref = TensorAnnotation},
  {"metadata_props", 16, _STRUCT, _REP, .ref = StringStringEntryProto},
  {NULL}
};
static
message_t ModelProto[] = {
  {"ir_version", 1, _INT64},
  {"opset_import", 8, _STRUCT, _REP, .ref = OperatorSetIdProto},
  {"producer_name", 2, _STRING},
  {"producer_version", 3, _STRING},
  {"domain", 4, _STRING},
  {"model_version", 5, _INT64},
  {"doc_string", 6, _STRING},
  {"graph", 7, _STRUCT, .ref = GraphProto},
  {"metadata_props", 14, _STRUCT, _REP, .ref = StringStringEntryProto},
  {"training_info", 20, _STRUCT, _REP, .ref = TrainingInfoProto},
  {"functions", 25, _STRUCT, _REP, .ref = FunctionProto},
  {"configuration", 26, _STRUCT, _REP, .ref = DeviceConfigurationProto},
  {NULL}
};
static
message_t TrainingInfoProto[]= {
  {"initialization", 1, _STRUCT, .ref = GraphProto},
  {"algorithm", 2, _STRUCT, .ref = GraphProto},
  {"initialization_binding", 3, _STRUCT, _REP, .ref = StringStringEntryProto},
  {"update_binding", 4, _STRUCT, _REP, .ref = StringStringEntryProto},
  {NULL}
};
static
message_t AttributeProto [] = {
  {"name", 1, _STRING, _OPT},
  {"ref_attr_name", 21, _STRING, _OPT},
  {"doc_string", 13, _STRING, _OPT},
  {"type", 20, _ENUM, _OPT, .ref = AttributeType},
  {"f", 2, _FLOAT, _OPT},
  {"i", 3, _INT64, _OPT},
  {"s", 4, _STRING, _OPT},
  {"t", 5, _STRUCT, _OPT, .ref = TensorProto},
  {"g", 6, _STRUCT, _OPT, .ref = GraphProto},
  {"sparse_tensor", 22, _STRUCT, _OPT, .ref = SparseTensorProto},
  {"tp", 14, _STRUCT, _OPT, .ref = TypeProto},
  {"floats", 7, _FLOAT, _REP},
  {"ints", 8, _INT64, _REP},
  {"strings", 9, _STRING, _REP},
  {"tensors", 10, _STRUCT, _REP, .ref = TensorProto},
  {"graphs", 11, _STRUCT, _REP, .ref = GraphProto},
  {"sparse_tensors", 23, _STRUCT, _REP, .ref = SparseTensorProto},
  {"type_protos", 15, _STRUCT, _REP, .ref = TypeProto},
  {NULL}
};
static
message_t ValueInfoProto [] = {
  {"name", 1, _STRING, _OPT},
  {"type", 2, _CHOICE, _OPT, .ref = TypeProto},
  {"doc_string", 3, _STRING, _OPT},
  {"metadata_props", 4, _STRUCT, _REP, .ref = StringStringEntryProto},
  {NULL}
};
static
message_t DeviceConfigurationProto[] = {
  {"name", 1, _STRING, _OPT},
  {"num_devices", 2, _INT32, _OPT},
  {"device", 3, _STRING, _REP},
  {NULL}
};
static
message_t StringStringEntryProto [] = {
  {"key", 1, _STRING, _OPT},
  {"value", 2, _STRING, _OPT},
  {NULL}
};
static
message_t TensorAnnotation [] = {
  {"tensor_name", 1, _STRING, _OPT},
  {"quant_parameter_tensor_names", 2, _STRUCT, _REP, .ref = StringStringEntryProto},
  {NULL}
};
static
message_t NodeDeviceConfigurationProto [] = {
  {"configuration_id", 1, _STRING, _OPT},
  {"sharding_spec", 2, _STRUCT, _REP, .ref = ShardingSpecProto},
  {"pipeline_stage", 3, _INT32, _OPT},
  {NULL}
};
static
message_t IntIntListEntryProto [] = {
  {"key", 1, _INT64, _OPT},
  {"value", 2, _INT64, _REP},
  {NULL}
};
static
message_t ShardingSpecProto [] = {
  {"tensor_name", 1, _STRING, _OPT},
  {"device", 2, _INT64, _REP},
  {"index_to_device_group_map", 3, _STRUCT, _REP, .ref = IntIntListEntryProto},
  {"sharded_dim", 4, _STRUCT, _REP, .ref = ShardedDimProto},
  {NULL}
};
static
message_t ShardedDimProto [] = {
  {"axis", 1, _INT64, _OPT},
  {"simple_sharding", 2, _STRUCT, _REP, .ref = SimpleShardedDimProto},
  {NULL}
};
static
message_t SimpleShardedDimProto [] = {
  {"dim_value", 1, _INT64, _ONEOF},
  {"dim_param", 2, _STRING, _ONEOF},
  {"num_shards", 3, _INT64, _OPT},
  {NULL}
};
static
message_t SparseTensorProto [] = {
  {"values", 1, _STRUCT, _OPT, .ref = TensorProto},
  {"indices", 2, _STRUCT, _OPT, .ref = TensorProto},
  {"dims", 3, _INT64, _REP},
  {NULL}
};
static
message_t DimensionProto [] = {
  {"dim_value", 1, _INT64, _ONEOF},
  {"dim_param", 2, _STRING, _ONEOF},
  {"denotation", 3, _STRING, _OPT},
  {NULL}
};
static
message_t TensorShapeProto [] = {
  {"dim", 1, _STRUCT, _REP, .ref = DimensionProto},
  {NULL}
};


static message_t TensorDataType[] = { // форматы ONNX - перенести
  {"FLOAT", 1},
  {"UINT8", 2},
  {"INT8",  3},
  {"UINT16",4},
  {"INT16", 5},
  {"INT32", 6},
  {"INT64", 7},
  {"STRING",8},
  {"BOOL",  9},
  {"FLOAT16", 10},
  {"DOUBLE",  11},
  {"UINT32",  12}, 
  {"UINT64",  13},
  {"COMPLEX64",  14},
  {"COMPLEX128", 15},
  {"BFLOAT16",      16},
  {"FLOAT8E4M3FN",  17},
  {"FLOAT8E4M3FNUZ",18},
  {"FLOAT8E5M2",    19},
  {"FLOAT8E5M2FNUZ",20},
  {"UINT4",         21},
  {"INT4",          22},
  {"FLOAT4E2M1",    23},
  {"FLOAT8E8M0",    24},
  {NULL},
};

static message_t _Tensor [] = {
  {"elem_type", 1, _ENUM, _OPT, .ref = TensorDataType}, // enum ONNX_PROTO
  {"shape", 2, _STRUCT, _OPT, .ref = TensorShapeProto},
  {NULL}
};
static message_t _Sequence [] = {
  {"elem_type", 1, _STRUCT, _OPT, .ref = TypeProto},
  {NULL}
};
static message_t _Map [] = {
  {"key_type", 1, _INT32, _OPT},
  {"value_type", 2, _STRUCT, _OPT, .ref = TypeProto},
  {NULL} 
};
static message_t _Optional [] = {
  {"elem_type", 1, _STRUCT, _OPT, .ref = TypeProto},
  {NULL}
};
static
message_t _SparseTensor [] = {
  {"elem_type", 1, _INT32, _OPT},
  {"shape", 2, _STRUCT, _OPT, .ref = TensorShapeProto},
  {NULL}
};
static
message_t TypeProto [] = {
  {"tensor_type", 1, _SEQUENCE, _REQ, .ref = _Tensor},
  {"sequence_type", 4, _SEQUENCE, _REQ, .ref = _Sequence},
  {"map_type", 5, _SEQUENCE, _REQ, .ref = _Map},
  {"optional_type", 9, _SEQUENCE, _REQ, .ref = _Optional},
  {"sparse_tensor_type", 8, _SEQUENCE, _REQ, .ref = _SparseTensor},
//  {"opaque_type", 7, _SEQUENCE, _REQ, .ref = _Opaque},
  {NULL}
};
static
message_t OperatorSetIdProto [] = {
  {"domain", 1, _STRING, _OPT},
  {"version", 2, _INT64, _OPT},
  {NULL}
};
static
message_t OperatorStatus [] = {
  {"EXPERIMENTAL", 0},
  {"STABLE", 1}, 
  {NULL}
};
static
message_t FunctionProto [] = {
  {"name", 1, _STRING, _OPT},
  {"input", 4, _STRING, _REP},
  {"output", 5, _STRING, _REP},
  {"attribute", 6, _STRING, _REP},
  {"attribute_proto", 11, _STRUCT, _REP, .ref = AttributeProto},
  {"node", 7, _STRUCT, _REP, .ref = NodeProto},
  {"doc_string", 8, _STRING, _OPT},
  {"opset_import", 9, _STRUCT, _REP, .ref = OperatorSetIdProto},
  {"domain", 10, _STRING, _OPT},
  {"overload", 13, _STRING, _OPT},
  {"value_info", 12, _STRUCT, _REP, .ref = ValueInfoProto},
  {"metadata_props", 14, _STRUCT, _REP, .ref = StringStringEntryProto},
  {NULL}
};


static
message_t MainProto [] = {
  {"version", 1, _INT32},
  {"name", 2, _STRING},
  {"graph", 7, _STRUCT, .ref=GraphProto},
  {"opset_import", 8, _STRUCT, .ref=OperatorSetIdProto},
  {NULL}
};
/*! \brief отладочная печать структуры Protobuf
 */
void qnn_proto_print(GSList* list){
    while (list){
        Protobuf_t* pb = (Protobuf_t*) list->data;
        switch (pb->type){
        case PB_INT:  printf("int %lld\n", pb->value.i);
            break;
        case PB_UINT: printf("%d:uint %llu\n", pb->id, pb->value.u);
            break;
        case PB_STRING: 
            
            printf("%d:str \"%s\"\n", pb->id, pb->value.s);
            break;
        case PB_OBJECT: 
            printf("%d:obj {\n", pb->id);
            printf("}\n");
            break;
        case PB_NULL: printf("null\n");
            break;
        default:
            printf("unknown\n");
            break;
        }
        list = list->next;
    }
}
#if defined(TEST_PROTOBUF)

int main(int argc, char **argv)
{
    uint8_t *buf = NULL;

    size_t size = 0;
    if (argc > 1){
        FILE *fp = fopen(argv[1], "rb");
        if (fp==NULL) return -1;

        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        buf = malloc(size);
        fread(buf, 1, size, fp);
        uint8_t *tail = NULL;
        GSList* list = NULL;
        list = qnn_proto_decode(buf, size, &tail, MainProto, 0);
        printf("len = %zu, %zd ..done\n", size, tail - buf);
        //qnn_proto_print(list);
        fclose(fp);
        free(buf);
    } else 
        return -1;
    return 0;
}
#endif
