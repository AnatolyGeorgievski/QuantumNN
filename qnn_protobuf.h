//qnn_protobuf.h
#ifndef QNN_PROTOBUF_H
#define QNN_PROTOBUF_H

#include <stddef.h>
#include <stdint.h>
#include <glib.h>

#ifdef __cplusplus
extern "C" {
#endif


enum ONNX_PROTO { // форматы ONNX - перенести
    ONNX_PROTO_FLOAT  = 1,
    ONNX_PROTO_UINT8  = 2,
    ONNX_PROTO_INT8   = 3,
    ONNX_PROTO_UINT16 = 4,
    ONNX_PROTO_INT16  = 5,
    ONNX_PROTO_INT32  = 6,
    ONNX_PROTO_INT64  = 7,
    ONNX_PROTO_STRING = 8,
    ONNX_PROTO_BOOL   = 9,
    ONNX_PROTO_FLOAT16        = 10,
    ONNX_PROTO_DOUBLE         = 11,
    ONNX_PROTO_UINT32         = 12,
    ONNX_PROTO_UINT64         = 13,
    ONNX_PROTO_COMPLEX64      = 14,
    ONNX_PROTO_COMPLEX128     = 15,
    ONNX_PROTO_BFLOAT16       = 16,
    ONNX_PROTO_FLOAT8E4M3FN   = 17,
    ONNX_PROTO_FLOAT8E4M3FNUZ = 18,
    ONNX_PROTO_FLOAT8E5M2     = 19,
    ONNX_PROTO_FLOAT8E5M2FNUZ = 20,
    ONNX_PROTO_UINT4          = 21,
    ONNX_PROTO_INT4           = 22,
    ONNX_PROTO_FLOAT4E2M1     = 23,
    ONNX_PROTO_FLOAT8E8M0     = 24,
};
/*! \brief Decode a varint from a buffer */
static inline uint8_t* proto_varint(uint8_t* buf, uint64_t *val) {
    uint64_t v = 0;
    int i=0;
    do {
        v |= (uint64_t)(buf[0] & 0x7fu)<<(i*7);
        i++;
    } while ((*buf++ & 0x80u)!=0 && i<8);
    *val = v;
    return buf;
}
#define PROTO_FIELD_NUMBER(v)   ((v)>> 3)
#define PROTO_WIRE_TYPE(v)      ((v)&0x7)
// декодирование целых со знаком после VARINT
#define PROTO_SINT(v)      ((v)&1? ~((uint64_t)(v) >> 1) : ((uint64_t)(v) >> 1))
// The “tag” of a record is encoded as a `varint` formed from the field number and the wire type via 
//    the formula: `(field_number << 3) | wire_type`
enum PROTO_TAG {
	PROTO_WIRE_TYPE_VARINT,//	int32, int64, uint32, uint64, sint32, sint64, bool, enum
	PROTO_WIRE_TYPE_I64   ,//	fixed64, sfixed64, double
	PROTO_WIRE_TYPE_LEN	  ,//   string, bytes, embedded messages, packed repeated fields
	PROTO_WIRE_TYPE_SGROUP,//	group start (deprecated)
	PROTO_WIRE_TYPE_EGROUP,//	group end (deprecated)
	PROTO_WIRE_TYPE_I32	  ,//   fixed32, sfixed32, float
};
/*
typedef struct _GSList GSList;
struct _GSList {
    void *data;
    struct _GSList* next;
}; */ 
typedef struct _Protobuf Protobuf_t;
struct _Protobuf {
    uint32_t type:8;
    uint32_t id  :24;
    union _PB_value {
        uint64_t u;
         int64_t i;
        double   f;
        const char* s;
        struct _GSList*  list;
    } value;
};

typedef struct  _message_pb message_t;
struct _message_pb {
	const char *name;	// 
	uint16_t field_num	:16;//! 0..8192 - контекстный идентификатор
	uint8_t  type   :4;	//!< тип данных BASE (3 бита) + ARRAY (типизованные массивы)
	uint8_t	 flags	:4; // optional required repeated
  union {
    struct _message_pb* ref;//!< ссылка на сообщение(список полей)
    union _PB_value default_value;
  };
};
enum {PB_NULL, PB_UINT, PB_INT, PB_FLOAT, PB_DOUBLE, PB_STRING, PB_OBJECT, PB_CHOICE};
GSList * qnn_proto_decode( uint8_t *buf, size_t size, uint8_t ** tail, 
        const message_t* msg,  int level);
#ifdef __cplusplus
}
#endif
#endif// QNN_PROTOBUF_H