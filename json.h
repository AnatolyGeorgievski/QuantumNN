/*! */
#ifndef JSON_H
#define JSON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <glib.h>
#include <stdint.h>
#include <stdbool.h>

#if 0 // определено в r3_asn.h
/* типы как их понимает BACNet и R3 базовый тип 4 бита.  Типы кодируются через основные типы 
и дополнительные признаки размера (ll)
+-----------+--------+--------+-----------+
| Length ll | uint   | sint   | float     |
+===========+========+========+===========+
| 0         | uint8  | sint8  | binary16  |
+-----------+--------+--------+-----------+
| 1         | uint16 | sint16 | binary32  |
+-----------+--------+--------+-----------+
| 2         | uint32 | sint32 | binary64  |
+-----------+--------+--------+-----------+
| 3         | uint64 | sint64 | binary128 |
+-----------+--------+--------+-----------+
		Table 1: Length Values 
+-------+-------------------------------------------------------+
| Field | Use                                                   |
+=======+=======================================================+
| 0b010 | the constant bits 0, 1, 0                             |
+-------+-------------------------------------------------------+
| f(2)  | 0 for integer, 1 for float                            |
+-------+-------------------------------------------------------+
| s(0)  | 0 for float or unsigned integer, 1 for signed integer |
+-------+-------------------------------------------------------+
| e     | 0 for big endian, 1 for little endian                 |
+-------+-------------------------------------------------------+
| ll    | A number for the length (Table 1).                    |
+-------+-------------------------------------------------------+
		Table 2: Bit Fields in the Low 8 Bits of the Tag
*/
enum _ASN_TYPE {// типы для внутреннего представления
ASN_TYPE_NULL		=0,
ASN_TYPE_BOOLEAN	=1,// 1:signed содержит только знак
ASN_TYPE_UNSIGNED	=2,// 2:unsigned number
ASN_TYPE_INTEGER	=3,// 1:signed + 2:number
ASN_TYPE_REAL		=4,// 5:float 
ASN_TYPE_DOUBLE		=5,// кодируется через float+ll
ASN_TYPE_OCTETS		=6,
ASN_TYPE_STRING		=7,
ASN_TYPE_BIT_STRING	=8,// битовые упакованные типы
ASN_TYPE_ENUMERATED	=9,//
ASN_TYPE_DATE		=10,
ASN_TYPE_TIME		=11,
ASN_TYPE_OID		=12,// =0xC

ASN_TYPE_CHOICE	=13,	// =0xD псевдо тип не используется в протоколе
ASN_TYPE_ANY	=14,	// =0xE псевдо тип не используется в протоколе
ASN_TYPE_UDEF	=15,	// =0xF не определенный тип
};
#define ASN_CONTEXT(n) (((n)<<4)|0x08) // контекстный тип в протоколе BACnet
#endif

//#define OBJECT_ID_MASK  0x3FFFFF // 22 бита на идентификатор
#define UNDEFINED       0x3FFFFF
enum {// базовые типы данных, перечень должен быть общий для object.h json и IR -- внутреннее представление. 
    JSON_NULL=0,
    JSON_BOOL=1,
	JSON_UINT=2,
    JSON_INT =3,
    JSON_FLOAT=5,
    JSON_STRING=7,

//	JSON_ENUM  =9,
    JSON_OID   =0xC,//0x10,

//	JSON_CHOICE=13,// выбор типа формата данных по идентификатору
//    JSON_LINK  =14,// символьные ссыкли кодируются как STRING, содержит путь в форме / - от корня, ./от текущей позиции, # зацепки anchor отмеченные меткой $id 

    JSON_OBJECT=0x0F,// конструктивный тип CBOR_MAP
    JSON_ARRAY =0x08,// Массив может быть типизованный, флаг JSON_ARRAY комбинируется с базовым типом
//	JSON_CONST =128,
};

typedef struct _JsonNode JsonNode;
typedef struct _JsonSchema JsonSchema;
struct _JsonSchema {
	GQuark tag;
    int8_t type;		//!< тип содержимого по классификации
	int8_t is_list:1;
	int8_t optional:1;
	int16_t offset;	//!< смещение данных относительно начала структуры offsetof
	int16_t size; 	//!< размер элемента данных байт
	struct _JsonSchema* next;//!< вложение
};
struct _JsonNode {
//	struct _JsonNode * next;// переделать всё чтобы избавиться от GSList
    int type;// основной тип
    GQuark tag_id;
    union {
#if 0//defined(__arm__)
        float f;
        uint32_t u;
        int32_t i;
#else
        double f;
        uint64_t u;
        int64_t i;
#endif
        char* s;
        void* ref;
         bool b; // bool
        GSList* list;// переделать в child
		//struct _JsonNode * child;
    } value;
};

static inline JsonNode* json_new(int type)
{
    JsonNode* js = (JsonNode*)g_slice_alloc0(sizeof(JsonNode));
    js->type = type;
    return js;
}
void      json_free(JsonNode* js);
void 	  json_free_antistatic(JsonNode* js);
int 	  json_patch (JsonNode* js, JsonNode* patch);
void 	  json_object_merge_patch(JsonNode* js, JsonNode* patch);
JsonNode* json_value(char* s, char** tail, GError** error);

JsonNode* json_object_get(JsonNode* js, GQuark id);

static inline GSList* json_get_object(GSList* list, GQuark id){
	while (list) {
		JsonNode* node = (JsonNode*)list->data;
		if (node->tag_id == id) {
/*			if (node->type==JSON_LINK) {
				list = node->value.list;
				node = list->data;
			} */
			break;
		}
		list=list->next;
	}
	return list;
}

/*! \breif выбрать элемент массива
	\param idx n-th array element */
static inline JsonNode* json_array_nth(JsonNode* js, int idx)
{
	GSList* list = g_slist_nth(js->value.list, idx);
    return list!=NULL? (JsonNode*)list->data: NULL;
}
static inline uint32_t json_array_length(JsonNode* js)
{
	return g_slist_length(js->value.list);
}
/*! \brief удалить элемент из списка полей объекта */
static inline
JsonNode* json_object_append(JsonNode* parent, JsonNode* js)
{
    parent->value.list = g_slist_append(parent->value.list, js);
    return parent;
}

void json_to_string(JsonNode* js, GString* str, int offset);
// API совместимо с KeyFile
void json_set_id_integer(JsonNode* js, GQuark group_id, GQuark attr_id, int32_t);// тип int может отличаться для разных платформ
void json_set_id_double (JsonNode* js, GQuark group_id, GQuark attr_id, double);
void json_set_id_value  (JsonNode* js, GQuark group_id, GQuark attr_id, const char*);
void json_set_id_boolean(JsonNode* js, GQuark group_id, GQuark attr_id, int);

int json_schema_valid(JsonSchema *, GError** error);
int json_schema_run  (JsonNode* node, const JsonSchema *schema, void* data, GError** error);

static inline float    json_object_get_float(JsonNode* js, GQuark id, float default_value){
	JsonNode* node = json_object_get(js, id);
	if (node!=NULL &&  node->type==JSON_FLOAT) 
		return node->value.f;
	return default_value;
}
static inline uint32_t json_object_get_uint(JsonNode* js, GQuark id, uint32_t default_value){
	JsonNode* node = json_object_get(js, id);
	if (node && (node->type==JSON_UINT || node->type==JSON_INT)) 
		return node->value.u;
	return default_value;
}
static inline  int32_t json_object_get_sint(JsonNode* js, GQuark id,  int32_t default_value){
	JsonNode* node = json_object_get(js, id);
	if (node && (node->type==JSON_INT || node->type==JSON_UINT)) 
		return node->value.i;
	return default_value;
}
static inline const char* json_object_get_string(JsonNode* js, GQuark id, const char* default_value){
	JsonNode* node = json_object_get(js, id);
	if (node!=NULL && (node->type==JSON_STRING)) 
		return node->value.s;
	return default_value;
}
static inline GSList*  json_object_get_array(JsonNode* js, GQuark id){
	JsonNode* node = json_object_get(js, id);
	if (node!=NULL && (node->type==JSON_ARRAY || node->type==JSON_OBJECT)) 
		return node->value.list;
	return NULL;
}
static inline GSList* json_set_uint(GSList* root, GQuark id, uint64_t value){
	GSList* list = root;
	if (id  !=0 && root!=NULL) list = json_get_object(root, id);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_UINT || node->type==JSON_INT)
			node->value.u = value;
	} else {
	    JsonNode* node = (JsonNode*)g_slice_alloc0(sizeof(JsonNode));
	    node->type 	  = JSON_UINT;
		node->value.u = value;
		node->tag_id  = id;
		root = g_slist_append(root, node);
	}
	return root;
}
static inline GSList* json_set_string(GSList* root, GQuark id, char* value){
	GSList* list = root;
	if (id  !=0 && root!=NULL) list = json_get_object(root, id);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_STRING)
			node->value.s = value;
	} else {
	    JsonNode* node = (JsonNode*)g_slice_alloc0(sizeof(JsonNode));
	    node->type 	  = JSON_STRING;
		node->value.s = value;
		node->tag_id  = id;
		root = g_slist_append(root, node);
	}
	return root;
}
#if 1// json_get_*(list) Интерфейс работает со списками 
static inline uint64_t json_get_oid (GSList* list, GQuark id, uint32_t default_value){
	if (id  !=0) list = json_get_object(list, id);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_OID)
			return node->value.u;
	}
	return default_value;
}
static inline uint64_t json_get_uint(GSList* list, GQuark id, uint64_t default_value){
	if (id  !=0) list = json_get_object(list, id);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_UINT || node->type==JSON_INT)
			return node->value.u;
	}
	return default_value;
}
static inline  int64_t json_get_sint(GSList* list, GQuark id, int64_t default_value){
	if (id  !=0) list = json_get_object(list, id);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_UINT || node->type==JSON_INT)
			return node->value.i;
	}
	return default_value;
}
static inline float    json_get_float(GSList* list, GQuark id, float default_value){
	if (id  !=0) list = json_get_object(list, id);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_FLOAT)
			return node->value.f;
	}
	return default_value;
}
static inline const char* json_get_string(GSList* list, GQuark id, const char* default_value){
	if (id  !=0) list = json_get_object(list, id);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_STRING)
			return node->value.s;
	}
	return default_value;
}
/*! \brief по списку параметров находит объект или список с данным идентификатором 
	\param id идентификатор тега
	\return возвращает список или содержимое объекта */
static inline GSList* json_get_array(GSList* list, GQuark id){
	if (id  !=0) list = json_get_object(list, id);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_ARRAY || node->type==JSON_OBJECT)
			return node->value.list;
	}
	return NULL;
}
#endif
/*
static inline GSList* json_array_get(GSList* list, GQuark id){
	if (id  !=0) list = json_get_object(list, id);
	if (list!=NULL){
		JsonNode* node  = list->data;
		if (node->type==JSON_ARRAY || node->type==JSON_OBJECT)
			return node->value.list;
	}
	return NULL;
}*/
void json_array_to_string(GSList* list, GString* str, int offset);
static inline const char* json_array_get_string(GSList* list, int idx, const char* default_value){
	if (idx !=0) list = g_slist_nth(list, idx);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_STRING)
			return node->value.s;
	}
	return default_value;
}
static inline uint32_t json_array_get_uint(GSList* list, int idx, uint32_t default_value){
	if (idx !=0) list = g_slist_nth(list, idx);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_UINT || node->type==JSON_INT)
			return node->value.u;
	}
	return default_value;
}
static inline int32_t json_array_get_int(GSList* list, int idx, int32_t default_value){
	if (idx !=0) list = g_slist_nth(list, idx);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_INT || node->type==JSON_UINT)
			return node->value.i;
	}
	return default_value;
}
static inline float json_array_get_float(GSList* list, int idx, float default_value){
	if (idx !=0) list = g_slist_nth(list, idx);
	if (list!=NULL){
		JsonNode* node  = (JsonNode*)list->data;
		if (node->type==JSON_FLOAT)
			return node->value.f;
	}
	return default_value;
}
#ifdef __cplusplus
}
#endif
#endif // JSON_H
