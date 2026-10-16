/*! \brief Формат CBOR
2023 Anatoly Georgievskii <anatoly.georgievski@gmail.com>
\see [RFC 8949][STD94]

\see [RFC 8746] Concise Binary Object Representation (CBOR) Tags for Typed Arrays

\TODO кодирование типов данных CBOR_odt и упаковка float -> half
 */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "cbor.h"
//#include "json.h"
//#include "iot_objects.h"

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
static inline uint64_t ntohll(uint64_t a){
    return __builtin_bswap64(a);// LE
}
static inline uint64_t htonll(uint64_t a){
    return __builtin_bswap64(a);// LE
}
static inline uint32_t ntohl (uint32_t a){
    return __builtin_bswap32(a);// LE
}
static inline uint32_t htonl (uint32_t a){
    return __builtin_bswap32(a);// LE
}
static inline uint16_t ntohs (uint16_t a){
    return __builtin_bswap16(a);// LE
}
static inline uint16_t htons (uint16_t a){
    return __builtin_bswap16(a);// LE
}
#else
static inline uint64_t htonll(uint64_t a){
    return (a);// BE
}
static inline uint64_t ntohll(uint64_t a){
    return (a);// BE
}
static inline uint32_t htonl (uint32_t a){
    return (a);// BE
}
static inline uint32_t ntohl (uint32_t a){
    return (a);// BE
}
static inline uint16_t htons (uint16_t a){
    return (a);// BE
}
static inline uint16_t ntohs (uint16_t a){
    return (a);// BE
}
#endif


/*! \brief Декодирование базовых типов

\see [CBOR] Appendix B. Jump Table for Initial Byte

Major
Type 	Meaning 		Content
0 		unsigned int 	N -
1 		negative int	-1-N 	-
2 		byte string 	N bytes
3 		text string 	N bytes (UTF-8 text)
4 		array 			N data items (elements)
5 		map 			2N data items (key/value pairs)
6 		tag of number N 1 data item
7 		simple/float 	-
 */
uint8_t* cbor_decode_tag(uint8_t* buf, uint8_t *tag, uint64_t *value){
	uint8_t ib = *buf++;
	*tag = ib;
    switch(ib&0x1F) {// ориентация байтов BE - сетевой порядок байт
    case 0 ... 23: *value = ib&0x1F; break;
    case 24: *value = *(uint8_t *)buf; buf+=1; break;
    case 25: *value = ntohs (*(uint16_t*)buf); buf+=2; break;
    case 26: *value = ntohl (*(uint32_t*)buf); buf+=4; break;
    case 27: *value = ntohll(*(uint64_t*)buf); buf+=8; break;
	case 28 ... 30: break;
    case 31: *value = ~0ULL; break;
    }
	return buf;
}
static uint8_t* cbor_decode_len(uint8_t* buf, size_t *len){
	uint8_t ib = *buf++ & 0x1F;
    if (ib<24) { *len = ib;
    } else if (ib==24) {
        *len = *buf++;
    } else if (ib==25) {
        *len = buf[0]<<8 | buf[1];
        buf+=2;
    }
	return buf;
}

/*! \brief Кодирование базовых типов
 кодирование поля включает тэг, и длину. При кодировании целых типов в поле val записывается значение */
uint8_t* cbor_encode_tag (uint8_t* buf, uint8_t tag, uint64_t val)
{//
	if (val<=23)
		*buf++ = (tag<<5)|((uint8_t)val);
	else {
		unsigned int sz = (__builtin_clzll(val))>>3;// число байт для представления, функция возвращает число нулей в слове.
		//*buf++ = (tag<<5)|(24+sz);
		/* All multi-byte values are encoded in network byte order (that is, most significant byte first, also known as "big-endian"). */
		switch(sz) {// ориентация байтов BE
		case 7: *buf++ = (tag<<5)|24;
			*(uint8_t *)buf = val; buf+=1; break;
		case 6: *buf++ = (tag<<5)|25;
			*(uint16_t*)buf = htons (val); buf+=2; break;
		case 4 ... 5: *buf++ = (tag<<5)|26;
			*(uint32_t*)buf = htonl (val); buf+=4; break;
		case 0 ... 3: *buf++ = (tag<<5)|27;
			*(uint64_t*)buf = htonll(val); buf+=8; break;
		}
	}
	return buf;
}
#include <math.h>
/*
Exponent 	fraction = 0 	fraction ≠ 0 	Equation
00H = 000000002 	±zero 	subnormal number 	( − 1 )^sign × 2^{−126} × 0.fraction
01H, ..., FEH = 000000012, ..., 111111102 	normal value 	( − 1 )^sign × 2^{exponent − 127} × 1. fraction
FFH = 111111112 	±infinity 	NaN (quiet, signalling) 	

*/
/*
IEEE 754 half-precision binary floating-point format: binary16
Exponent 	Significand = zero 	Significand ≠ zero 	Equation
0b00000 	zero, −0 	subnormal numbers 	(−1)^signbit × 2^{−14} × 0.significantbits
0b00001, ..., 111102 	normalized value 	(−1)^signbit × 2^{exponent−15} × 1.significantbits
0b11111 	±infinity 	NaN (quiet, signalling) 	
*/
#include <math.h>
float cbor_decode_half(uint8_t *buf){
	uint16_t half = ntohs(*(uint16_t*)buf);
	uint32_t sign = half >>15;
	uint32_t exp  =(half >>10)&0x1F;//10 бит
	uint32_t mant = half & 0x3ff;//10 бит
	float fval;
	if (exp==0) {
		uint32_t d = (sign<<31|mant);
		fval = *(float*)&d;
	} else if (exp!=31) {
		uint32_t d = (sign<<31|(exp+(127-15))<<23|mant<<13);
		fval = *(float*)&d;
	} else
		fval = mant == 0 ? INFINITY : NAN;
	return fval;
}
float cbor_decode_float(uint8_t *buf){
	uint32_t data = ntohl(*(uint32_t*)buf);
	return *(float*)&data;
}
double cbor_decode_double(uint8_t *buf){
	uint64_t data = ntohll(*(uint64_t*)buf);
	return *(double*)&data;
}

#define FAIL ((void*)0)
uint8_t* cbor_well_formed_indefinite(uint8_t* buf, uint8_t mt);
// конструктивные типы кодируются в приложении
// Тестирование формата
#include <stdio.h>
uint8_t* cbor_well_formed(uint8_t* buf) {
  // process initial bytes
	uint8_t ib = *buf++;
	uint8_t mt = ib >> 5;
	uint64_t val;// = ib & 0x1f;
	switch (ib & 0x1f) {
	default: val = ib & 0x1f; break;
	case 24: val = *(uint8_t *)buf; buf+=1; break;
	case 25: val = ntohs(*(uint16_t*)buf); buf+=2; break;
	case 26: val = ntohl(*(uint32_t*)buf); buf+=4; break;
	case 27: val = ntohll(*(uint64_t*)buf); buf+=8; break;
	case 28: case 29: case 30: return FAIL;
	case 31:
		buf = cbor_well_formed_indefinite(buf, mt);
		if (buf!=FAIL && buf[0]==0xFF) return buf+1;
		return FAIL;
	}
  // process content
	switch (mt) {
	// case 0, 1, 7 do not have content; just use val
	case 2:
	case 3: buf += val; break; // bytes/UTF-8
	case 4:
		for (unsigned int i = 0; i < val; i++) {
			buf = cbor_well_formed(buf);
			if (buf == FAIL) break;
		}
		break;
	case 5:
		for (unsigned int i = 0; i < val*2; i++) {
			buf = cbor_well_formed(buf);
			if (buf == FAIL) break;
		}
		break;
	case 6:
		buf = cbor_well_formed(buf);
		break;     // 1 embedded data item
	case 7:
		if ((ib & 0x1f) == 24 && val < 32) buf = FAIL; // bad simple
		break;
	}
  return buf;                    // definite-length data item
}
// Тестирование данных, допускается упаковка
uint8_t* cbor_well_formed_indefinite(uint8_t* buf, uint8_t mt)
{
	switch (mt) {
    case 2: case 3: // byte/text string
		while (buf[0] != 0xFF) {// need definite-length chunk of same type
			if ((buf[0]>>5)!=mt) return FAIL;
			buf = cbor_well_formed(buf);
			if (buf == FAIL) return FAIL;
		}
		break;
    case 4: // array
		while (buf[0] != 0xFF){
			buf = cbor_well_formed(buf);
			if (buf == FAIL) return FAIL;
		}
		break;
    case 5: // map
		while (buf[0] != 0xFF) {
			buf = cbor_well_formed(buf);
			if (buf == FAIL) { return FAIL;}
			buf = cbor_well_formed(buf);
			if (buf == FAIL) { return FAIL;}
		}
		break;
    default:
		printf("wr mt\n");
		return FAIL;// wrong mt
	}
	return buf;                    // indefinite-length data item
}
#if 0
#include <glib.h>


static const uint8_t base64url_table[64] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789-_";
/*! \brief Кодирование Base64-url safe (base64url)
    \param dst - указатель на выходной буфер, длина буфера должна быть больше (length(src)*4 + 3)/3
    \param src - указатель на входной буфер. Строка должна заканчиваться нулем
    \return указатель на конец записи в выходном буфере.
 */
void cbor_base64url_enc(GString *str, uint8_t *src, size_t length)
{
    uint32_t acc=0;
    int bits=0;
    while (length--)
    {
        bits+=8;
        acc |= (uint32_t)(*src++) << (32-bits);
        do {
			g_string_append_c(str, base64url_table[(acc>>(32-6))&0x3F]);
            acc <<= 6, bits-=6;
        } while (bits >= 6);
    }
    if (bits) {
        g_string_append_c(str, base64url_table[(acc>>(32-6))&0x3F]);
    }
}
// $ gcc -DTEST_CBOR -o cbor cbor.c  `pkg-config --libs --cflags glib-2.0`
/*! \brief Преобразование бинарного формата CBOR в формат JSON
    \return Возвращает ошибку если не разобран
*/
uint8_t* cbor_to_json(uint8_t * buf, size_t len, GString* str)
{
	uint64_t value = 0;
	uint64_t context_tag = 0;
	size_t length;
	int i;
	//while (len<0 || --len>=0)
	{
	uint8_t tag = buf[0];
	if ((tag&0xE0)==0xC0){// контектсный тег
		buf = cbor_decode_tag(buf, &tag, &context_tag);
		tag = buf[0];
	}
	if ((tag&0xE0)==0xE0){
		buf++;
		switch(tag&0x1F){
		case 0 ... 0x13:// 0-23
			g_string_append_printf(str, "%u",(unsigned) (tag&0x1F));
			break;
		case 0x14:
			g_string_append_len(str, "false",5);
			break;
		case 0x15:
			g_string_append_len(str, "true",4);
			break;
		case 0x16:
			g_string_append_len(str, "null",4);
			break;
		case 0x17:
			g_string_append_len(str, "undefined",9);
			break;
		case 0x18:// simple byte
			g_string_append_printf(str, "s:%u",(unsigned) buf[0]);
			buf+=1;
			break;
		case 0x19:{//   half-precision
			float fval = cbor_decode_half(buf); buf+=2;
			g_string_append_printf(str, "%g", fval);
		}	break;
		case 0x1A:{// single-precision float (four-byte IEEE 754)
			float fval = cbor_decode_float(buf); buf+=4;
			g_string_append_printf(str, "%g", fval);
		} break;
		case 0x1B:{// double-precision float (eight-byte IEEE 754)
			double fval = cbor_decode_double(buf); buf+=8;
			g_string_append_printf(str, "%g", fval);
		} break;
		case 0x1C ... 0x1E: // reserved
		case 0x1F: // break - stop condition
		default: printf("undefined");
			break;
		}
	} else {
	buf = cbor_decode_tag(buf, &tag, &value);
	switch (tag>>5){
	case 0:// unsigned int
		g_string_append_printf(str, "%lu", value);
		break;
	case 1:// negative int
		g_string_append_printf(str, "-%lu", value+1);// -1-N
		break;
	case 2:// byte string
        g_string_append_c(str, '"');
		if (cbor_indefined(tag)){// состоит из обрезков ("chunks")
			do{
                tag = buf[0];
				if (tag == CBOR_BREAK) break;
				buf = cbor_decode_len(buf, &length);
                if (context_tag==0){//context_tag==CBOR_TAG_BASE64URL) {				// применить кодирование base64
					cbor_base64url_enc(str, buf, length);
                } else {// кодирование в форме BASE16
                    for (int i=0; i<length; i++){
                        g_string_append_printf(str, "%02X", buf[i]);
                    }
                }
				buf+=length;
			} while(1);//buf[0]!=0xFF);
		} else {
			g_string_append_len(str, (char*)buf, value);
			buf+=value;
		}
        g_string_append_c(str, '"');
		break;
	case 3:// utf8 string
		if (cbor_indefined(tag)){// состоит из обрезков строк ("chunks")
			do {
                tag = buf[0];
				if (tag == CBOR_BREAK) { buf++; break; }
				buf = cbor_decode_len(buf, &length);
				g_string_append_c(str, '"');
				g_string_append_len(str, (char*)buf, length);
				g_string_append_c(str, '"');
				buf+=length;
			} while(1);
		} else {// TODO заменить обратный слеш и кавычку \ " 
			g_string_append_c(str, '"');
			g_string_append_len(str, (char*)buf, value);
			buf+=value;
            g_string_append_c(str, '"');
		}
		break;
	case 4:// array
		g_string_append_c(str, '[');
		length = value;
		for (i=0; i<length; i++){
            if (buf[0]==CBOR_BREAK) { buf++; break; }
            if (i!=0) g_string_append_c(str, ',');
			buf = cbor_to_json(buf, len, str);// длина до конца буфера
		}
        g_string_append_c(str,']');
		break;
	case 5: // map
		g_string_append_c(str, '{');
		length = value;
		for (i=0; i<length; i++){
			if (buf[0]==CBOR_BREAK) { buf++; break; }
            if (i!=0) g_string_append_c(str, ',');
			buf = cbor_to_json(buf, len, str);// длина до конца буфера
			g_string_append_c(str, ':');
			buf = cbor_to_json(buf, len, str);// длина до конца буфера
		}
		g_string_append_c(str,'}');
		break;
	case 6:// tag - дополнительный идентификатор формата, см. контексно-зависимый разбор строк
		//context_tag = value;
		break;
	case 7:// simple
		break;
	}
	}
	}
	return buf;
}

/*!	\brief Преобразование внутреннего формата JSON в CBOR */
uint8_t* cbor_encode_list(uint8_t * buf, GSList* param_list)
{
	// 
	return buf;
}
/* ODT - таблица описания объекта в памяти CAN */
typedef struct _odt odt_t;
struct _odt {
	unsigned  offset:16;// 0..8192
	unsigned  type:4;	// тип данных ASN
	unsigned  is_array:1; // поле является мультиплексором
	unsigned  pos:3;	// позиция по адресу в битах 0..7
	unsigned  len:6;	// длина поля в битах -1 0..63
	unsigned  align:2; // 0=1,1=2,2=4,3=8
	uint8_t* ref;	// адрес без выравнивания
};
#define ASN_TYPE_NULL	         0x0
#define ASN_TYPE_BOOLEAN         0x1
#define ASN_TYPE_UNSIGNED        0x2
#define ASN_TYPE_INTEGER         0x3
#define ASN_TYPE_REAL	         0x4
#define ASN_TYPE_DOUBLE	         0x5
#define ASN_TYPE_OCTETS    		 0x6
#define ASN_TYPE_STRING    		 0x7
#define ASN_TYPE_BIT_STRING		 0x8
#define ASN_TYPE_ENUMERATED		 0x9
#define ASN_TYPE_DATE			 0xA
#define ASN_TYPE_TIME			 0xB
#define ASN_TYPE_OID             0xC

#define ASN_TYPE_CHOICE 0xD// псевдо тип не используется в протоколе
#define ASN_TYPE_ANY    0xE// псевдо тип не используется в протоколе


#define ALIGN(x) (((x)>>4)&3)
/*! представление данных в CBOR по таблице */
uint8_t * cbor_odt(uint8_t* buf, uint8_t* addr, GSList * list)
{
	while(list){
		odt_t * odt = list->data;
		switch (odt->type&0x7){
		case ASN_TYPE_UNSIGNED:{
			uint64_t value = *(uint64_t*)addr;
			value = (value >> odt->pos) & (~0ULL>>(63 - odt->len));
			buf = cbor_encode_uint(buf, value);
		}	break;
		case ASN_TYPE_INTEGER: {
			int64_t value = *(int64_t*)addr;
			value = (value >> odt->pos) & (~0ULL>>(63 - odt->len));
			buf = cbor_encode_sint(buf, value);
		}	break;
		case ASN_TYPE_OCTETS:
			buf = cbor_encode_data(buf, (uint8_t*)addr, odt->len>>3);
			break;
		case ASN_TYPE_STRING:
			buf = cbor_encode_string(buf, (char*)addr, odt->len>>3);
			break;
		case ASN_TYPE_ENUMERATED:{
			uint32_t value = *(uint32_t*)addr;// без выравнивания
			value = (value >> odt->pos) & (~0UL>>(31 - odt->len));
			buf = cbor_encode_simple(buf, value);
		}	break;
		case ASN_TYPE_BOOLEAN:{
			uint32_t value = *(uint32_t*)addr;// без выравнивания
			value = (value >> odt->pos) & (~0UL>>(31 - odt->len));
			buf = cbor_encode_bool(buf, value);
		}	break;
		case ASN_TYPE_REAL:{
			float fval = *(float*)addr;
			buf = cbor_encode_float(buf, fval);
		} 	break;
		case ASN_TYPE_DOUBLE:{
			double fval = *(double*)addr;
			buf = cbor_encode_double(buf, fval);
		}	break;
		default:
			break;
		// векторный тип?
		}
		list = list->next;
	}
	return buf;
}

//#define OID_MASK 0x3FFFFFuL
static uint8_t* cbor_encode_oid	  (uint8_t* pdu, uint32_t oid){// пакуется в 3-5 байт
	pdu = cbor_encode_sint (pdu, (int32_t)(oid));
	return pdu;
}
/*! \brief представление данных в CBOR по таблице */
uint8_t * cbor_encode_param(uint8_t* buf, Object_t* object, uint16_t param_id, uint32_t index)
{
	const ObjectProperty_t *prop = &object->property_list[param_id];
	uint8_t* data = object->data + (prop->offset);

	if (IS_ARRAY(prop)){
		int align = ALIGN(prop->type);
		if ((prop->type&0x7)==_TYPE_REAL) align++;
		data += (1u<<align)*index;
	}
		switch (prop->type&0x7){
		case _TYPE_NULL:{// это псевдо-тип
			switch (ALIGN(prop->type)){
			case 0: *buf++ = CBOR_NULL; break;
			case 1: {// list
				GSList* list = *(GSList**)data;
				list = g_slist_nth(list, index);
				if (list==NULL) *buf++ = CBOR_UNDEF;//
				else {
					data = list->data;// элемент списка, про него нужно знать класс
				}
				// применить кодирование вложенного типа
			}	break;
			case 2:	buf = cbor_encode_oid(buf, *(uint32_t*)data); break;
			}
		}	break;
		case _TYPE_BOOL:{
			uint64_t value;
			switch (ALIGN(prop->type)){
			case 0: {//bool
				uint8_t value = *(uint8_t*)data;// без выравнивания
				*buf++ = (value&(1u<<prop->mask.pos))? CBOR_TRUE: CBOR_FALSE;
			} break;
			case 1: {// enum
				
			}
			}
		}	break;
		case _TYPE_UINT:{
			uint64_t value;
			switch (ALIGN(prop->type)){
			case 0: value = *(uint8_t *)data; break;
			case 1: value = *(uint16_t*)data; break;
			case 2: value = *(uint32_t*)data; break;
			case 3: value = *(uint64_t*)data; break;
			}
			if (PACKED(prop->type)){// если тип определен, как битовое поле
				unsigned len = 64-prop->mask.len;// длина в битах
				value = value<<(len-prop->mask.pos)>>len;
			}
			buf = cbor_encode_uint(buf, value);
		}	break;
		case _TYPE_SINT:{
			int64_t value;
			switch (ALIGN(prop->type)){
			case 0: value = *(int8_t *)data; break;
			case 1: value = *(int16_t*)data; break;
			case 2: value = *(int32_t*)data; break;
			case 3: value = *(int64_t*)data; break;
			}
			if (PACKED(prop->type)){// если тип определен, как битовое поле
				unsigned len = 64-prop->mask.len;// длина в битах
				value = value<<(len-prop->mask.pos)>>len;
			}
			buf = cbor_encode_sint(buf, value);
		}	break;
		case _TYPE_REAL:{
			double value;
			switch (ALIGN(prop->type)){// форматы _FloatN - предназначены для обмена данными
			case 0: buf = cbor_encode_half  (buf, *(_Float16*)data);break;// внутреннее представление может быть half, float или _Float16 или _BFloat16
			case 1: buf = cbor_encode_float (buf, *(float*)data); 	break;
			case 2: buf = cbor_encode_double(buf, *(double*)data);	break;
			case 3: {// если платформа поддерживает тип binary128 $ echo -n ""| gcc -dM -E - | grep __FLT_
/*
$ echo -n ""| gcc -dM -E - | grep _MANT
#define __FLT128_MANT_DIG__ 113
#define __FLT64X_MANT_DIG__ 64
#define __LDBL_MANT_DIG__ 	64
#define __FLT64_MANT_DIG__ 	53
#define __DBL_MANT_DIG__ 	53
#define __FLT32X_MANT_DIG__ 53
#define __FLT_MANT_DIG__ 	24
#define __FLT32_MANT_DIG__ 	24
#define __FLT16_MANT_DIG__ 	11
#define __BFLT16_MANT_DIG__ 8

#define __DEC32_MANT_DIG__ 	7
#define __DEC64_MANT_DIG__ 	16
#define __DEC128_MANT_DIG__ 34

Float16 					binary16
float, _Float32 			binary32
double, _Float64, _Float32x binary64
long double, _Float64x 		80-bit 	binary64-extended
_Float128 					binary128

*/
#if 0
				*buf++ = 0xfc;
				*(_Float128*)buf = *(long double*)data;// конвертация binary64x в binary128 ? не ясно
				buf+=sizeof(_Float128);
#endif
			}	break;
			}

		} 	break;
		case _TYPE_BSTR:{
			
		}	break;
		case _TYPE_TSTR:
			if (ALIGN(prop->type)){
//				*buf++ = CBOR_TAG_TXT_B64U;
//				*buf++ = CBOR_TAG_TXT_B64;
			} else
				buf = cbor_encode_string(buf, (char*)data, prop->size?prop->size:256);// 
			break;
		default:
			break;
		// векторный тип?
		}
	return buf;
}
#endif
#ifdef TEST_CBOR
#include <stdio.h>
// Тесты на упаковку - распаковку
int main(){
	GString *str = g_string_new(NULL);
if(1){
	printf("3.2.2. Indefinite-Length Arrays and Maps\n");
	// abstract array [1, [2, 3], [4, 5]]
	uint8_t tst1[] = "\x83\x01\x82\x02\x03\x82\x04\x05";
	uint8_t tst2[] = "\x9f\x01\x82\x02\x03\x9f\x04\x05\xff\xff";
	uint8_t tst3[] = "\x9f\x01\x82\x02\x03\x82\x04\x05\xff";
	uint8_t tst4[] = "\xbf\x63\x46\x75\x6e\xf5\x63\x41\x6d\x74\x21\xff";
	uint8_t tst5_1[] = "\xfb\x3f\xf0\x00\x00\x00\x00\x00\x00"; 
	uint8_t tst5[] = "\xfa\x3f\x80\x00\x00"; 
	uint8_t tst6[] = "\xf9\x3c\x00";
	uint8_t tst7[] = "\xfa\x45\xad\x9c\x00";
	uint8_t tst8[] = "\xf9\x45\x80";
	uint8_t tst9[] = "\xf9\x7e\x00";
	uint8_t tstA[] = "\x81\x18\x64";
	cbor_to_json(tst1, -1, str); 
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tst1)!=FAIL) printf("..ok\n");
	str->len = 0;
	cbor_to_json(tst2, 1, str);
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tst2)!=FAIL) printf("..ok\n");
	str->len = 0;
	cbor_to_json(tst3, -1, str);
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tst3)!=FAIL) printf("..ok\n");
	str->len = 0;
	cbor_to_json(tst4, -1, str);
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tst4)!=FAIL) printf("..ok\n");
	str->len = 0;
	cbor_to_json(tst5_1, -1, str);
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tst5_1)!=FAIL) printf("..ok\n");
	str->len = 0;
	cbor_to_json(tst5, -1, str);
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tst5)!=FAIL) printf("..ok\n");
	str->len = 0;
	cbor_to_json(tst6, -1, str);
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tst6)!=FAIL) printf("..ok\n");
	str->len = 0;
	cbor_to_json(tst7, -1, str);
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tst7)!=FAIL) printf("..ok\n");
	str->len = 0;
	cbor_to_json(tst8, -1, str);
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tst8)!=FAIL) printf("..ok\n");
	str->len = 0;
	cbor_to_json(tst9, -1, str);
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tst9)!=FAIL) printf("..ok\n");
	str->len = 0;
	cbor_to_json(tstA, -1, str);
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tstA)!=FAIL) printf("..ok\n");
}

	printf(" F.1. Examples of CBOR Data Items That Are Not Well-Formed \n");
	uint8_t tstB[] = "\xf8\x00\xf8\x01\xf8\x18\xf8\x1f";
	uint8_t tstC[] = "\x9f\x82\x9f\x81\x9f\x9f\xff\xff\xff\xff";
	str->len = 0;
	cbor_to_json(tstB, sizeof(tstB)-1, str);
	printf("%*s\n",(int)str->len,str->str);
	if (cbor_well_formed(tstB)!=FAIL) printf("..ok\n");
	str->len = 0;
	cbor_to_json(tstC, sizeof(tstC)-1, str);
	printf("%*s\n",(int)str->len,str->str);

	return 0;
}
#endif
