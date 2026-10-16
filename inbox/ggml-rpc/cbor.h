#ifndef CBOR_H
#define CBOR_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>
/* Кодирование структурированных данных интернет стандарт STD94
	[CBOR, RFC8949] Concise Binary Object Representation (CBOR), Dec 2020
\see https://www.rfc-editor.org/rfc/rfc8949.html

Простые правила кодирования структурированных данных системного протокола
Вариант1. Дописать код Array_of_length(N) Code = 0x80|N перед блоком данных,
	где N- число элементов, не более 23
Варинат2. Дописать код Array_indefined() Code = (0x80|0x1F) перед блоком данных,
	и код BRAKE(0xFF) в конце блока
Так же применяются правила кодирования объектов Map Code 0xA0|N
Для последующего представления в JSON структура MAP должна содержать пары "имя":значение
Для системного протокола имена полей могут быть представлены:
строкой, числом, тэгом или простым типом(контекстно зависимое кодирование)

Major Types
Type 	Meaning 		Content
0 		unsigned int 	N -
1 		negative int	-1-N 	-
2 		byte string 	N bytes
3 		text string 	N bytes (UTF-8 text)
4 		array 			N data items (elements)
5 		map 			2N data items (key/value pairs)
6 		tag of number N 1 data item
7 		simple/float 	-

Конвертация в Json
0xF4 0xE0|20 false
0xF5 0xE0|21 true
0xF6 0xE0|22 null

0xF7 0xE0|23 undefined

0xF9 0xE0|25 Half
0xFA 0xE0|26 Float Single Precision
0xFB 0xE0|27 Double

0xFF 0xE0|31 Break

bytestring base64url
Кодирование строк неопределенной длины
4F {строки определенной длины} FF
5F {вложения определенной длины} FF


*/
#define CBOR_FALSE 0xF4
#define CBOR_TRUE  0xF5
#define CBOR_NULL  0xF6
#define CBOR_UNDEF 0xF7
// F4+5
#define CBOR_HALF  	0xF9
#define CBOR_FLOAT 	0xFA
#define CBOR_DOUBLE 0xFB
//#define CBOR_FLT128 0xFC
// FC FD FE
#define CBOR_BREAK 0xFF

enum {
	CBOR_TYPE_INT,
	CBOR_TYPE_NEG,
	CBOR_TYPE_BYTES,
	CBOR_TYPE_STRING,
	CBOR_TYPE_ARRAY,
	CBOR_TYPE_MAP,
	CBOR_TYPE_TAG,
	CBOR_TYPE_SIMPLE,
};
// кодирование данных одним байтом : 3 старшие бита - основной тип, len - число элементов 0..23, если элементов больше разметка 1F и 0xFF 
#define CBOR_ENCODE(type, len) ((CBOR_TYPE_##type<<5)|(len))
#define CBOR_TYPE(tag) ((tag)>>5)
enum {//https://www.iana.org/assignments/cbor-tags/cbor-tags.xhtml
// https://datatracker.ietf.org/doc/draft-bormann-cbor-notable-tags/11/
CBOR_TAG_ISO_TIME	=CBOR_ENCODE(TAG, 0),//0	text string	Standard date/time string; see Section 3.4.1
CBOR_TAG_DATE_TIME	=CBOR_ENCODE(TAG, 1),//1	integer or float	Epoch-based date/time; see Section 3.4.2

CBOR_TAG_BIGNUM		=CBOR_ENCODE(TAG, 2),//2	byte string	Unsigned bignum; see Section 3.4.3
CBOR_TAG_BIGNUM_NEG	=CBOR_ENCODE(TAG, 3),//3	byte string	Negative bignum; see Section 3.4.3

CBOR_TAG_DECIMAL	=CBOR_ENCODE(TAG, 4),//4	array	Decimal fraction; see Section 3.4.4
CBOR_TAG_BIGFLOAT	=CBOR_ENCODE(TAG, 5),//5	array	Bigfloat; see Section 3.4.4

CBOR_TAG_BASE64URL	=CBOR_ENCODE(TAG,21),//21	(any)	Expected conversion to base64url encoding; see Section 3.4.5.2
CBOR_TAG_BASE64		=CBOR_ENCODE(TAG,22),//22	(any)	Expected conversion to base64 encoding; see Section 3.4.5.2
CBOR_TAG_BASE16		=CBOR_ENCODE(TAG,23),//23	(any)	Expected conversion to base16 encoding; see Section 3.4.5.2

CBOR_TAG_CBOR		=CBOR_ENCODE(TAG,24),//24	byte string	Encoded CBOR data item; see Section 3.4.5.1
CBOR_TAG_RATIONAL	=CBOR_ENCODE(TAG,30),/*30	Rational number     
d8 1e      ---- Tag 30
      82      ---- Array length 2
         01   ---- 1
         03   ---- 3 */
CBOR_TAG_TXT_URI	=CBOR_ENCODE(TAG,32),//32	text string	URI; see Section 3.4.5.3
CBOR_TAG_TXT_B64U	=CBOR_ENCODE(TAG,33),//33	text string	base64url; see Section 3.4.5.3
CBOR_TAG_TXT_B64	=CBOR_ENCODE(TAG,34),//34	text string	base64; see Section 3.4.5.3
CBOR_TAG_TXT_REGEXP	=CBOR_ENCODE(TAG,35),// [RFC7049]
CBOR_TAG_TXT_MIME	=CBOR_ENCODE(TAG,36),//36	text string	MIME message; see Section 3.4.5.3
CBOR_TAG_UUID		=CBOR_ENCODE(TAG,37),//37	byte string	Binary UUID ([RFC9562, Section 4])	[https://github.com/lucas-clemente/cbor-specs/blob/master/uuid.md][Lucas_Clemente]	
//38	array	Language-tagged string	[RFC9290, Appendix A]	
//39	multiple	Identifier	[https://github.com/lucas-clemente/cbor-specs/blob/master/id.md][Lucas_Clemente]	
//40	array of two arrays [1]	Multi-dimensional Array, row-major order	[RFC8746]	
//41	array	Homogeneous Array	[RFC8746]	
//42	byte string	IPLD content identifier	[https://github.com/ipld/cid-cbor/][Volker_Mische]	

//43	text string	YANG bits datatype; see Section 6.7.	[RFC9254]	
//44	text string	YANG enumeration datatype; see Section 6.6.	[RFC9254]	
//45	unsigned integer or text string	YANG identityref datatype; see Section 6.10.	[RFC9254]	
//46	unsigned integer or text string or array	YANG instance-identifier datatype; see Section 6.13.	[RFC9254]	
//47	unsigned integer	YANG Schema Item iDentifier (sid); see Section 3.2.	[RFC9254]	
//48	byte string	IEEE MAC Address	[RFC9542]	

//52	byte string or array	IPv4, [prefixlen,IPv4], [IPv4,prefixpart]	[RFC9164]	
//53	Unassigned			
//54	byte string or array	IPv6, [prefixlen,IPv6], [IPv6,prefixpart]	[RFC9164]	

//61	CBOR Web Token (CWT)	CBOR Web Token (CWT)	[RFC8392][Michael_B._Jones]	

//63	byte string	Encoded CBOR Sequence [RFC8742]	[draft-bormann-cbor-notable-tags-02, Section 2.1]	

//64	byte string	uint8 Typed Array	[RFC8746]	
//65	byte string	uint16, big endian, Typed Array	[RFC8746]	
//66	byte string	uint32, big endian, Typed Array	[RFC8746]	
//67	byte string	uint64, big endian, Typed Array	[RFC8746]	
//68	byte string	uint8 Typed Array, clamped arithmetic	[RFC8746]	
//69	byte string	uint16, little endian, Typed Array	[RFC8746]	
//70	byte string	uint32, little endian, Typed Array	[RFC8746]	
//71	byte string	uint64, little endian, Typed Array	[RFC8746]	
//72	byte string	sint8 Typed Array	[RFC8746]	
//73	byte string	sint16, big endian, Typed Array	[RFC8746]	
//74	byte string	sint32, big endian, Typed Array	[RFC8746]	
//75	byte string	sint64, big endian, Typed Array	[RFC8746]	
//76	byte string	(reserved)	[RFC8746]	
//77	byte string	sint16, little endian, Typed Array	[RFC8746]	
//78	byte string	sint32, little endian, Typed Array	[RFC8746]	
//79	byte string	sint64, little endian, Typed Array	[RFC8746]	
//80	byte string	IEEE 754 binary16, big endian, Typed Array	[RFC8746]	
//81	byte string	IEEE 754 binary32, big endian, Typed Array	[RFC8746]	
//82	byte string	IEEE 754 binary64, big endian, Typed Array	[RFC8746]	
//83	byte string	IEEE 754 binary128, big endian, Typed Array	[RFC8746]	
//84	byte string	IEEE 754 binary16, little endian, Typed Array	[RFC8746]	
//85	byte string	IEEE 754 binary32, little endian, Typed Array	[RFC8746]	
//86	byte string	IEEE 754 binary64, little endian, Typed Array	[RFC8746]	
//87	byte string	IEEE 754 binary128, little endian, Typed Array	[RFC8746]	

//96	COSE_Encrypt	COSE Encrypted Data Object	[RFC9052]	
//97	COSE_Mac	COSE MACed Data Object	[RFC9052]	
//98	COSE_Sign	COSE Signed Data Object	[RFC9052]	

//100	Unsigned or negative integer	Number of days since the epoch date 1970-01-01	[RFC8943]
// | 264        | array       | Decimal fraction with arbitrary exponent | bigfrac.html 
// | 265        | array       | Bigfloat with arbitrary exponent         | bigfrac.html
// | 268        | array       | Extended decimal fraction     	| extended.html |
// | 269        | array       | Extended bigfloat     			| extended.html |
// | 270        | array       | Extended rational number    	| extended.html |
};

typedef struct _GSList GSList;
uint8_t* cbor_encode_tag (uint8_t* buf, uint8_t  tag, uint64_t  value);
uint8_t* cbor_encode_list(uint8_t* buf, GSList* list);
uint8_t* cbor_decode_tag (uint8_t* buf, uint8_t *tag, uint64_t *value);
/* для кодирования массивов и строк внутри пакета размеры менее 255 байт */
/* для строк большего размера строки должны разбиваться на фрагменты
 */
static inline uint8_t* cbor_encode_len(uint8_t* pdu, unsigned len, uint8_t tag){
	if (len<24)
		*pdu++= (tag<<5) | len;
	else {
		*pdu++= (tag<<5) | 24;
		*pdu++= len;
	}
	return pdu;
}
static inline uint8_t* cbor_encode_uint(uint8_t* buf, uint64_t val){// 0x00 - unsigned int
	return cbor_encode_tag(buf, 0x00, val);
}
static inline uint8_t* cbor_encode_sint(uint8_t* buf, int64_t val){// 0x00, 0x20 - signed int
	uint8_t tag;
	if (val < 0) {
		tag = 1;// signed
		val = ~val;// операция -1-N
	} else
		tag = 0;// unsigned
	return cbor_encode_tag(buf, tag, val);
}
static inline uint8_t* cbor_encode_data(uint8_t* buf, uint8_t* data, size_t len){// 0x40 - бинарная строка
	buf = cbor_encode_len(buf, len, CBOR_TYPE_BYTES);
	__builtin_memcpy(buf, data, len);
	return buf+len;
}
static inline uint8_t* cbor_encode_string (uint8_t* buf, char* data, size_t len){// 0x60 -- текстовая строка в UTF-8
	buf = cbor_encode_len(buf, len, CBOR_TYPE_STRING);
	__builtin_memcpy(buf, data, len);
	return buf+len;
}
static inline uint8_t* cbor_encode_bool(uint8_t* buf, bool val){ // 0xf4 - false, 0xf5 - true
	*buf = 0xf4|(val?1:0);
	return buf+1;
}
static inline uint8_t* cbor_encode_null(uint8_t* buf){// 0xf6 - null
	*buf = 0xf6;
	return buf+1;
}
static inline uint8_t* cbor_encode_simple(uint8_t* buf, uint8_t val){// 0xe0-0xf3, 0xf8
	if (val<=0x13)
		*buf++ = 0xe0|val;
	else {
		*buf++ = 0xf8;
		*buf++ = val;
	}
	return buf;
}
static inline uint8_t* cbor_encode_half (uint8_t* buf, _Float16 val){// 0xf9 	half-precision float (two-byte IEEE 754)
	*buf++ = 0xf9;
	*(_Float16*)buf = val; // копирование без выравнивания данных в памяти
	return buf + sizeof(_Float16);
}

static inline uint8_t* cbor_encode_float (uint8_t* buf, float val){// 0xfa 	single-precision float (four-byte IEEE 754)
	*buf++ = 0xfa;
	*(_Float32*)buf = val; // копирование без выравнивания данных в памяти
	return buf + sizeof(_Float32);
}
static inline uint8_t* cbor_encode_double(uint8_t* buf, double val){// 0xfb 	double-precision float (eight-byte IEEE 754)
	*buf++ = 0xfb;
	*(_Float64*)buf = val; // копирование без выравнивания данных
	return buf + sizeof(_Float64);
}
// Функции разбора пакетов
/*! \brief Проверка, является ли запись неопределенной */
static inline bool cbor_indefined(uint8_t tag){
	return (tag&31)==31;
}


static inline int cbor_skip(uint8_t tag){
	const uint8_t len = tag&0x1F;
	return (len>23)? 1<<(len-23): 0;
}

float 	cbor_decode_half(uint8_t *buf);
float 	cbor_decode_float(uint8_t *buf);
double 	cbor_decode_double(uint8_t *buf);


uint8_t* cbor_well_formed(uint8_t* buf);
uint8_t* cbor_well_formed_indefinite(uint8_t* buf, uint8_t mt);
#ifdef __cplusplus
}
#endif

#endif // _CBOR_H
