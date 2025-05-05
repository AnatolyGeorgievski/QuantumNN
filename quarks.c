#include <stdatomic.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "quarks.h"

#ifndef QUARK_UNDEF
 #define QUARK_UNDEF 0
#endif
#define Nbucket 256
struct _QTable {
	uint32_t n_bucket;  // число признаков, ограничимся 256 например
	uint32_t n_chain;   // длинный список например 1024-256
    uint32_t count ;    // число элементов 
	char* 	 dynstr;	// словарь
	size_t 	 dynstr_size;	// размер словаря в байтах
	uint32_t bucket[Nbucket];// два массива подряд =n_bucket + n_chain
	struct _QChain {
		uint32_t next;
		uint32_t offs;
	} chain[0];
};

static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

/*! Quarks - сопоставление имен и уникальных идентификаторов с использованием хеш таблицы */

/*! \brief Инициализация хеш таблицы
 */
static void _quark_init(QTable_t *htable, uint32_t n_bucket) 
{
	htable->n_bucket = n_bucket;
	uint32_t i;
	for (i=0; i<n_bucket; i++){
		htable->bucket[i]=QUARK_UNDEF;
	}
}
/*! \brief Создать словарь с размером 
    \param n_bucket размер хеш таблицы
    \param n_chain  предполагаемое число слов
 */
QTable_t * _quark_new(uint32_t n_bucket, uint32_t n_chain) 
{
    QTable_t *htable = (QTable_t *)malloc(sizeof(QTable_t) + (n_bucket - Nbucket)*sizeof(uint32_t) + n_chain * sizeof(struct _QChain));
	htable->n_chain = n_chain;
	htable->dynstr_size = 1024;
	htable->dynstr = malloc(1024);
	htable->count = 1;// один элемент - пустая строка
    _quark_init(htable, n_bucket);
    struct _QChain *chain = (struct _QChain *)(htable->bucket + htable->n_bucket);
    chain[0].offs = 0;// пустая строка, длина строки 0
	chain[0].next = QUARK_UNDEF;
    chain[1].offs = 0;
    return htable;
}

/*! \brief Увеличить размер словаря
 */
static 
QTable_t * _quark_resize(QTable_t* htable, uint32_t n_chain) 
{
	htable = (QTable_t *)realloc(htable, sizeof(QTable_t) + (htable->n_bucket - Nbucket)*sizeof(uint32_t) + n_chain * sizeof(struct _QChain));
	return htable;
}
/*! \brief Увеличить размер буфера строк
 */
static 
void _quark_dynstr_resize(QTable_t* htable, uint32_t n_chain) 
{
	size_t  dynstr_size = ((size_t)(((float)htable->dynstr_size*n_chain)/htable->n_chain) + 1023u)&~1023u;
	htable->dynstr = (char *)realloc(htable->dynstr, dynstr_size);
	htable->dynstr_size = dynstr_size;
}
/*! \brief Найти строку в словаре 
	\param cname строка - текстовый идентификатор
	\return индекс строки или QUARK_UNDEF
 */
int _quark_lookup(QTable_t * htable, const char *cname) 
{
	size_t   len = __builtin_strlen(cname);
	uint32_t key = fnv_hash(cname, len);
// printf("lookup cname=%s key=%08x\n", cname, key);
	uint32_t y = htable->bucket[key % (htable->n_bucket)];
    const struct _QChain *chain = (const struct _QChain *)(htable->bucket + htable->n_bucket);
	while (y<htable->count && y!=QUARK_UNDEF) {
		if (strncmp(htable->dynstr + chain[y].offs, cname, len)==0) 
			return y;
		y = chain[y].next;
	}
	return QUARK_UNDEF;
}
/*! \brief Добавить строку в словарь
	\param cname строка - текстовый идентификатор
	\return индекс добавленной строки
 */
int _quark_insert(QTable_t * htable, const char *cname) 
{
	size_t len = __builtin_strlen(cname);
	uint32_t key = fnv_hash(cname, len);
	uint32_t y = atomic_fetch_add(&htable->count,1);
	// if (y==htable->n_chain) htable = _quark_resize(htable, htable->n_chain*2);
    struct _QChain *chain = (struct _QChain *)(htable->bucket + htable->n_bucket);
	chain[y+1].offs = chain[y].offs + (len+1); // atomic_fetch_add(&htable->size, len);
    if (htable->dynstr_size < chain[y+1].offs) {// увеличить размер буфера имен
        // _quark_dynstr_resize(htable, );
    }
	__builtin_memcpy(htable->dynstr + chain[y].offs, cname, len);
	htable->dynstr[chain[y].offs + len] = '\0';

	uint32_t* head = &htable->bucket[key % htable->n_bucket];
	// \todo атомарно do {} while(!CAS);
	chain[y].next = *head;
	*head = y;
	return y;
}
/*! \brief печать таблицы имен в формате csv 

	Можно использовать для преобразования в JSON формат в виде массива имен
	\param htable - указатель на хэш таблицу
 */
void _quark_to_csv(QTable_t * htable){
	struct _QChain *chain = (struct _QChain *)(htable->bucket + htable->n_bucket);
	for (int i=1; i<htable->count; i++){
		int len = chain[i+1].offs - chain[i].offs;
		printf("\"%.*s\"%c", len, htable->dynstr+ chain[i].offs, (i==htable->count-1)?'\n':',');
	}
}