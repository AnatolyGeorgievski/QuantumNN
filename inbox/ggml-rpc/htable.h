#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

/* копировать имя, индекс отдельно */
static int  _cname(char* cname, const char* name, char ** end){
    int index = 0;
    const char* s = name;
    while (*s!='\0' && !isdigit(*s)) *cname++ = *s++;
    if (isdigit(*s)) *cname++ = '*'; 
    while (isdigit(*s)) index = index*10+(*s++ -'0');
    while (*s!='\0') *cname++ = *s++; // до конца строки
    *cname++ = '\0';
    if (end) *end = cname;
    return index;
}

#define STN_UNDEF 0
#define Nbucket 256
/*! \brief Структура хеш таблицы */
typedef struct _HTable HTable_t;
struct _HTable {
	uint32_t nbucket;  // число признаков, ограничимся 256 например
	uint32_t nchain;   // длинный список например 1024-256
    char*    dynstr;
    uint32_t dynstr_allocated; // размер выделенной памяти под строки
	uint32_t bucket[Nbucket];// два массива подряд =nbucket+nchain
};
static
void _htable_init(HTable_t *htable, uint32_t nbucket) 
{
	htable->nbucket = nbucket;
	htable->nchain = 0;
	uint32_t i;
	for (i=0; i<nbucket; i++){
		htable->bucket[i]=STN_UNDEF;
	}
    htable->dynstr = (char*)malloc(1024);// строки
    htable->dynstr_allocated = 1024;
}
/*! Таблица символов .symtab */
typedef struct _Elf32_Sym  Quark_t;
struct _Elf32_Sym {
	uint16_t st_name;// сменщение в буфере
};

/*! \brief Расчет ключа для Хеш таблицы символов динамической линковки 
    Функция определена в http://www.sco.com/developers/devspecs/gabi41.pdf
 */
static 
uint32_t elf_hash(const char *name)
{
  uint32_t h = 0;
  while (*name != '\0'){
    h = (h << 4) + *name++;
    uint32_t g = (h & 0xF0000000UL);
    if (g) {
      h ^=  g >> 24;
    }
    h &= ~g;
  }
  return h;
}

/*! \brief найти кварк в таблице строк
    \param dynsym - таблица символов
    \param dynstr - буфер строк

    \todo сделать callback под функцию сравнения и функцию хеша.
 */
static
uint32_t _htable_lookup(HTable_t *htable, const char* name, Quark_t* dynsym)
{
	uint32_t key = elf_hash(name);
	uint32_t y = htable->bucket[key % (Nbucket)];
    const uint32_t *chain = htable->bucket + htable->nbucket;
	while (y<htable->nchain && y!=STN_UNDEF) {
		if (strcmp(htable->dynstr + dynsym[y].st_name, name)==0) 
			return y;//dynsym[y].st_value;
		y = chain[y];
	}
	return STN_UNDEF;
}
/*! \brief Заполнение хеш таблицы

	Таблица хранится в форме массива. Переполнение массива не анализируется. Информация хранится в трех сегментах: символы, имена и хещ таблица. 
    Хеш таблица разделена на две части - ссылки head->next->...->next->STN_UNDEF Таблица инициализируется значением STN_UNDEF.
	
	\return индекс в таблице dynsym

    операцию можно сделать атомарно, атомарно изменить счетчик объектов, атомарно работать с цепочкой. 
    \todo увеличить цепочку и таблицу символов
*/
static
uint32_t _htable_insert(HTable_t *htable, const char *cname)
{
	uint32_t key = elf_hash(cname);
	uint32_t *chain = htable->bucket + htable->nbucket;
//    if () -- увеличение таблицы
	uint32_t y = htable->nchain++;// atomic_fetch_add(htable->nchain, 1);// увеличиваем число объектов (семафор)
    //if (htable->nchain > htable->nchain_length) { }
	// добавить запись в таблице символов, в конец таблицы 
	uint32_t* head = &htable->bucket[key % htable->nbucket];
	chain[y] = *head;
	*head = y;
	return y;
}
#ifdef __cplusplus
}
#endif
