#include "qnn.h"
#include "quarks.h"
#include <inttypes.h>

/*! \brief Находит источник в графе, для отображения зависимостей
    \param nodes 
    \param i 
    \param arg_idx для аргумента с индексом 
    \return relative position 1..i
 */
static int _get_offset(struct ggml_tensor ** nodes, int i, int arg_idx){
    int offs=0;
    int N = MIN(128,i);
    const struct ggml_tensor * node = nodes[i];
    if (node->src[arg_idx]!=NULL){
        for (offs=1;offs<=N; offs++)
            if (node->src[arg_idx] == nodes[i-offs]) break;
        if(offs>N) offs=0;//(cgraph->nodes[0] - node->src[arg_idx])/GGML_TENSOR_SIZE;
    } else 
        return -1;
    return offs;
}
/*! \brief ccылка на матрицу весов
 */
static int _get_leaf(const struct ggml_cgraph * cgraph, struct ggml_tensor * src){
    if ( src->op==GGML_OP_NONE ) 
    for (int i=0; i<cgraph->n_leafs; i++)
        if (cgraph->leafs[i]== src) return i;
    return -1;
}
/*! \brief находит длину повторяющейся последовательности для заданной позиции и смещения
 */
static int _sequence_len(const struct ggml_cgraph * cgraph, int i, int offs){
    const struct ggml_tensor * node = cgraph->nodes[i];
    const struct ggml_tensor * next = cgraph->nodes[i + offs];
    if (next->op != node->op) return 0;
    int  k;
    for (k=0; i+k+offs <cgraph->n_nodes; k++) {
        const struct ggml_tensor * tn0 = cgraph->nodes[i+k];
        const struct ggml_tensor * tn1 = cgraph->nodes[i+k+offs];
        if (tn0->op != tn1->op || tn0->type != tn1->type)
            break;
        // проверять аргументы по способу адресации и по размерности
        int j=0;
        for(j=0; j<GGML_MAX_SRC; j++)
        {
            int arg0 = _get_offset(cgraph->nodes,       i+k, j);
            int arg1 = _get_offset(cgraph->nodes+offs,  i+k, j);
            if (arg0!=arg1)//  && (cgraph->nodes[i+k]->flags&GGML_TENSOR_FLAG_INPUT)==0
                break;
            const struct ggml_tensor * t0 = tn0->src[j];
            const struct ggml_tensor * t1 = tn1->src[j];
            int are_same_shape = (t0 == t1) || (ggml_are_same_shape(t0, t1) && ggml_are_same_stride(t0, t1));
            if(!are_same_shape)
                break;
        }
        if (j!=GGML_MAX_SRC) break;
        // сравнить флаги
        if (tn0->flags != tn1->flags)
            break;
        // сравнить или нет?
        if (/* tn0->view_src != tn1->view_src || */ tn0->view_offs != tn1->view_offs){// view_src может быть не null см. _OP_NORM
            GGML_LOG_INFO("%d: view %s %zu %zu\n", i+k, ggml_op_name(node->op), tn0->view_offs, tn1->view_offs);
            // break;
        }
        if (tn0->buffer != tn1->buffer){// struct ggml_backend_buffer
            GGML_LOG_INFO("%d: vibackend buffer %s\n", i+k, ggml_op_name(node->op));
            // break;
        }
        // если оба содержат списки параметров, сравнить списки параметров
        if(1){
            int r;
            for (r=0; r<GGML_MAX_OP_PARAMS/sizeof(int32_t); ++r)
                if (tn0->op_params[r] != tn1->op_params[r]) {
                    //GGML_LOG_INFO("%d: param %s %d %d\n", r, ggml_op_name(node->op), i+offs, k);
                    break;
                }
            if (r!=GGML_MAX_OP_PARAMS/sizeof(int32_t)) break;
        }
    }
    GGML_LOG_INFO("%d: rle %s %d %d\n", i, ggml_op_name(node->op), i+offs, k);
    return k;// длина последовательности
}

/*! \brief выделение повторяющихся последовательностей
 */
static int _sequence(const struct ggml_cgraph * cgraph, int i){
    const struct ggml_tensor * node = cgraph->nodes[i];
    int N = MIN(128,cgraph->n_nodes);// глубина поиска не более
    int k=0;
    for (int offs=8;offs<N && i+2*offs<cgraph->n_nodes; offs++){
        const struct ggml_tensor *  next = cgraph->nodes[i + offs];
        if (next->op == node->op) {
            
            for (k=0; k<offs; k++) {
                if (cgraph->nodes[i + k]->op !=cgraph->nodes[i + offs+k]->op)
                    break; // дополнительно можно проверять аргументы по адресации и по размерности
                int arg0 = _get_offset(cgraph->nodes, i+k, 0);
                int arg1 = _get_offset(cgraph->nodes+offs, i+k, 0);
                
                if (arg0!=arg1)//  && (cgraph->nodes[i+k]->flags&GGML_TENSOR_FLAG_INPUT)==0
                    break;
                {                
                    const struct ggml_tensor * t0 = cgraph->nodes[i+k]->src[0];
                    const struct ggml_tensor * t1 = cgraph->nodes[i+k+offs]->src[0];
                    int are_same_shape = (t0 == t1) || (ggml_are_same_shape(t0, t1) && ggml_are_same_stride(t0, t1));
                    if(!are_same_shape)
                        break;
                }
                arg0 = _get_offset(cgraph->nodes, i+k, 1);
                arg1 = _get_offset(cgraph->nodes+offs, i+k, 1);
                if (arg0!=arg1)
                    break;
                {
                    const struct ggml_tensor * t0 = cgraph->nodes[i+k]->src[1];
                    const struct ggml_tensor * t1 = cgraph->nodes[i+k+offs]->src[1];
                    int are_same_shape = (t0 == t1) || (ggml_are_same_shape(t0, t1) && ggml_are_same_stride(t0, t1));
                    if(!are_same_shape)
                        break;
                }
            }
            // GGML_LOG_INFO("%d: %s %d %d\n", i, ggml_op_name(node->op), i+offs, k);
        }
        if (k==offs) {
            return offs;// цикл найден
        }
    }
    return 0;
}
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
    htable->dynstr = malloc(1024);// строки
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

/* перечислить все имена */
static HTable_t * _graph_names(const struct ggml_cgraph * cgraph, Quark_t* dynsym) {
    HTable_t * ht = malloc(Nbucket*sizeof(uint32_t) + sizeof(struct _HTable));
    
    _htable_init(ht, Nbucket);
    size_t str_len = 0;
    GGML_LOG_INFO("=== GRAPH NAMESPACE ===\n");
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        for (int k = 0; k<GGML_MAX_SRC; k++){
            struct ggml_tensor * src = node->src[k];
            if (src!=NULL /* && src->name!= NULL */
            &&  src->op==GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_PARAM)){// см ggml_visit_parents()
                char* cname = ht->dynstr + str_len;
                if (str_len+GGML_MAX_NAME > ht->dynstr_allocated) {
                    ht->dynstr = realloc(ht->dynstr, ht->dynstr_allocated+2048);
                    ht->dynstr_allocated+=2048;
                }
                char* end = NULL;
                int len = 0;// длина строки
                int index = _cname(cname, src->name, &end);
                int idx = _htable_lookup(ht, cname, dynsym);
                if (idx==STN_UNDEF) {
                    idx = _htable_insert(ht, cname);
                    dynsym[idx].st_name = cname - ht->dynstr;
                    str_len = end - ht->dynstr;
                }
                //GGML_LOG_INFO("cname='%s'[%d]=m%d\n", cname, index, idx);
            }
        }
    }
    GGML_LOG_INFO("=== GRAPH HTABLE STRINGS ===\n");
    for (int i=0; i<ht->nchain; i++){
        GGML_LOG_INFO("%2d:str='%s'\n", i, ht->dynstr + dynsym[i].st_name);
    }
    return ht;
}
void ggml_graph_print(const struct ggml_cgraph * cgraph) {
    char cname[GGML_MAX_NAME];
    Quark_t * dynsym = malloc(2048*sizeof(Quark_t));
    HTable_t * ht = _graph_names(cgraph, dynsym);
    char buf[128]; buf[0] =0;
    int  n=0, begin=0;
    int i;
    int skip=0;
    int range_min= 0;
    int range_max=40;
    if (1) for (i=range_min;i<range_max; i++) {
        n = _sequence(cgraph, i);
        if (n>8) break;
    }
    if (i!=range_max) begin = i-n;
    // найдена последовательность надо выяснить число повторов
    GGML_LOG_INFO("- repeat %d:%d\n", i, n);
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    for (i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        int offs = 0;
        int arg0 = _get_offset(cgraph->nodes, i, 0);
        int arg1 = _get_offset(cgraph->nodes, i, 1);
        const char* comment = "";
        enum ggml_type type=0; 
        buf[0] =0;
        if (arg0>=0 && node->src[0]!=NULL) {
            int idx = _cname(cname, node->src[0]->name, NULL);
            offs += sprintf(buf+offs, "%c%d(%zu,%zu,%zu)\t", (arg0>0?'r':'m'), 
                (arg0==0?_htable_lookup(ht, cname, dynsym):arg0), 
                // ggml_type_name(node->src[0]->type),
                node->src[0]->ne[0], node->src[0]->ne[1], node->src[0]->ne[2]);
            if (arg0==0) {
                comment = ggml_get_name(node->src[0]);
                type = node->src[0]->type;
                //leaf = _get_leaf(cgraph, node->src[0]);
            }
        }
        if (arg1>=0 && node->src[1]!=NULL) {
            int idx = _cname(cname, node->src[1]->name, NULL);
            offs += sprintf(buf+offs, "%c%d(%zu,%zu,%zu)", (arg1>0?'r':'m'), 
                (arg1==0?_htable_lookup(ht, cname, dynsym):arg1), 
                //(arg1==0?_get_leaf(cgraph, node->src[1]):arg1), 
                // ggml_type_name(node->src[1]->type),
                node->src[1]->ne[0], node->src[1]->ne[1], node->src[1]->ne[2]);
            if (arg1==0) {
                comment = ggml_get_name(node->src[1]);
                type = node->src[1]->type;
                //leaf = _get_leaf(cgraph, node->src[1]);
            }
        }

        if (n>0 && begin + n==i ) {
            GGML_LOG_INFO("---------------------------------\n");
            if (begin>0){
                int len = _sequence_len(cgraph, begin, n);
                skip = i+len;//(len >= n);
                begin = i;
                i+= len-1;
                continue;
            }
            begin = i;
        }
        if (i >= skip) GGML_LOG_INFO(" - %3d: [%5" PRId64 ",%5" PRId64 ",%5" PRId64 "] %-9.9s %-40.32s ;%s .%s\n",
                i,
                node->ne[0], node->ne[1], node->ne[2],
                (node->op == GGML_OP_UNARY? ggml_unary_op_name(ggml_get_unary_op(node)):ggml_op_name((node->op))), 
                buf, comment, ggml_type_name(type)
//              (node->flags & GGML_TENSOR_FLAG_INPUT) ? "in" : 
//                (node->flags & GGML_TENSOR_FLAG_PARAM) ? "x" : /*ggml_graph_get_grad(cgraph, node) */0? "g" : " "
                );
            
    }
    //if (0)

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    if(0) for (i = 0; i < cgraph->n_leafs; i++) {
        struct ggml_tensor * node = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %4" PRId64 ", %4" PRId64 "] %6s %-.24s\n",
                i,
                node->ne[0], node->ne[1],
                ggml_op_name(node->op),
                ggml_get_name(node));
    }

    GGML_LOG_INFO("========================================\n");
}
