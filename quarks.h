
#pragma once

#include <stdint.h>
#include <ctype.h>

typedef struct _QTable QTable_t;
extern QTable_t * _quark_new(uint32_t n_bucket, uint32_t n_chain);
extern int  _quark_lookup(QTable_t * htable, const char *cname);
extern int  _quark_insert(QTable_t * htable, const char *cname);
extern void _quark_to_csv(QTable_t * htable);