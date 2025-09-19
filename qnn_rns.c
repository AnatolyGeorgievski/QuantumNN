/*! \file qnn_rns.c 
    \brief Residue number system (RNS)

    Эта система представляет целые числа в виде вектора остатков от деления на несколько 
    взаимно простых модулей меньшей разрядности. 
    \see doc/RNS.md

Сокращения:
    CRT -- chinese remainder theorem
    RNS -- residue number system, непозиционная система остаточных классов
    MRC -- mixed radix conversion, позволяет восстановить число из остатков используя 
        позиционную систему из модулей
 */
#include <stdint.h>
#include <math.h>
static inline int32_t MODB(int64_t x, const uint32_t q){
    return ((int64_t)x)%q;
}
static inline int32_t ADDM(int32_t a, int32_t b, const uint32_t q){
    return ((int64_t)a+b)%q;
}
static inline int32_t SUBM(int32_t a, int32_t b, const uint32_t q){
    return ((int64_t)a-b)%q;
}
static inline int32_t MULM(int32_t a, int32_t b, const uint32_t q){
    return ((int64_t)a*b)%q;
}
static inline int32_t SQRM(int32_t a, const uint32_t q){
    return ((int64_t)a*a)%q;
}
static int32_t POWM(const int32_t b, int32_t a, const uint32_t q)
{
	int32_t r = b;
	int32_t s = 1;
    while (a!=0) {
		if (a&1) 
			s = MULM(s,r,q);
		r = SQRM(r,q);
		a>>=1;
	}
	return s;
}
static inline int32_t INVM(int32_t a, const uint32_t q){
    return POWM(a, q-2, q);
}

/*! \brief Кодирование в базисе RNS 
    \param[in] x число
    \param[out] r вектор остатков
    \param[in] p вектор взаимно простых модулей
    \param[in] n количество модулей
 */
void rns_encode(int64_t x, int32_t* r, const uint32_t* p, int n){
    for(int i=0; i<n; i++)
        r[i] = MODB(x, p[i]);
}
/*! \brief Расчет множителя g для CRT */
int32_t rns_factor(int32_t* a, const uint32_t* p, int n){
    float z = 0;
    for(int i=0; i<n; i++){
        int32_t pt = 1;// \tilde{p}_i
        for (int j=0; j<n; j++)
            if (i!=j) pt = MULM(pt,p[j],p[i]);
        pt = INVM(pt, p[i]);
        int32_t xi = MULM(a[i], pt, p[i]);
        z += (float)xi/p[i];
    }
    return rint(z);
}
/*! \brief Расширение базиса RNS
    \param[in] a вектор остатков в RNS
    \param[in] p вектор взаимно простых модулей
    \param[in] n количество модулей
    \param[in] q модуль расширения взаимно простой к {p}
    \return число по модулю q
 */
int32_t rns_ext(int32_t* a, const uint32_t* p, int n, uint32_t q){
    int32_t xi[n];
    float z = 0;
    for(int i=0; i<n; i++){
        int32_t pt = 1;
        for (int j=0; j<n; j++)
            if (i!=j) pt = MULM(pt,p[j],p[i]);
        pt = INVM(pt, p[i]);
        xi[i] = MULM(a[i], pt, p[i]);
        z += (float)xi[i]/p[i];
    }
    int32_t e = rint(z);
    int64_t x = 0;
    for(int i=0; i<n; i++){
        uint32_t ph =1;
        for(int j=0; j<n; j++)
            if (i!=j) ph = MULM(ph,p[j],q);
        x += MULM(xi[i],ph,q);
    }
    x = MODB(x,q);
    int32_t P = 1;
    for(int i=0; i<n; i++){
        P = MULM(P,p[i],q);
    }
    x = MODB(x + q - MULM(e,P,q), q);
    return x;
}
/*! \brief Преобразование в позиционную систему из RNS через MRC
    \param[in] a вектор остатков в RNS
    \param[in] p вектор взаимно простых модулей
    \param[in] n количество модулей
    \return число в стандартной системе
 */
int64_t rns_restore(int32_t* a, const uint32_t* p, int n){
// Шаг 1: Расчет коэффициентов g
    int32_t g[n];
    g[0] = 1;
    for (int k=1; k<n; k++){
        int32_t P = p[0];// %p[k];
        for(int j=1; j<k; j++)
            if (j!=k) P = MULM(P,p[j],p[k]);
        g[k] = INVM(P, p[k]);
    }
// Шаг 2: Расчет коэффициентов MRC из RNS 
    int32_t u;
    int32_t v[n];
    v[0] = a[0];
    for(int k=1; k<n; k++){
        u = v[k-1];
        for(int i=k-2; i>=0; i--)
            u = MODB((int64_t)u*p[i] + v[i],p[k]);
        v[k] = MULM(SUBM(a[k], u, p[k]),g[k], p[k]);
    }
// Шаг 3: Расчет стандартного представления числа из MRC
    int64_t x;
    x = v[n-1];
    for(int i=n-2; i>=0; i--){
        x = x*p[i] + v[i];
    }
    return x;
}
#ifdef TEST_RNS
int32_t primes[] = {
    0x7ffd5601, 0x7ffd2601, 0x7ff8e201, 0x7ff83a01, 
    0x7ff82e01, 0x7ff04201, 0x7fee9201, 0x7fea4201,
};
#include <stdio.h>
int main(int argc, char* argv[]){
    int n = 5;
    int count = 20;
    int32_t a_rns[n];
    int32_t q = primes[n];
    for (int64_t i=0; i<0x1FFFFFF; i++){
        int64_t a = i<<33;
        rns_encode(a, a_rns, primes, n);
        int64_t x = rns_restore(a_rns, primes, n);
        if (a!=x) {
            printf("fail:%d %llx != %llx\n", i, a, x);
            if (--count==0)
                break;
        }
        int64_t a_q = rns_ext(a_rns, primes, n, q);
        if (a_q != a%q) {
            printf("fail ext:%d %llx != %llx\n", i, a%q, a_q);
            if (--count==0) break;
        }
    }
    return 0;
}
#endif