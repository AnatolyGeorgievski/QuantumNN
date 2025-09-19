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
static uint32_t POWM(const uint32_t b, uint32_t a, const uint32_t q)
{
	uint32_t r = b;
	uint32_t s = 1;
    while (a!=0) {
		if (a&1) 
			s = ((uint64_t)s*r)%q;
		r = ((uint64_t)r*r)%q;
		a>>=1;
	}
	return s;
}
static inline uint32_t INVM(uint32_t a, const uint32_t q){
    return POWM(a, q-2, q);
}


#include <stdio.h>
#include <math.h>

void rns_encode(uint64_t x, uint32_t* r, const uint32_t* p, int n){
    for(int i=0; i<n; i++)
        r[i] = x % p[i];
}
int rns_factor(int32_t* a, const uint32_t* p, int n){
    int32_t xi[n];
    double z = 0;
    for(int i=0; i<n; i++){
        uint32_t pt = 1;// \tilde{p}_i
        for (int j=0; j<n; j++)
            if (i!=j) pt = ((uint64_t)pt*p[j]) %p[i];
        pt = INVM(pt, p[i]);
        xi[i] = ((int64_t)a[i]* pt) % p[i];
        z += (double)xi[i]/p[i];
    }
    return floor(z);
}
int64_t rns_ext(int32_t* a, const uint32_t* p, int n, uint32_t q){
    int32_t xi[n];
    float z = 0;
    for(int i=0; i<n; i++){
        uint32_t pt = 1;// \tilde{p}_i
        for (int j=0; j<n; j++)
            if (i!=j) pt = ((uint64_t)pt*p[j]) %p[i];
        pt = INVM(pt, p[i]);
        xi[i] = (int64_t)a[i]* pt % p[i];
        z += (float)xi[i]/p[i];
    }
    int32_t e = rint(z);
    //printf("e  :%d\n", e);
    int64_t x = 0;
    for(int i=0; i<n; i++){
        uint32_t ph =1;
        for(int j=0; j<n; j++)
            if (i!=j) ph = (uint64_t)ph*p[j] %q;
        x += (int64_t)xi[i]*ph %q;
    }
    x %= q;
    uint32_t P = 1;
    for(int i=0; i<n; i++){
        P = ((uint64_t)P*p[i]) %q;
    }
    x = (x + q - ((int64_t)e* P)%q) %q;
    return x;
}
int64_t rns_restore(uint32_t* a, const uint32_t* p, int n){
// Шаг 1: Расчет коэффициентов g
    uint32_t g[n];
    g[0] = 1;
    for (int k=1; k<n; k++){
        uint32_t P = 1;//p[0] %p[k];
        for(int j=0; j<k; j++)
            if (j!=k) P = ((uint64_t)P*p[j]) %p[k];
        g[k] = INVM(P, p[k]);
    }
// Шаг 2: Расчет коэффициентов MRC из RNS 
    int64_t x;
    int32_t v[n];
    v[0] = a[0];
    for(int k=1; k<n; k++){
        x = v[k-1];
        for(int i=k-2; i>=0; i--)
            x = (x*p[i] + v[i]) %p[k];
        v[k] = ((int64_t)a[k] - x)*g[k]  %p[k];
    }
// Шаг 3: Расчет стандартного представления числа из MRC
    x = v[n-1];
    for(int i=n-2; i>=0; i--){
        x = x*p[i] + v[i];
    }
    return x;
}
uint32_t primes[] = {
    0x7ffd5601, 0x7ffd2601, 0x7ff8e201, 0x7ff83a01, 
    0x7ff82e01, 0x7ff04201, 0x7fee9201, 0x7fea4201,
    0xffffd001, 0xfffbb001, 0xfff16001, 0xffddb001, 
    0xffd4b001, 0xffbd7001, 0xffb5f001, 0xffb11001, 
    0xffae7001, 0xff83b001, 0xff5da001, 0xff502001, 
    0xff4ae001, 0xff382001,
};
#include <stdio.h>
int main(int argc, char* argv[]){
    int n = 5;
    int count = 20;
    uint32_t a_rns[n];
    uint32_t q = primes[n];
    for (uint64_t i=0; i<0x1FFFFFF; i++){
        uint64_t a = i<<17;
        rns_encode(a, a_rns, primes, n);
        uint64_t x = rns_restore(a_rns, primes, n);
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