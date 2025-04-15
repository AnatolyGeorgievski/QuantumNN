#include <stdint.h>
#include <math.h>
static inline int clip(int x, int upper) {
    return (x>=upper?(upper-1):(x<0?0:x));
}
static inline float lerp(float s, float e, float t) {
    return fmaf(t, e - s, s);// mix(s,e, t);
}
static float cubic_interp (
	float v[4],  //interpolation points
	float x      //point to be interpolated
)
{
    float d3 = 3*(v[1] - v[2]) - (v[0] - v[3]);
	return  v[1] + 0.5f * x * ((v[2] - v[0]) +
		x * ((v[0] - v[1]) - (v[1] - v[2]) - d3 +
		x * d3));
}
static float cubic_u8_interp (
	uint8_t v[4],  //interpolation points
	float x      //point to be interpolated
)
{
    int d3 = 3*(v[1] - v[2]) - (v[0] - v[3]);
	return  v[1] + 0.5f * x * ((v[2] - v[0]) +
		x * ((v[0] - v[1]) - (v[1] - v[2]) - d3 +
		x * d3));
}
/*!
    \param stride - длина строки исходного изображения в байтах (nb01)
Замечание - использует одну точку по горизонтали и вертикали, нужна функция clip 
или выбрать фрагмент изображения чуть меньше, чтобы не выходило за границы

Три варианта алгоритма эквиваленты. Пока не решил, какой оставить. 
    \todo векторизовать явно цикл по компонентам цвета
 */ 
void bicubic(uint8_t *src, uint8_t *dst,  
    uint32_t ne00, uint32_t ne01, uint32_t stride,
    uint32_t ne0,  uint32_t ne1,  uint32_t nb1,
    int n_channels)
{
    const int nc = n_channels; //число каналов цветности
    //src+= nc;
    unsigned int i,j,k,r;
    int x,y;
    float dx,dy;
    float tx,ty;
// я уменьшил и сдвинул изображение, чтобы не считать clip()
    tx = (float)(ne00-3) /ne0;
    ty = (float)(ne01-3) /ne1;
        
    for(i=0; i<ne1; i++)
    for(j=0; j<ne0; j++)
    {
        x = truncf(tx*j);
        y = truncf(ty*i);
        dx= tx*j - x;
        dy= ty*i - y;
        float v[4] = {1, dy, dy*dy, dy*dy*dy};
        float u[4] = {1, dx, dx*dx, dx*dx*dx};
        for(k=0;k<nc;k++)
        {// векторизовать по цветам
            float c[4];// можно сделать int32, если dx задавать дробью
            for(r=0;r<4;r++)
            {
                uint8_t d[4];
                d[0] = src[(y+r)*stride + (x+0)*nc +k];
                d[1] = src[(y+r)*stride + (x+1)*nc +k];
                d[2] = src[(y+r)*stride + (x+2)*nc +k];
                d[3] = src[(y+r)*stride + (x+3)*nc +k];
#if 0

                float a[4];
                a[0] = d[1];
                a[1] = -1.f / 3 * (d[0]-d[1]) + (d[2]-d[1]) - 1.f / 6 * (d[3]-d[1]);
                a[2] =  1.f / 2 * (d[0]-d[1]) +      1.f / 2 * (d[2]-d[1]);
                a[3] = -1.f / 6 * (d[0]-d[1]) -      1.f / 2 * (d[2]-d[1]) + 1.f / 6 * (d[3]-d[1]);
                c[r] = a[0] + a[1] * dx + a[2] * dx * dx + a[3] * dx * dx * dx;
#elif 1 // этот вариант быстрее работает
                int32_t a[4];
                a[0] =           2*d[1];
                a[1] = -1*d[0]          + 1*d[2];
                a[2] =  2*d[0] - 5*d[1] + 4*d[2] - 1*d[3];
                a[3] = -1*d[0] + 3*d[1] - 3*d[2] + 1*d[3];
                c[r] = (a[0]*u[0] + a[1]*u[1] + a[2]*u[2] + a[3]*u[3])*0.5f;
#else
                c[r] = cubic_u8_interp(d, dx);
#endif
            }
#if 1 // этот вариант быстрее
          float b[4];
            b[0] =           2*c[1];
            b[1] = -1*c[0]          + 1*c[2];
            b[2] =  2*c[0] - 5*c[1] + 4*c[2] - 1*c[3];
            b[3] = -1*c[0] + 3*c[1] - 3*c[2] + 1*c[3];

            float Cc = (b[0]*v[0] + b[1]*v[1] + b[2]*v[2] + b[3]*v[3])*0.5f;
#else
            float Cc = cubic_interp(c, dy);
#endif
            // совместить с нормализацией: (x-mean)/std
            // разделить на планы по цветам
            dst[i*nb1 +j*nc +k] = fminf(255.f,fmaxf(0.f, roundf(Cc)));// convert_uchar_rne_sat()
        }
    }
}

void bilinear(uint8_t *src, uint8_t *dst,  
    uint32_t ne00, uint32_t ne01, uint32_t stride,
    uint32_t ne0,  uint32_t ne1,  uint32_t nb1,
    int n_channels)
{
    const int nc = n_channels; //число каналов цветности
    //src+= nc;
    unsigned int i,j,k,r;
    unsigned int x,y;
    float dx,dy;
    float tx,ty;
// уменьшил изображение на один пиксель, чтобы не считать clip()
    tx = (float)(ne00-1) /ne0;
    ty = (float)(ne01-1) /ne1;
        
    for(i=0; i<ne1; i++)
    for(j=0; j<ne0; j++)
    {
        x = truncf(tx*j);
        y = truncf(ty*i);
        dx= tx*j - x;
        dy= ty*i - y;
        float v[4] = {1, dy, dy*dy, dy*dy*dy};
        float u[4] = {1, dx, dx*dx, dx*dx*dx};
        for(k=0;k<nc;k++)
        {
            float c[2];
            uint8_t d[2];
            d[0] = src[(y+0)*stride + (x+0)*nc +k];
            d[1] = src[(y+0)*stride + (x+1)*nc +k];
            c[0] = lerp(d[0], d[1], dx);
            d[0] = src[(y+1)*stride + (x+0)*nc +k];
            d[1] = src[(y+1)*stride + (x+1)*nc +k];
            c[1] = lerp(d[0], d[1], dx);

            float Cc = lerp(c[0], c[1], dy);
            dst[i*nb1 +j*nc +k] = fminf(255.f,fmaxf(0.f, roundf(Cc)));// convert_u8_rne_sat()
        }
    }
}