/*

$ gcc -O3 -march=native -o mwc mwc_png.c -lpng -lz
$ ./mwc test.png


*/
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <png.h>
#include <zlib.h>

#define OK 0
#define ERROR -1

png_voidp error_ptr = NULL;
png_error_ptr error_fn = NULL; 
png_error_ptr warn_fn  = NULL;
/* Write a png file */
int write_png(char *file_name, uint8_t* image, int width, int height)
{
	int png_transforms = PNG_TRANSFORM_IDENTITY;
	
	FILE *fp;
	png_structp png_ptr;
	png_infop   info_ptr;

	/* Open the file */
	fp = fopen(file_name, "wb");
	if (fp == NULL)
		return ERROR;

   /* Create and initialize the png_struct with the desired error handler
    * functions.  If you want to use the default stderr and longjump method,
    * you can supply NULL for the last three parameters.  We also check that
    * the library version is compatible with the one used at compile time,
    * in case we are using dynamically linked libraries.  REQUIRED.
    */
   png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
		error_ptr, error_fn, warn_fn);
   if (png_ptr == NULL)
   {
      fclose(fp);
      return ERROR;
   }

   /* Allocate/initialize the image information data.  REQUIRED. */
   info_ptr = png_create_info_struct(png_ptr);
   if (info_ptr == NULL)
   {
      fclose(fp);
      png_destroy_write_struct(&png_ptr,  NULL);
      return ERROR;
   }

   /* Set up error handling.  REQUIRED if you aren't supplying your own
    * error handling functions in the png_create_write_struct() call.
    */
   if (setjmp(png_jmpbuf(png_ptr)))
   {
      /* If we get here, we had a problem writing the file. */
      fclose(fp);
      png_destroy_write_struct(&png_ptr, &info_ptr);
      return ERROR;
   }
	png_init_io(png_ptr, fp);

int bit_depth = 16;
const int bytes_per_pixel = 2;
	png_set_IHDR  (png_ptr, info_ptr, width, height, bit_depth,
       PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
       PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info_before_PLTE(png_ptr, info_ptr);
	png_write_info(png_ptr, info_ptr);

	for (int y = 0; y < height; y++)
         png_write_row(png_ptr, image + y * width * bytes_per_pixel);

	png_write_end  (png_ptr, info_ptr);

	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);
	return OK;
}

#ifdef TEST_PNG

#include <stdint.h>
uint32_t mwc32x( uint32_t* state, const uint32_t A)
{
	uint32_t x = *state;
	*state = A*(uint16_t)(x) + (x>>16);
    return x;//((x>>16) ^ (x&0xFFFFU));
}
uint32_t mwc24x( uint32_t* state)
{
	const uint32_t A=0xF3;//F3
	uint32_t x = *state;
	*state = A*(x&0xFFFFFFu) + (x>>24);
    return x;
}
/* преобразование чисел в формат float32 дает распрделение [0,1) */
static inline float u64_float(uint64_t x) {
	return ((uint32_t)((x>>32) ^ (x&0xFFFFFFFFU)) >> 8) * 0x1.0p-24;
}
float uniform (uint32_t *state, uint32_t A1) {
	mwc32x(state, A1);
	return ((mwc32x(state, A1)<<6)&0xFFFFFFU)*0x1.0p-24f;
}
float uniform24 (uint32_t *state) {
	mwc24x(state);
	return ((mwc24x(state))&0xFFFFFFU)*0x1.0p-24f;
}

//#define MWC_A0 0xfffeb81bULL
#define MWC_A0 0xffe67fc0ULL
#define MWC_A1 0xFFFEB7CDULL


/*! \brief Генерация псевдо-случайного числа. Один шаг алгоритма */
uint64_t mwc64x( uint64_t* state)
{
	uint64_t x = *state;
	*state = MWC_A0*(uint32_t)(x) + (x>>32);
    return x;//((x>>32) ^ (x&0xFFFFFFFFU));
}

float uniform64 (uint64_t *state/*, uint32_t A1 */) {
	//mwc64x(state);
	return ((mwc64x(state))&0xFFFFFFU)*0x1.0p-24f;
}
#include <math.h>

int main(int argc, char *argv[]){

	const unsigned width = 1024;
	const unsigned height= 1024;
	// 0xFE94, 0xFEA0;//0xFE30;
	const uint32_t A1 =0xFEA0;//0xFE94;//0xFF00;//0xFFA8;//0xFEE4;

	if (argc<2) return 1;
	char *file_name = argv[1];

	uint16_t *image = malloc(width*height*2);
	memset (image, 0, width*height*2);

	uint32_t s = 0xFFFF;
	uint64_t s64 = 0x1;
	uint32_t MASK = (width*height -1);
	for (uint64_t i=0; i<=0x7FFFuLL*width*height; i++){
#if 1
		uint32_t x = uniform64(&s64)*width;
		uint32_t y = uniform64(&s64)*height;
#elif 0
		uint32_t x = uniform24(&s)*width;
		uint32_t y = uniform24(&s)*height;
#elif 1
		uint32_t x = uniform(&s, A1)*width;
		uint32_t y = uniform(&s, A1)*height;
#elif 0
		uint32_t v = mwc32x(&s, A1);
		uint32_t x = (v>>10)&0x3FF;
		uint32_t y = v&0x3FF;
#else
		mwc32x(&s, A1);
		uint32_t x = (mwc32x(&s, A1)^mwc32x(&s, A1))%width;
		uint32_t y = (mwc32x(&s, A1)^mwc32x(&s, A1))%height;
#endif		
		//if (v<width*height) 
		uint16_t col = __builtin_bswap16(image[x + y*width]);
		col += 1;
		if (col<1) col = (1<<16)-1;
		image[x + y*width] = __builtin_bswap16(col);
	}

	
	double sum = 0;
	for (int y=0; y<height; y++){
		double row = 0;
		for (int x=0; x<width; x++){
			row += __builtin_bswap16(image[x + y*width]);
		}
		sum+=row*(1.0/width);
	}
	double avg = sum * (1.0/height);
	
	sum = 0;
	for (int y=0; y<height; y++){
		double row = 0;
		for (int x=0; x<width; x++){
			double val = avg - __builtin_bswap16(image[x + y*width]);
			row += val*val;
		}
		sum+=row;
	}
	double dev = sqrt(sum) * (1.0/(height*height));
	
	printf("average =%1.1f dev = %1.1g\n", avg, dev/avg);
	write_png(file_name, (uint8_t*)image, width, height);
	return 0;
}
#endif