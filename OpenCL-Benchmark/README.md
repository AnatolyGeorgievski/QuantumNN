Тест производительности OpenCL-Benchmark
==============================


## Сборка и запуск

Сборка теста с использованием Makefile. В оригинальной версии используются библиотеки OpenCL в составе портируемого пакета, в нашей версии используются системные библиотеки. Для запуска теста необходимо установить `opencl-headers` и `opencl-icd` и драйвера от производителя. 

MSYS2: Установка под Windows префикс `ucrt64/mingw-w64-ucrt-x86_64-` или `clang64/mingw-w64-clang-x86_64-`
```sh
pacman -S %{PREFIX}-opencl-clhpp   #    OpenCL C++ header files
pacman -S %{PREFIX}-opencl-headers #    OpenCL (Open Computing Language) header files
pacman -S %{PREFIX}-opencl-icd     #    OpenCL ICD Loader
```
Компиляция с использованием `g++`, при запуске используется порядковый номер карты (0..N).
```sh
$ make
$ ./benchmark 1
```