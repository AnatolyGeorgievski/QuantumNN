https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/cuda/wkv7.cu

## Использование и обозначение квантизации моделей

Это не окончательный вариант, но он полезен при чтении исходного кода или вывода в консоль, чтобы понять, что обычно означает каждое из этих обозначений.

`<Encoding>_<Variants>`\
`<Encoding>` : Это определяет наиболее распространенную кодировку индивидуальных весов в модели
Форматы с плавающей запятой:
* `BF16`:  16-битная Bfloat16 Google Brain усечённая форма 32-битного IEEE 754 (1 знаковый бит, 8 бит показателя степени, 7 дробных бит)
* `F64`: 64-битные числа с плавающей запятой IEEE 754 (1 знаковый бит, 11 бит экспоненты, 52 дробных бита)
* `F32`: 32-битные числа с плавающей запятой IEEE 754 (1 знаковый бит, 8 бит показателя степени, 23 дробных бита)
* `F16`: 16-битные числа с плавающей запятой IEEE 754 (1 знаковый бит, 5 бит порядка, 10 дробных бит)
Целочисленные форматы:
* `I<X>`: X бит на единицу веса, где X может быть 4 (для 4 бит) или 8 (для 8 бит) и т. д...
Квантованные форматы:
* `Q<X>`: X бит на единицу веса, где X может быть `4` (для 4 бит),`5`,`6`  или `8` (для 8 бит) и т. д...
* `KQ<X>` (или `Q<X>_K`) : модели на основе k-квантов. `X` бит на весовой коэффициент, где `X` может быть `4` (для 4-бит), `5`,`6` или `8` (для 8-бит) и т. д...
* `IQ<X>`: модели на основе `i`-квантов. `X` бит на вес, где `X` может быть `4` (для 4-бит) или `8` (для 8-бит) и т. д...

`<Variants>`Это различные стратегии упаковки квантованных весов в файл формата [gguf](/ggerganov/gguf). 

Следует отметить некоторую особенность использования форматов данных при запуске и исполнении моделей. Внутреннее представление модели может быть различным. Модели в высоком разрешении исполняются и сохраняются в памяти в формате Float32.Для повышения производительности на GPU внутреннее представление коэффициентов весовых матриц и векторов может быть представлено в формате Float16 или BFloat16. Внутреннее представление коэффициентов завязано на конкретные операции, такие как активация и нормализация слоя. Результатом активации может быть вектор `F32`(Float32) или `F16`. А вот матрица коэффициентов может храниться и применяться без изменения формата данных, т.е. операция над вектором состояния может быть представлена как произведение коэффициентов матрицы в формате `Q8_1` на состояние в формате `Q8_1` или `F16`. Внутренне представление коэффициентов Может быть представлено в целых или рациональных числах. Так, например, формат `Q8_1` представляет собой набор весовых коэффициентов в формате `int8_t` и отдельно рассчитанный нормировочный коэффициент в формате `float`. Альтернативный способ хранения может содержать нормировочный коэффициент и минимальное (среднее) значение. Исполнение моделей на CPU и используемый при этом формат представления тензоров зависит от поддерживаемой системы команд процессора, формат выбирается исходя из наличия в системе команд тех или иных векторных инструкций. 

На платформе `x86` относительно высокая производительность модели на GPU может быть получена в системе команд с разрядностью векторных регистров 512 бит:
* `AVX512_FP16` -- поддержка операций Float16, и операций скалярного произведения векторов.
* `AVX512_BF16` -- поддержка операций BFloat16.
* `AVX512_IFMA` -- поддержка операций Fused-Multiply-Accumulate в целых числах
* `AVX512_VNNI` -- поддержка инструкций малой разрядности для нейронных сетей
* `AVX512F` -- поддержка инструкций FMA для чисел Float32.

Эти же операции могут быть представлены в системах команд AVX10 для векторов 256 бит.

Использование коэффициентов `BF16` в чем-то близко по качеству к `Q8_0` и `Q8_1`. Качество моделей `F16` заведомо выше, поскольку в формате `F16` разрядность мантиссы выше, чем разрядность `Q8` и `BF16`. Представление в рациональных числах может дать заметный выигрыш и по качеству и по производительности операций, в случае если аппаратно поддерживается операция скалярного произведения векторов _dot-product_ и сложение с накоплением _fma_, без потери точности при сложении.

На GPU представление коэффициентов может быть `Q8_1` или `F16`, варианты представления включают общую нормировку по вектору. 
Дело в том что расчет  функций типа Softmax сопровождается вычислением нормы. А нормализация слоя предусматривает два цикла - расчет суммы и нормировка каждого элемента. Таким образом норма вектора может передаваться на следующий слой, чтобы исключить операцию двойной нормировки элементов вектора. Такая техника существенно ускоряет работу модели.

## RWKV v7

float sa = dot(a,state)
```math
sa =  a_j \ast s_j\\
s_j := s_j \cdot w_j + v_t\cdot k_j + sa \cdot b_j \\
y = r_j \circ ()\\

```

Функция активации exp(-exp(x)) возвращает значение в интервале [0,1]
![Функция активации y = exp(-exp(x))](rwkv_activation.png)

**Softmax**
> Многопеременная логистическая функция Softmax — это обобщение логистической функции для многомерного случая. Функция преобразует вектор  $z$ размерности $K$ в вектор 
$\sigma$ той же размерности, где каждая координата $\sigma _{i}$ полученного вектора представлена вещественным числом в интервале [0,1] и сумма координат равна 1.

Координаты $\sigma _{i}$ вычисляются следующим образом:
```math
\sigma (z)_{i}={\frac {e^{z_{i}}}{\displaystyle \sum _{k\mathop {=} 1}^{K}e^{z_{k}}}}
```
Многопеременная логистическая функция применяется в машинном обучении для задач классификации, когда количество возможных классов больше двух (для двух классов используется логистическая функция). 

Координаты $\sigma _{i}$ полученного вектора при этом трактуются как вероятности того, что объект принадлежит к классу $i$. Вектор-столбец $z$ при этом рассчитывается следующим образом:
$z=w^{T}x-\theta$,\
где $x$ — вектор-столбец признаков объекта размерности $M\times 1$; 
$w^{T}$ — транспонированная матрица весовых коэффициентов признаков, имеющая размерность 
$K\times M$; 
$\theta$ — вектор-столбец с пороговыми значениями размерности 
$K\times 1$ (см. перцептрон), где $K$— количество классов объектов, 
а $M$ — количество признаков объектов.

Часто Softmax используется для последнего слоя глубоких нейронных сетей для задач классификации. Для обучения нейронной сети при этом в качестве функции потерь используется перекрёстная энтропия.

Вектор-столбец $z=y-\max(y)$, где y=$w^Tx$.

**Attn оператор**

$$\operatorname{Attn}(Q,K,V)_t = \frac{\sum_{i=1}^t e^{q_t^\mathsf{T} k_i} \odot v_i}{\sum_{i=1}^t e^{q_t^\mathsf{T} k_i}}$$


$$\operatorname{Attn^{+}}(W,K,V)_t = \frac{\sum_{i=1}^t e^{w_{t,i}+k_i} \odot v_i}{\sum_{i=1}^t e^{w_{t,i}+k_i}}$$

*AFT*, сокращение от *Attention Free Transformer*, представляет собой подход, отличный от традиционного механизма "внимания", и включает в себя изученные парные смещения позиций, обозначаемые, как $w_{t,i}$, где каждое $w_{t,i}$ является скалярным значением.

$\odot$ - произведение Адамара (поэлементное).

**WKV оператор**
В RWKV весовые коэффициенты $w_{t,i}$ рассматриваются, как вектор затухания по каналам. Этот вектор масштабируется в зависимости от относительной позиции и уменьшается c шагом времени в соответствии с правилом: 
$$w_{t,i} = -(t-i)w$$
В итоге оператор запишется
```math
WKV_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} \odot v_i + e^{u+k_t}\odot v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}}
```
Для понимания почему это работает надо рассматривать антологию архитектур. Среди которых выделяется ряд: LSTM GRU и вот тут определили WKV оператор и Mu оператор. В частности определили как связан оператор из сетей Трансформеры ATTN с оператором WKV. Говоря простым языком, ATTN общая структура, которую можно представить, как Softmax и как WKV-оператор. Причем это упрощение снижает сложность обучения. Подход с обучением мне видится как вырождение более общего (избыточного) представления решения в простую форму. Так введение оператора Mu - это тоже упрощение, потому что в этом месте архитектура всегда вырождается в конструкцию типа Mu. Отдельно стоит упомянуть прямую связь обходное значение. Весь элемент сети выражается, как разница (измерение). Изначально эта архитектурная особенность появилась ради возможности тренировать глубокие слои. 

Нащ подход заключается в утверждении, что любую функцию многих переменных с входными значениями $v\in [0,1]$ можно представить, как суперпозицию (сложную функцию) от функций одного переменного и функцию сложения (функция двух переменных), см. KAT - теорема Колмогорова-Арнольда. Суперпозиция функций (операторов) в плане архитектуры сети - это каскад составленный из операторов.

Другое утверждение, что из функций смешивания (mix) можно построить: а) B-сплайн и б) функцию распределения в) цифровой фильтр. Оператор Mu следует преобразовать в элемент цифрового фильтра с обратной связью. 

**$\mu$-оператор**

$x = W \cdot ((1-\mu)\odot x_t + \mu\odot x_{t-1})$

**Нормализация слоя**

x :=  (x - x.mean)/x.std;

* mean - среднее арифметическое
* std - средне-квадратичное отклонение от среднего - норма вектора $\|x - \bar{x}\|$. 

std(x) = sqrt(mean(abs(x - x.mean())**2));


https://ggml.ai/

ĀĿΞX, [11.12.2024 3:27]
Предлагаю попробовать Vulkan

сперва стянем себе LLaMa.cpp

> git clone https://github.com/ggerganov/llama.cpp.git

в ней есть папка пакета ggml. Отдельно установим GGML

> git clone https://github.com/ggerganov/ggml.git
> cp -r ggml llama.cpp/ggml

и скопируем внутрь папки llama.cpp, при этом скачанная GGML заменяет собой то, что было в папке ggml изначально. Теперь перейдём в папку ggml и открываем [CMakeLists.txt](CMakeLists.txt)\
Там есть опции вида:

```
option(GGML_CUDA   "ggml: use CUDA"   OFF)
option(GGML_VULKAN "ggml: use Vulkan" OFF)
option(GGML_OPENCL "ggml: use OPENCL" OFF)
```
ĀĿΞX, [11.12.2024 3:45]
У меня эти опции отображаются в IDE, в виде галок

я для себя включил:
```
GGML_ACCELERATE:BOOL=ON
GGML_AVX:BOOL=ON
GGML_AVX2:BOOL=ON
GGML_CPU:BOOL=ON
GGML_CUDA:BOOL=ON
```
остальное оставил по умолчанию

ĀĿΞX, [11.12.2024 3:50]
Сборку с Vulkan у меня не получилось выполнить из-за проблем с Vulkan-специфическими пакетами для моего драйвера nVidia. Однако, на другой системе с правильными пакетами, я полагаю, должно собираться и работать без проблем.

```
pacman -S mingw64/mingw-w64-x86_64-ninja
$ pip install -r requirements.txt

git clone https://github.com/ggerganov/ggml
cd ggml

# install python dependencies in a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# build the examples
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```

## Мои изменения в коде GGML для запуска моделей на Intel GPU
```
option(GGML_OPENCL                     "ggml: use OpenCL"                       ON)
option(GGML_OPENCL_EMBED_KERNELS       "ggml: embed kernels"                    ON)
option(GGML_OPENCL_USE_ADRENO_KERNELS  "ggml: use optimized kernels for Adreno" OFF)
```
Файл `llama.cpp` использует технологию mmap. На Windows применяется функция ядра (), которая у меня не работают без объявления типа. Объявление типа включаю явным образом в исходник. 
```c
#if defined (_WIN32)
typedef struct _WIN32_MEMORY_RANGE_ENTRY {
  PVOID  VirtualAddress;
  SIZE_T NumberOfBytes;
} WIN32_MEMORY_RANGE_ENTRY, *PWIN32_MEMORY_RANGE_ENTRY;
#endif
```

В моделях может поддерживаться квантизация BF16. 
```
__m512 _mm512_dpbf16_ps (__m512 src, __m512bh a, __m512bh b)
#include <immintrin.h>
Instruction: vdpbf16ps zmm, zmm, zmm
CPUID Flags: AVX512_BF16 + AVX512F

Description
Compute dot-product of BF16 (16-bit) floating-point pairs in a and b, accumulating the intermediate single-precision (32-bit) floating-point elements with elements in src, and store the results in dst.
```
Для разных квантов поддерживаются операции
1. Умножение матриц `mul_mat`,`mul_mv` 
2. Аффинное преобразование? поэлементные операции `mul`, `add`
3. Операция скалярного произведения векторов (`dot` product)
4. Нормализация вектора `norm`, `rms_norm`, `clamp`
5. LERP Операция смешивания векторов `mix`
6. Унарные операции и функции активации `relu`,`silu`,`gelu`,`softmax` и др.

*Требования*
Все операции должны быть выровнены на 128, 256, 512, 1024 бит, и выполняются в группе 32 по элемента (Q8_1) или 256 элементов (Q8_K). 

## Принцип загрузки матриц без копирования

## Тип представления тензоров

* `Q8_1` тип представлен целым числом 8 бит и нормой F16 или F32 Значения пакуются в структуру 
```c
strcuct _q8_0 {
    float32_t d;
    int8_t    qs[Q8_0];
};
```
$y[i] = d * qs[i]$
```c
strcuct _q8_1 {
    float16_t d; // максимальное значение 
    float16_t s; // d* sum(s[i])
    int8_t    qs[Q8_0];// 8 бит на квант
};
```
```c
strcuct _q5_1 {
    float16_t d; // максимальное значение
    float16_t m; // минимальное значение
    uint32_t   qh;       // старший бит
    uint8_t    qs[QK5/2]; // 4 бита на квант
};
```
$y[i] = d * [qh[i], qs[i]] + m$


#### Скалярное произведение векторов

$$ r = a\odot h$$

Числа со знаком или без знака
```c 
void dot(float *r, struct block_Q8_1 *a, float *b){

    float s = 0;
    s+= (a->d)*dot_Q8_F32(a->s, b);

    s = work_group_reduce_add(s);
    if (local_id == 0)
        r[get_group_id(0)] = s*a->d;
}
```
Подобные функции могут существовать в большом количестве экземпляров для разных квантов. 

#### Смешивание (time-mix)

$$r_t = (1-\mu)\odot h_{t-1}+ \mu \odot \tilde{h}_t$$
Операция - линейная интерполяция между внутренним состоянием ($h_{t-1}$)и новым значением ($\tilde{h}$). $\mu \in [0,1]^N$ - коэффициенты нормированные, могут быть представлены в целых числах без знака с общим нормировочным коэффициентом. 

Этот оператор можно назвать дифференциальным оператором. Или оператором линейной интерполяции векторов (сокр. *lerp*).

В системе команд может присутствовать векторная операция `mix(a,b, mu) = a + (b-a) * mu`

Рассмотрим систему дифференциальных уравнений 

В контексте тензорных вычислений мы сводим уравнения к нормальной форме. Т.е. каждую производную по времени мы заменяем на новую переменную. Из этих переменных составим вектор. В общем виде можно представить как
```math
\frac{d}{dt} y = W\cdot y + U\cdot x + b
```
где $W \in \mathbb{R}^{d\times d}$; $U \in \mathbb{R}^{d\times e}$; $x\in \mathbb{R}^{e}$;
$y, b\in \mathbb{R}^{d}$.

Следующее приближение - диагонализация матрицы W. Нам нужна такая замена переменных чтобы уравнение приняло вид
```math
\begin{aligned}
\frac{d}{dt} h &= diag(W)\cdot h + U\cdot \tilde{x} + b\\
y &= W_y \cdot h + b_y
\end{aligned}
```
Для этого возможно разложить практически любую матрицу в произведение трех матриц $W_y \Sigma U$, где матрица $\Sigma$ - диагональная и может быть представлена вектором $\mu$. Тогда уравнение примет вид

```math
\begin{aligned}
\tilde{h}_t &= \delta(U_h\cdot x + b)\\
        h_t &= (1-\mu)\odot h_{t-1} + \mu \odot \tilde{h}_t\\
        y_t &= W_y \cdot h_t + b_y
\end{aligned}
```
Это выражение нужно дополнить. Предлагается ввести две дополнительные операции, которые бы выполняли функцию инициализации (сброса, reset gate) и управляли коэффициентом $\mu$ (управление забыванием) обе функции должны давать значения в диапазоне $[0,1]$. Функцию "активации" мы добавляем, чтобы значение оставалось в заданных пределах. В нашей терминологии это может быть квантизатор с диффузией ошибки, нормировка или нелинейная функция типа `sigmoid()`.
```math
\begin{aligned}
        r_t &= \sigma(W_r\cdot h_{t-1} + U_r\cdot x + b_r)\\
      \mu_t &= \sigma(W_\mu\cdot h_{t-1} + U_\mu\cdot x + b_\mu)\\
\tilde{h}_t &= \theta(W_h\cdot (r_t \odot h_{t-1}) + U_h\cdot x + b)\\
        h_t &= (1-\mu_t)\odot h_{t-1} + \mu_t \odot \tilde{h}_t\\
        y_t &= W_y \cdot h_t + b_y
\end{aligned}
```
Векторная функция $r_t$ может быть логической и принимать два значения 1 или 0 или быть представлена в вещественных или рациональных числах с квантизацией. Функция может быть вероятностной и принимать промежуточные значения на интервале [0,1]. Таким образом функция $\sigma$ в выражениях - квантизатор и/или пороговая функция. Одной из таких функций может быть `softmax`, которая сочетает в себе два свойства - нормировку и пороговую активацию. Функцией активации может быть практически любая функция хоть сколько нибудь напоминающая сглаженную ступеньку. В частности, хорошим кандидатом является функция `smoothstep` - интерполяция кубическим сплайном. Но может быть и функция `clamp`. 

Некоторая вольность в выборе функции активации для "reset gate" строится на предположении, что в результате обучения (подбора коэффициентов и нормализации) логика работы не изменится. Функция активации должна иметь гладкие края и быть многократно дифференцируемой. Это выполняется для интерполяции сплайнами.

Функция забывания $\mu_t$ далеко не всегда вообще должна зависеть от параметра, т.е. ее надо определить как опциональную может быть выражена вектором констант, тогда ее можно паковать методом `Q8`, `F16`. Функция может быть скаляром. 

В научных публикациях и практических реализациях используется множество различных функций активации. Среди этого множества можно выделить два класса, `Re` и `Si` причем к этим классам надо относиться таким образом, чтобы $Si(x) \equiv Re'(x)$. Функция "сигма" должна быть производной от "выпрямителя", чтобы можно было перейти от одного представления узла сети к другому. В курсе математического анализа мы вводим обобщенные и финитные функции (функции определенные на интервале [0,1]) и разбираем базисные полиномы в качестве финитных.

Таким образом мы обозначили возможность представления некоторого алгоритма в форме элемента GRU, (сокр. Gated Recurrence Unit). 

Тут все равно картина не полная. Не полной она является по двум 1) параметрам глубина модели и возможность интерпретации полученных результатов в терминах плотности вероятности и эквивалентного представления физических моделей в гильбертовом пространстве (многомерном и комплексном). 

Элемент "фильтра" в действительных числах должен быть второго и более порядка. Ранее в курсе мы вводили представление оператора задержки $z^{-1}$ для дискретных процессов во времени. Система дифференциальных уравнений может быть записана в операторной форме с использованием оператора задержки. 

Зависимость от входного и выходного значения можно разложить по принципу 
```math
(1 - \bar{\alpha} z^{-1})\hat{H}(z) Y= \alpha \hat{U}(z) X
```
Оператор можно представить, как каскад и сумма операторов вида $(1+ z^{-1}), (1- z^{-1})$ или иными словами. Разложение на одном уровне сети должно давать вторую степень производной или производная должна быть комплексно значная функция. 

# OpenCL C backend

Тут мы ориентируемся на архитектуру Intel GPU. Архитектура состоит из `EU` вычислительных узлов со своей памятью `SLM` (Shared Local Memory 64-256кБ). В SLM поддерживаются атомарные операции и коллективные операции с подгруппами. Каждый EU состоит из 8-и XVE (векторных вычислительных блоков), каждый XVE поддерживает векторизацию до 512 Бит (float16: 16x32бит). Эффективно операции нарезаются по 8 (long), 16 (float) и 32 элемента (half). Каждый такой элемент EU из восьми XVE представляет собой вычислительную подгруппу со своей выделенной локальной памятью/кешем. Отдельно существует возможность исполнения блоков на матричных ускорителях XMX 2048 бит (у DataCentre GPU 4096 бит), которые должны быть ориентированы на упакованные форматы, такие как Q2, Q4, Q8, F16, BF16, TF32. Слайс состоит из 4-х EU. Intel ARC B580 имеет всего на 5 слайсов, на которых приходится 5\*4\*8 = 160 вычислительных блоков XVE и XMX. При этом каждый вычислительный блок может независимо загружать XMX и XVE и отдельно считать функции типа `exp`. Как писать программы, чтобы они грузили оба блока не ясно, документировано плохо. 

Отдельно стоит описать структуру элемента RDNA и CDNA в ускорителях AMD. Там по сути 32 битные процессорные элементы организованные по 64/128 потоков, которые могут загружать векторный блок и ожидать завершения операции на вектором блоке. Хорошая производительность получается, если удается загрузить векторный блок.



## Оптимизация векторных операций

Существует стандартизированный подход к оптимизации векторных операций на EU - через использование коллективных операций типа `sub_group_reduce_{op}`, где операция может быть: `min`, `max`, `add` и битовая `and`, `xor`. Размер подгруппы выбирается исходя из размерности EU под использование SIMD8, SIMD16 и SIMD32 (на Intel GPU). Следует отметить, что ускорители других производителей могут использовать другую компоновку вычислительного узла, например SIMD64 и SIMD128. Максимальный размер подгруппы является параметром компиляции кода. При составлении программы SIMDx - размерность вычислительного блока будет параметром группы, который получается через опрос параметров скомпилированной функции `clGetKernelSubGroupInfo`.

* [] Analysis of OpenCL Work-Group Reduce for Intel GPUs <...>

Для работы с вычислительными блоками мы применяем терминологию подгруппа и тред. 

Техника оптимизации. Основная операция матричных вычислений _dot product_ с редуцированием по подгруппе. Требуется поддержка расширения `__opencl_c_subgroups: enable`
```c
uint stride = get_max_sub_group_size();
float s = 0;
for (uint i=get_sub_group_local_id(); i<N; i+=stride)
    s += dot(a[i], v[i]);
s = sub_group_reduce_add(s);

if (get_sub_group_local_id() == 0) 
    slm[row] = s;
```
Операция суммирования с редуцированием по подгруппе. Число элементов `N` выбирается кратным `max_sub_group_size`. Результат сохраняется в локальную память slm, общую для EU и всех XVE. Подобную технику можно применить и на CPU, если под операцией редуцирования понимать атомарную операцию сложения. Для вычисления матриц можно ориентироваться на правило, что каждый EU считает одну операцию _dot product_. 
Перед обращением к общей переменной в SLM памяти (если переменная изменена в одном из тредов), требуется использование барьера синхронизации `sub_group_barrier()`. Туже операцию можно выполнить с приватными переменными треда используя "рассылку", `sub_group_broadcast()`.

Элементы `a`, `v` могут быть векторные такие как `float4` или `half8`. 

Аппаратно независимая реализация может строиться на использовании понятия work_group, операции над work_group введены в стандарт OpenCL C 3.0. `__opencl_c_work_group_collective_functions : enable`

```c
uint stride = get_local_size(0);
float s = 0;
for (uint i=get_local_id(0); i<N; i+=stride)
    s += dot(a[i], v[i]);
s = work_group_reduce_add(s);

if (get_local_id(0) == 0) 
    slm[get_group_id()] = s;
```
Операция редуцирования по рабочей группе строится следующим образом. Такая конструкция применяется, когда платформа не поддерживает `work_group_reduce_add` с резервированием буфера в локальной памяти [примеры nVidia]:
```c
    partialDotProduct[get_local_id(0)] = sum;
    for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
        // Synchronize to make sure each work-item is done updating
        // shared memory; this is necessary because work-items read
        // results that have been written by other work-items
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Only the first work-items in the work-group add elements together
        if (get_local_id(0) < stride) {
        
            // Add two elements from the "partialDotProduct" array
            // and store the result in partialDotProduct[index]
            partialDotProduct[get_local_id(0)] += partialDotProduct[get_local_id(0) + stride];
        }
    }
    // Write the result of the reduction to global memory
    if (get_local_id(0) == 0)
        W[get_group_id(0)] = partialDotProduct[0];
```
* <https://developer.nvidia.com/opencl>

По сути такая техника применяется только при нормировании матрицы (сумма по группе) и расчете `softmax` (поиск максимума по группе), все остальные матричные операции позволяют ограничится коллективными функциями `sub_group_reduce_{op}`. Коллективные функции могут не поддерживаться на данной платформе.

Операция редуцирования по группе, очевидно, составная и строится на использовании редуцирования по подгруппе и последующего редуцирования по группе с использованием барьеров памяти и атомарных операций. Заметим, что атомарные операции эффективно работают только в пределах EU и кеша первого уровня, в локальной памяти. Эффективная реализация возможна, когда группа по размеру совпадает или кратна размеру подгруппы. Мы ориентируемся на размер группы кратный 128 элементов. Intel GPU поддерживает подгруппы размером 8,16,32.

* <https://github.com/intel/intel-graphics-compiler/tree/master/IGC/BiFModule/Languages/OpenCL>

### DP4a (Dot Product of 4 Elements and Accumulate) 

Поддержка векторной _dot product_ операций над упакованными форматами `Q8`, `Q4`, `Q2`. Для реализации операции следует смотреть стандартные методы `dot_4x8packed_uu`() Расширение `__opencl_c_integer_dot_product_input_4x8bit_packed`

Эмуляция инструкции DP4A
```c
for (i = 0; i < exec_size; ++i) {
    if (ChEn[i]) {
    dst[i] = src0[i] + src1[i][ 7: 0]*src2[i][ 7: 0] + src1[i][15: 8]*src2[i][15: 8] 
                     + src1[i][23:16]*src2[i][23:16] + src1[i][31:24]*src2[i][31:24];
    }
}
```

### DPAS (Dot Product and Accumulate Systolic) 

### FCVT type conversion between FP8 and HF

```c
for (i = 0; i < exec_size; ++i) {
    if (ChEn[i]) {    // ChEn[i] is always true if dst has FP8 type
        dst[i] = src0[i];
    }
}
```
* FP8 here is BF8, aka E5M2
* BF16 bfloat16 is a 16-bit float type (E8M7, aka truncated IEEE 754 single-precision 32-bit float). 
* FP16 is the IEEE 754 half.
* TF32 is 19-bit tensor float type (E8M10), which has 1-bit sign, 8-bit exponent, and 10-bit mantissa.

### применение операции scan и broadcast 

Операции надо рассматривать, как некоторая технология расчета больших векторов, когда вектор разбивается на группы и считается на нескольких вычислительных блоках EU. Существует две стратегии расчета: `scan` подразумевает пересылку результата между блоками XVE и на последнем блоке получаем результат с накоплением. Чтобы использовать результат накопления в последующих вычислениях нужно выполнить `broadcast`, раздать результат во все треды. Техника _scan-broadcast_ выполняется по группе или подгруппе с использованием локальной памяти. 

### Использование семплера для работы с матрицами

В архитектуре элемента EU существует два отдельных конвейера загрузки данных, каждый со своим локальным кешем L1, который может использоваться как SLM. Один из них ассоциирован с подгрузкой фрагмента изображения, второй с локальной памятью EU. Эффективность можно повысить, если использовать семплер для подгрузки коэффициентов матриц. `read_imageui` и тп.

Семплер можно использовать в подгруппе через использование операций `intel_sub_group_block_read`. Особенность работы в подгруппе, элементы вектора распределены по подгруппе и загружаются с шагом `max_sub_group_size`. Таким образом при использовании операции `uint8 col = intel_sub_group_block_read8()` будет загружена колонка матрицы 8x8 с номером элемента подгруппы. При этом надо понимать что элементы подгруппы загрузили все элементы матрицы, а процесс загрузки включает последовательное чтение по строкам и распределение между элементами подгруппы. 


## Базовые алгоритмы линейной алгебры 

Матрица A [M x K], B [K x N]
**L3 Произведение матрицы на матрицу `gemm`**
Матрица A [M x K], B [K x N]
```c
for (uint i = 0; i < M; i++)// по строкам
for (uint j = 0; j < N; j++){
	float sum = 0.f;
	for (uint k = 0; k < K; k++)
		sum += A[i*K + k]*B[k*N+j];
	r[i*N+j] = sum;
}
```

**L2 Произведение матрицы на вектор `gemv`**

M=1
```c
for (int j = 0; j < N; j++){
	float sum = 0.f;
	for (int k = 0; k < K; k++)
		sum += A[j*K + k]*v[k];
	r[j] = sum;
}
```
**L1 Скалярное произведение `dot`**
M=1,N=1
```c
	float sum = 0.f;
	for (int k = 0; k < K; k++)
		sum += A[k]*v[k];
```

Данную операцию так же раскладываем на группы и подгруппы через использование операции _dot product_. Вернее нужно умудриться разложить с использованием операции DPAS
Матричная операция DPAS организована с учетом работы семлера. Т.е. максимальная производительность операции достигается через последовательное использование индексов в подгруппе. 
Единицей загрузки является DW - слово. Слово может быть разложено на один (TF32) два элемента (FP16 или BF16) или на четыре байта (i8 u8) или на восемь 4-х битных (u4) или 2-х битных значений (u2).
XMX - матричный ускоритель работает только с операндами с мантиссой <=10 бит, к которым относятся и TF32(E8M10) и FP16(E5M10), BF16(E8M7), FP8(E5M2), FP4(?). 
XVE - векторный ускоритель работает с типами F32(E8M23)

Примитивом является операция матричного умножения плиток (Tile) фиксированного размера M x K, где Exec_Size=8 - число DW слов, с числом элементов k=8,16,32,64 в зависисисмости от разрядности выбранного типа.
Аппаратно поддерживаются типы TF32, FP16, BF16, FP8, U8, I8, U4, U2
```c
	for (int k = 0; k < K; k++)
		sum += A[m*M+k]*B[k*K+n];
```
Попробуем описпать тоже самое в терминах `DW` - слова, с размером `Exec_Size=8` или `Exec_Size=16`.
```c
	for (int k = 0; k < Exec_Size; k++)
		sum.DW[k] +=dot(A.DW[k],B.DW[k]);
```
где операция `dot` выполняется над элементами упакованными в слово 32 бита. 
* Упаковка 2x16bit выполняется над FP16 и BF16. 
* Упаковка 4x8bit производится над типами u8 s8. 
* Упаковка 8x4bit производится над типами u4 s4 и меньшей разрядности. 
см. операцию `dot_4x8packed_acc_sat()`.

Теперь в описание следует добавить число строк каждой матрицы
```c
	for ( d = 0; d < SD; ++d)// Systolic Depth =8
	for ( i = 0; i < Exec_Size; ++i)
		sum.DW[i] +=dot(B.DW[i],A.DW[k]);
```

DPAS is a matrix multiply-add operation as follows:
$D = C + A \times B$
```c
    k = 0;
    for (r = 0; r < RC; ++r) {// Rows 1,2,4,8
        temp = C.R[r];
        for (d = 0; d < SD; ++d ) {// Systolic Depth
            m = d;  // to select GRF
            for ( i = 0; i < Exec_size; i++ ) // for each channel 
                temp.DW[i] += dot(B.R[m].DW[i], A.DW[k]);
            k++;
        }
        dst.R[r] = temp;
    }
```
**Размеры регистров GRF**. Каждый GRF состоит из SD=8 *DW


FDPAS(f16_f16_matrix_mad_k16) Означает что A имеет размер half16 (int8). B half*n* 
Эмуляция матричной операции 
```c
int B = intel_sub_group_block_read(b_src);// RC=1
int8 A = vload8(mb, a_src);
dst = intel_sub_group_f16_f16_matrix_mad_k16(C, B, A);
```

Параллельно могут считаться пары F16x2 BF16x2 - упакованный формат 4х16bit. Умножение чисел F32 и TF32 может быть представлено форматом F16x2, для этого нужен специальный алгоритм квантизации F16x2 [3]. 

* [3]  <https://arxiv.org/abs/2203.03341> "Recovering single precision accuracy from Tensor Cores while surpassing the FP32 theoretical peak performance"

В языке OpenCL текущей версии 3.0 представлены операции над упакованными векторами 4x8bit такие как `dot_acc_sat` и `dot_4x8packed_uu_sat()`. Для использования данных операций нужна промежуточная квантизация `Q8_1` или `Q8_K`.

Операцию матричного умножения можно разложить на умножение плиток (tile) размером MxN c фиксированным K, представить шаблоном
```c
    long n = get_global_id(0);// число столбцов 
    long m = get_global_id(1);// число строк
    long mb= get_global_id(2);
// опция транспонирования матриц
    long stride_a_m = transa ? lda : 1;
    long stride_a_k = transa ? 1 : lda;
    long stride_b_k = transb ? ldb : 1;
    long stride_b_n = transb ? 1 : ldb;

    ACC_DATA_T acc = 0;
    for (int k = 0; k < K; ++k) {
        long off_a = mb * stride_a_mb + m * stride_a_m + k * stride_a_k;
        long off_b = mb * stride_b_mb + k * stride_b_k + n * stride_b_n;
        acc += TO_ACC(A_TO_REF(a[off_a]) - ATTR_A0)
                * TO_ACC(B_TO_REF(b[off_b]) - ATTR_B0);
    }
```
<https://github.com/oneapi-src/oneDNN/blob/main/src/gpu/intel/ocl/gemm/ref_gemm.cl>

Для матрицы А размером $M \times K$ элементов, матрица B $K \times N$
* `stride_a_m` - размер строки матрицы
* `stride_a_k` - размер элемента матрицы

**Транспонирование матрицы** \
Операция транспонирования не рассматривается как отдельная, она может выполняться в составе операции умножения матриц и сводится к перестанвоке индексов. 

Однако стоит отдельно рассмотреть ситуацию блочных матричных операций. Тогда необходимо выполнить перестановку элементов в блоке 32 х 32 элемента.
```c
    int bx = get_group_id(0); // номер блока по X
    int by = get_group_id(1); // номер блока по Y
    int ix = get_sub_group_local_id();// индекс внутри подгруппы группы
    uint mb = sub_group_broadcat(get_group_id(0), ix);// номер блока в подгруппе
    long offs_a = bx*stride_bx + 
    c[offs_b] = a[offs_a + ix] * b;
```
(О чем пример?)

```c
int8 b = intel_sub_group_block_read8(b);// загружает матрицу 8x8, с шагом max_sub_group_size
    ix =   get_sub_group_local_id();
vstore8(b, ix, dst);

```

**Деквантизация Qn_0**\

Я хочу реализовать умножение квантизованной матрицы A на вектор B. Для этого надо квантизовать вектор `B`.
half2 B = ;// содержит квантизованные значения Q8
B[i*SD+j] = vloda 



`saxpy`: $Ax + y$, A [M x K], B [k]
```c
    k = 0;
    for (r = 0; r < RC; ++r) {// Rows 1,2,4,8
        temp = y.R[r];
        for (d = 0; d < SD; ++d ) {// Systolic Depth
            m = d;  // to select GRF
            for ( i = 0; i < Exec_size; i++ ) // for each channel 
                temp.DW[i] += dot(A.R[m].DW[i], x.DW[k]);
            k++;
        }
        dst.R[r] = temp;
    }
```

**Ускорение загрузки данных, блочные операции ввода/вывода.** EU состоит из 8 потоковых векторных ядер. Потоковые ядра могут работать в режиме подгруппы (WARP группа) и выполнять общий поток инструкций. Чтобы эффективно грузить данные из памяти нужно каждое слово DW грузить на свой потоковый процессор. Т.е при параллельном чтении за такт на конвейер попадает 8/16 слов предназначенных для своего процессора. Таким образом загрузка каждого отдельного потокового процессора будет происходить с шагом i= 16 слов. Эффективным является т.н блочный способ чтения. См. расширение `cl_intel_sub_group_block_io`. Все операции заточены под блочный ввод/вывод. Заявленная производительность векторнго потокового ядра и матричного ядра может быть получена только при таком способе загрузки данных. 

`saxpy`: $Ax + y$, A [M x K], B [k]
```c
    k = 0;
    for (r = 0; r < RC; ++r) {// Rows 1,2,4,8
        temp = y.R[r];
        for (d = 0; d < SD; d++) {// Systolic Depth
            m = d;  // to select GRF
            i = get_sub_group_local_id(); // for each channel 
            temp.DW[i] += dot(A.R[m].DW[i], x.DW[k]);
            k++;
        }
        dst.R[r] = temp;
    }
```
```c
    k = 0;
    for (r = 0; r < RC; ++r) {// Rows 1,2,4,8
        temp = y.R[r];
        for (d = 0; d < SD; d++) {// Systolic Depth
            a = vload(get_sub_group_local_id(), A.R+m*get_max_sub_group_size()); // блочный метод
            temp += dot(a, x.DW[k]);
            k++;
        }
        dst.R[r] = temp;
    }
```
