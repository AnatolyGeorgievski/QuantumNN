# Операции с полиномами

Полиномы мы рассматриваем, как специальный класс для работы с числовыми последовательностями и операторами. При работе с полиномами мы крайне редко применяем операцию вычисления значения. Математический класс полиномы используется для выполнения операций над коэффициентами. Важным приложением является факторизация и представление полиномов в виде разложений по базисным функциям.  Коэффициенты определены на одном из возможных множеств, и класс наследует арифметические операции от базового класса. 

Мы ориентируемся на совместимость с существующими пакетами, по этой причине функции интерфейса представляем в том порядке 

```c
struct _Poly {
    uint32_t n;
    float a[n];
};
```
Полином это вектор с размером $n$, a - коэффициенты полинома $p(z) = a_0 z^{n-1} + ... + a_{n-2} z + a_{n-1}$. 

*{Нумерация индексов вектора коэффициентов от старшей степени к младшей позволяет исключить копирование подстрок при факторизации.}* 

[1]: <https://www.gnu.org/software/gsl/doc/html/poly.html> "GSL - полиномы"

* `_poly_eval` вычисление значения полинома

* `_poly_add` - сложение 
* `_poly_sub` - вычитание
* `_poly_mul` - умножение полиномов
* `_poly_div` - деление полиномов с остатком

## Разложение полинома по разделенным разностям

API [[1]] кроме степенного ряда $\sum_k a_k z^k$ предлагает варианты представления в виде разделенных разностей (divided difference), 
для интерполяции Ньютона. Разложение по полиномам Тейлора. И разложение полинома для интерполяции полиномами Эрмита. 
Я бы предложил рассматривать более широкий класс разложений полиномов. В нашем курсе присутствуют полиномы Чебышева, Эрмита, Бернштейна, Тейлора, Лагранжа и др. В цифровых фильтрах используются полиномы Баттерворта и Бесселя. Необходима факторизация полиномов и разложение рациональных функций на простейшие дроби. 

Базовым классом для полиномов может быть: вещественные числа $\mathbb{R}$ с плавающей точкой `half`, `float`, `double`; комплексные числа $\mathbb{C}$, рациональные числа $\mathbb{Q}$.

В криптографии встречаются полиномы построенные на булевой алгебре, на модульной арифметике, и на арифметике Галуа, в конечном поле. Например полиномы в конечном поле используются в помехоустойчивом кодировании.

Разложение полиномов в разделенные разности $P_n(x) = f(x_0) + \sum\limits_{k=1}^n [x_0,..., x_k] (x-x_0)...(x-x_{k-1})$, где $[x_0,..., x_k]$ - разделенные разности.
* `_poly_dd_init` функция возвращает вектор $dd_k = [x_0,...,x_k]$ на вход принимает вектор точек $x$ и вектор значений $y$.
* `_poly_dd_taylor` представление из разделенных разностей в виде разложения Тейлора вблизи точки $x_p$
* `_poly_dd_hermite`


$H_{2n+1}(x) = f(z_0) + \sum\limits_{k=1}^{2n+1} [z_0,..., z_k] (x-z_0)...(x-z_{k-1})$, где $[z_0,..., z_k]$ - разделенные разности определенные как $z_{2k} = z_{2k+1} = x_k$. 

## Интерполяция полиномами Лагранжа с постоянным шагом

## Двумерная интерполяция Лагранжа

Используется для построения фильтров для цифровой обработки изображений. 

## Разложение полинома по базовым полиномам 

## Билинейная и би-кубическая интерполяция

## Интерполяция B- сплайнами

[^2] <https://www.gnu.org/software/gsl/doc/html/bspline.html> "Базисные сплайны" 

Сплайны бывают разные. Интерполяция кривая проходит через заданные узлы. Сплайн - узловые точки используются для задания кривизны. 

## Разложение полинома по базовым полиномам 
