# Алгоритмы разложения матриц в конечных полях

- [Алгоритмы разложения матриц в конечных полях](#алгоритмы-разложения-матриц-в-конечных-полях)
  - [Элементарные матрицы в конечных полях](#элементарные-матрицы-в-конечных-полях)
  - [Матрицы преобразования Гаусса](#матрицы-преобразования-гаусса)
  - [Ассоциированные матрицы трансформации](#ассоциированные-матрицы-трансформации)
  - [Матрицы трансформации с нормировкой по угловым минорам](#матрицы-трансформации-с-нормировкой-по-угловым-минорам)
  - [Базовые операции преобразования подобия](#базовые-операции-преобразования-подобия)
  - [Разложение матриц в целых числах](#разложение-матриц-в-целых-числах)
    - [Определение минора $A\_{i,j}^{(k)}$](#определение-минора-a_ijk)
    - [Fraction-Free Gauss-Jordan (FFGJ) и Inversion (FFGI)](#fraction-free-gauss-jordan-ffgj-и-inversion-ffgi)
    - [Fraction-Free LU- и LUP-decomposition](#fraction-free-lu--и-lup-decomposition)
    - [Fraction-Free QR- decomposition](#fraction-free-qr--decomposition)
  - [U-LoRA: Унимодулярные низкоранговые проекции](#u-lora-унимодулярные-низкоранговые-проекции)
    - [U-LoRA — Унимодулярная Low-Rank Adaptation](#u-lora--унимодулярная-low-rank-adaptation)
    - [Как это использовать в KV-кэше](#как-это-использовать-в-kv-кэше)
  - [KV-кэш ассоциативная память в целых квантах](#kv-кэш-ассоциативная-память-в-целых-квантах)
  - [Разложение матриц в конечном поле](#разложение-матриц-в-конечном-поле)
  - [Вычисление характеристического полинома](#вычисление-характеристического-полинома)
  - [Rank-revealing decomposition](#rank-revealing-decomposition)
  - [Метод Крылова и циклические подпространства](#метод-крылова-и-циклические-подпространства)
  - [RNS-представление](#rns-представление)

## Элементарные матрицы в конечных полях

**Матрицы Гаусса** (элементарные матрицы Гаусса) — это специальные матрицы, которые получаются из единичной матрицы $I_n$ путём выполнения *ровно одной* элементарной операции над строками. Они используются в методе Гаусса для приведения матрицы к треугольному (или ступенчатому) виду.

Существует *три типа* элементарных матриц:

1. **Перестановочные матрицы** (тип I):  
   $P_{ij}$ — матрица, полученная перестановкой строк $i$ и $j$ в $I_n$.  
   Свойство: $P^{-1} = P^T = P$.

2. **Диагональные масштабирующие матрицы** (тип II):  
   $D_i(\lambda)$ — единичная матрица, у которой в $i$-й позиции диагонали стоит $\lambda \neq 0$.  
   Свойства: $D^{-1} = D_i(\lambda^{-1})$; $det(D(\lambda)) = \lambda$.

3. **Трансвекционные матрицы** (тип III):  
   **Трансвекция** (transvection, shear matrix, матрица сдвига) — это матрица вида  
```math
T_{ij}(c) = I + c \, \mathbf{e}_i \mathbf{e}_j^T \quad (i \neq j),
```
  где:
- $c$ — элемент поля (в GF(p) — число от 0 до p−1),
- $\mathbf{e}_k$ — стандартный базисный вектор-столбец,
- $\mathbf{e}_i \mathbf{e}_j^T$ — матрица, у которой *единственный* элемент в позиции (i,j) равен 1, а всё остальное — 0.

**Ключевые свойства трансвекций**:
- $\det(T_{ij}(c)) = 1$ (поэтому они лежат в группе $\mathrm{SL}(n, F)$).
- Обратная матрица очень простая:  $T_{ij}(c)^{-1} = T_{ij}(-c) = I - c \, \mathbf{e}_i \mathbf{e}_j^T$.
- Над GF(2): $c$ может быть только 0 или 1, $T_{ij}(1)^{-1} = T_{ij}(1)$  

**Как трансвекции используются в методе Гаусса**  
Приведение матрицы $A$ к верхнетреугольному виду происходит путем умножения *слева* на произведение трансвекций (и перестановок):
```math
U = T_k \cdots T_2 T_1 A
```

Каждая $T_m = T_{i j}(-c)$ *обнуляет* один элемент ниже ведущего элемента,  вычитая строку $j$ (с множителем $c$) из строки $i$. 

**Пример над GF(2)** (n=3):

Пусть  
```math
A = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix}.
```

1. Чтобы обнулить $a_{21}=1$:  
   $T_{21}(1) = I + E_{21}$,  
   $T_{21}(1) A$ даёт новую строку 2 = строка 2 $\oplus$ строка 1.

2. Чтобы обнулить $a_{32}=1$:  
   $T_{32}(1)$ и т.д.

В результате получаем верхнетреугольную ($U$), а $\det(A) = (-1)^s \prod U_{ii} \pmod{q}$ (знак от числа перестановок).

**Использование трансвекций в процессе приведения к форме Хессенберга**  
Для *сохранения подобия* в каждой итерации, чтобы получить $P^{-1} A P = H$, мы применяем *пару* трансвекций, слева и справа:

- Слева: $T \cdot A$ (обнуляем элемент ниже поддиагонали).
- Справа: $A \cdot T^{-1}$ (компенсируем изменения в столбцах).

**Почему трансвекции важны в теоретическом плане**
- Они *генерируют* специальную линейную группу $\mathrm{SL}_n(F)$ над полем $F$.
- Любая обратимая матрица может быть представлена, как произведение трансвекций и перестановок.

## Матрицы преобразования Гаусса
```math
F={\begin{pmatrix}1&0&0&\cdots &0\\0&1&0&\cdots &0\\0&a_{32}&1&\cdots &0\\\vdots &\vdots &\vdots &\ddots &\vdots \\0&a_{n2}&0&\cdots &1\end{pmatrix}}
```
[Матрица Фробениуса](https://en.wikipedia.org/wiki/Frobenius_matrix) обратима. Обратная матрица Фробениуса - также является матрицей Фробениуса, и вычисляется просто сменой знака вне-диагональных элементов.
```math
F^{-1}={\begin{pmatrix}1&0&0&\cdots &0\\0&1&0&\cdots &0\\0&-a_{32}&1&\cdots &0\\\vdots &\vdots &\vdots &\ddots &\vdots \\0&-a_{n2}&0&\cdots &1\end{pmatrix}}
```
Тоже справедливо и для транспонированной формы матрицы. Свойство обратимости с изменением знака вне-диагональных элементов можно обобщить на более широкий класс матриц, у которых единицы на главной диагонали, а одна строка или столбец содержат ненулевые элементы. 

Обобщение матриц трансформации (сложные трансвекции):
```math
\begin{aligned}
F &= I + \mathbf{u} \mathbf{e}_k^T, &F^{-1} &= I - \mathbf{u} \mathbf{e}_k^T &\text{ (одна ненулевая строка k)}, u_k=0\\
F &= I + \mathbf{e}_k \mathbf{v}^T, &F^{-1} &= I - \mathbf{e}_k \mathbf{v}^T &\text{ (один ненулевой столбец k)}, v_k = 0\\
\end{aligned}
```
Обязательное требование - на диагонали единицы, это означает что $k$-й элемент вектора $u$ и $v$ в определении должен быть нулевым.

Матрицы Фробениуса называются матрицами преобразования Гаусса (Gauss transformation matrix) используются для представления преобразований Gauss-elimination в матричной форме. Матрица с ненулевыми элементами в столбце соответствует операции вычитания строк ниже ведущего элемента. Транспонированная форма - это линейная комбинация строк - соответствует обратному ходу алгоритма Гаусса-Жордана, операциям выше диагонали.

Не следует путать с рациональной канонической формой Фробениуса и матрицами-компаньонами Фробениуса. 

Матрицы Фробениуса (Gauss Transformations) являются унимодулярными $\mathrm{det}(F) = 1$.

**Связь с трансвекциями и методом Гаусса**  
Матрицы Фробениуса (Gauss transformation) точно соответствуют одной трансвекции или комбинации трансвекций в одной строке или столбце. 

* см. [Golub & van Loan] 3.2.1 Gauss Transformations

Если вы применяете пару $F \cdot A \cdot F^{-1}$, то получаете преобразование подобия: матрица остаётся подобна исходной ($B = F A F^{-1}$), а значит, спектр (собственные значения) сохраняется.

Frobenius matrix (Gauss transformation) — это аналог одной трансвекции Хаусхолдера, но без сохранения нормы, зато с det=1 и простой обратной. Вместе (GCDex + Gauss transformation) позволяют строить унимодулярные разложения, пригодные для конечных полей и fraction-free алгоритмов.

В нашем изложении (и в работе Storjohann):
* Gcdex даёт локальное унимодулярное «вращение» 2×2 блока (аналог Givens rotation в модульной арифметике): приводит 2×1 вектор к виду $(g, 0)^T$ с детерминантом = единица в кольце.
* Gauss transformation даёт исключение подстроки/-столбца.

## Ассоциированные матрицы трансформации

[1] Malashonok, G.I. Computation of the Characteristic Polynomial of an Endomorphism of a Free Module. Journal of Mathematical Sciences 108, 966–976 (2002).  
(https://doi.org/10.1023/A:1013536204334) (https://www.mathnet.ru/links/7f583444587c661dc29153bea979b790/znsl1018.pdf)

[2] Malashonok, G.I. Effective Methods in Algebraic Geometry, ed. by
T. Mora and C. Traverso, Progress in Mathematics 94, Birkhauser, 1991, 289–298 [arXiv:1711.11471](https://arxiv.org/pdf/1711.11471)

> Это перевод одного раздела из работы [1], где дается некоторое обобщение матрицам трансформации Гаусса. Этот класс вводится как базовые операции вращения в конечных полях. Операции трансформации в конечных полях определены через ассоциированные матрицы. 

Введём операцию ассоциации между матрицами вращения.

Пусть $K_{p,q}^L$ обозначает подгруппу группы $GL(p + q, K)$, образованную матрицами

```math
\begin{pmatrix}
a I_p & 0 \\
C & d I_q
\end{pmatrix}, \quad
a, d \in K^*, \quad C \in K^{q \times p},
```

где $I_p$ — единичная матрица порядка $p$, $K^* = K \setminus \{0\}$.

Транспонирование переводит эту подгруппу в изоморфную подгруппу верхнетреугольных матриц. Эта подгруппа будет обозначаться $K_{p,q}^R$.

Инволютивный автоморфизм подгруппы $K_{p,q}^L$:
```math
G = \begin{pmatrix}
a I_p & 0 \\
C & d I_q
\end{pmatrix}
\leadsto
\begin{pmatrix}
d I_p & 0 \\
-C & a I_q
\end{pmatrix}
= \tilde{G},
```

будет называться *ассоциацией*. Эта операция обозначается знаком «тильда».

Инволютивный автоморфизм подгруппы $K_{p,q}^R$:
```math
\begin{pmatrix}
a I_p & B \\
0 & d I_q
\end{pmatrix}
\leadsto
\begin{pmatrix}
d I_p & -B \\
0 & a I_q
\end{pmatrix}
```
определён таким образом, что операции транспонирования и ассоциации коммутируют: $\tilde{G}^T = (\tilde{G})^T$.

Очевидно, что $G\tilde{G} = \tilde{G}G = ad \, I_{p+q}$ — скалярная матрица,
$\tilde{FG} = \tilde{G}\tilde{F}$, $\tilde{\tilde{G}} = G$, $\tilde{I} = I$.

**Инволютивный автоморфизм** группы $GL(2, K)$:
```math
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\leadsto
\begin{pmatrix}
d & -b \\
-c & a
\end{pmatrix}
```
также будет называться _ассоциацией_. Если $G$ — матрица второго порядка, то её ассоциированная матрица совпадает с присоединённой (adjoint): $\tilde{G} = G^{\ast}$.


## Матрицы трансформации с нормировкой по угловым минорам

> Существует несколько вариантов рекуррентных формул для детерминантов, которые позволяют построить fraction-free алгоритмы разложения и их аналоги в конечных полях. Г.Малашонок разбирает четыре варианта алгоритмов, среди которых Dodgson, Bareiss и два собственных варианта оптимизированных по числу операций [[arXiv:1711.11471](https://arxiv.org/pdf/1711.11471)]. В своей работе он вводит матрицы трансформации с масштабными множителями, которые выражаются, как угловые миноры - детерминанты матриц.



Пусть $A = (a_{ij})$ — матрица порядка $n$ над кольцом $K$,
```math
A_{i,j}^{k} =
\begin{pmatrix}
a_{1,1} & \dots & a_{1,k-1} & a_{1,j} \\
\dots & \dots & \dots & \dots \\
a_{k-1,1} & \dots & a_{k-1,k-1} & a_{k-1,j} \\
a_{i,1} & \dots & a_{i,k-1} & a_{i,j}
\end{pmatrix}
```

Обозначим $a_{i,j}^k = \det(A_{i,j}^k)$, $\delta_k = a_{k,k}^k$ - угловой минор.

Пусть $v_k = (a_{k+1,k}^k, \dots, a_{n,k}^k)^T$ — столбец с $n - k$ элементами,
```math
L_k = \begin{pmatrix} \delta_k & 0 \\ v_k & I_{n-k} \end{pmatrix}, \quad
\tilde{L}_k = \begin{pmatrix} 1 & 0 \\ -v_k & \delta_k I_{n-k} \end{pmatrix}, \quad k = 1, 2, \dots, n
```
— матрицы порядка $n - k + 1$, и
```math
L_k = \mathrm{diag}(I_{k-1}, L_k) =
\begin{pmatrix}
1 & \dots & 0 & 0 & \dots & 0 \\
\vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
0 & \dots & \delta_k & 0 & \dots & 0 \\
0 & \dots & a_{k+1,k}^k & 1 & \dots & 0 \\
\dots & \dots & \dots & \dots & \dots & \dots \\
0 & \dots & a_{n,k}^k & 0 & \dots & 1
\end{pmatrix},
```

```math
\tilde{L}_k = \mathrm{diag}(I_{k-1}, \tilde{L}_k) =
\begin{pmatrix}
1 & \cdots & 0 & 0 & \cdots & 0 \\
\vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
0 & \cdots & 1 & 0 & \cdots & 0 \\
0 & \cdots & -a_{k+1,k}^k & \delta_k & \cdots & 0 \\
\vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
0 & \cdots & -a_{n,k}^k & 0 & \cdots & \delta_k
\end{pmatrix}.
```

Такие нормированные трансформации естественным образом обобщают обычные матрицы преобразования Гаусса (Frobenius matrices) и позволяют вести все вычисления в целых числах или в конечных полях без потери точности.

## Базовые операции преобразования подобия

**GCDex** эта операция вводится в дисертации [ARNE STORJOHANN. Algorithms for Matrix Canonical Forms](https://cs.uwaterloo.ca/~astorjoh/diss2up.pdf) и является аналогом вращения Гивенса. Операция используется как базовый механизм для синтеза алгоритмов разложения. Разница между преобразованиями подобия: Вращения и отражения - порождают ортогональные матрицы, которые сохраняют Евклидову норму $U^TU=1$. В конечных полях понятие нормы может вводится через квадратичную форму $x^TQx$. В конечном поле мы используем элементарные преобразования, которые порождают унимодулярные матрицы, $det(U) =\pm 1$. Любую унимодулярную матрицу можно разложить на элементарные матрицы и матрицы транфсормаций Гаусса. 

Определение вращения в конечном поле
```math
\begin{bmatrix} 
s & t \\
u & v
\end{bmatrix} 
\begin{bmatrix} a \\ b \end{bmatrix} =
\begin{bmatrix} g \\ 0 \end{bmatrix} 
```
Необходимо решить систему уравнений
1. $sa + tb = g \to$ методом $g = GCD(a,b)$ находим g такое что $a=g*a_1$, $b = g*b_1$
2. $ua + vb = 0 \to [u=-b_1, v=a_1]$
3. $sv - tu = 1$

В контексте Hermite normal form (HNF), Howell normal form и Smith normal form (SNF) над кольцами (ℤ, ℤ/nℤ, K[x] и др.) почти все современные алгоритмы используют именно такую операцию на столбцах/строках.

Таблица. Пример, варианты решения для GF(2)
|a |b |g | Матрица U | вид
|:---:|:---:|:---:|:---:|:---|
|0 |0 |0 |I=(1001)  $\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ | единичная
|1 |0 |1 |I=(10​01​)  $\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ | единичная
|0 |1 |1 |P=(0110)  $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | перестановка
|1 |1 |1 |E=(1011)  $\begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$ | трансвекция

В поле GF(2) может оказаться выгодным хранить табличные решения по перестановкам. Этот метод известен как метод четырех русских (M4R). Метод может применяться и для умножения матриц и для поиска обратных.

Если $b$ в поле делится на $a$ (т.е. всегда существует $b = r a$) — специальный случай:
* берём $s = 1$, $t = 0$, $g = a$, $u = -r$, $v = 1$;
* Тогда $U = \begin{pmatrix} 1 & 0 \\ -r & 1 \end{pmatrix}$ (элементарная матрица- трансвекция) -- _вертикальный_ сдвиг.

Если $a$ в поле делится на $b$ (т.е. всегда существует $a = q b$) — специальный случай:
* Тогда $U = \begin{pmatrix} 0 & 1 \\ 1 & -q \end{pmatrix}$

Две другие формы
* Тогда $L = \begin{pmatrix} -r & 1 \\ 1 & 0 \end{pmatrix}$ -- горизонтальный сдвиг.
* Тогда $L = \begin{bmatrix} 1 & -q \\ 0 & 1 \end{bmatrix} \begin{bmatrix} a \\ b\end{bmatrix} = \begin{bmatrix} 0 \\ g \end{bmatrix}$ -- горизонтальный сдвиг, $g=b$.


## Разложение матриц в целых числах

Основная идея - переход от вещественных чисел с плавающей точкой к матрицам в рациональных числах. Матрицы можно представить в целых числах с общим делителем. Эта концепция порождает ряд методов, которые помимо эффективности, не теряют точность вычисления. Благодаря свойству [Sylvester's determinant identity](https://vmath.ru/vf5/dets/sylvester) [1] удается выстроить алгоритм таким образом, чтобы деление производилось без остатка и всякое выражение для преобразования матрицы строилось с учетом тождества.



Построение алгоритма Gauss Elimination с использованием матриц трансформации:

1. $d \leftarrow 1$
2. $\text{for } k = 1 \text{ to } n$
3. $\quad e_k \leftarrow  A[k,k]$
4. $\quad E_k \leftarrow  e_k I_n; ~E_k[*,k] = -A[*,k]; E_k[k,k] = d;$ -- матрица трансформации $L_k$
5. $\quad A \leftarrow  {1 \over d}E_k A$
6. $\quad d \leftarrow e_k$

В цикле формируется матрица трансформации Гаусса с ведущими минорами (leading minors) на диагонали.
Ключевое свойства подобных операций - возможность разложения матрицы на трансформации Гаусса в форме 
```math
U = \left({1 \over d_{k-1}}\right)E_k\cdots \left({1 \over d_1}\right)E_2\left({1 \over d_0}\right)E_1 A~.
```
При этом каждое деление выполняется без остатка. Кроме того операция $1/d_{k} E_{k+1} E_k$ позволяет комбинировать матрицы трансформации в контексте fraction-free операций. В общем случае трансформацию можно рассматривать, как комбинацию трансвекций - вычитаний строк ($E_k$) и перестановок ($P_k$) строк. Группировка матриц трансформаций позволяет выполнять операции в блочном виде. 

Представляем ряд методов, которые могут использоваться в целочисленных вычислениях без перехода в множество рациональных чисел (fraction-free) и обеспечивают переход к разложениям методов линейной алгебры в конечных полях. 

**Fraction-Free Gauss Elimination**  
  Приведение матрицы к верхне-треугольному виду U

**Fraction-Free Gauss-Jordan**  
  Решение уравнения Ax=b

**Fraction-Free Gauss-Jordan Inversion**  
  Обращение матрицы

**Fraction-Free LU factorization**  
  LU-разложение - Основной метод представления матриц, удобный для вычисления и решения уравнений и поиска обратной матриц. Уравнения $LUx = b$ решаются с использованием двух шагов: левой и правой подстановки Ly = b и Ux = y
* **Fraction-Free back substitution**  
* **Fraction-Free forward substitution**  

**Fraction-Free complete (LDUP) factorization**  
  Описание алгоритмов разложения $PA = LDU$ и backward/forward substitution см. [5].

**Roundoff-Error-Free LDL Cholesky decomposition**  
  Любую квадратную матрицу можно разложить на симметричную и антисимметричную. Метод расчета должен исключать ошибки округления. 

**Fraction-Free LDL- decomposition**  
Симметричные и анти-симметричные матрицы в целых числах можно раскладывать с точным делением.

**Fraction-Free QDR- decomposition**  
Существует аналог процесса разложения Грамма-Шмидта, который дает аналогичное разложение в целых числах.

[1] E.H. Bareiss, Computational solutions of matrix problems over an integral domain, 
    IMA J. Appl. Math. 10 (1) (1972) 68–104, https://doi.org/10.1093/imamat/10.1.68. 

[1a] E.H. Bareiss, Sylvester’s identity and multistep integer-preserving Gaussian elimination, Math. Comp. 22 (1968) 565–578.

[2] George C. Nakos, Peter R. Turner, and Robert M. Williams. 1997.
    Fraction-free algorithms for linear and polynomial equations. 
    SIGSAM Bull. 31, 3 (Sept. 1997), 11–19. https://doi.org/10.1145/271130.271133

[3] Middeke, J.; Jeffrey, D.J.; Koutschan, C. (2020), "
    Common Factors in Fraction-Free Matrix Decompositions", 
    Mathematics in Computer Science, 15 (4): 589–608, [arXiv:2005.12380](https://arxiv.org/pdf/2005.12380), doi: https://doi.org/10.1007/s11786-020-00495-9

[4] Adolfo R. Escobedo and Erick Moreno-Centeno. 2015. Roundoff-Error-Free Algorithms for Solving Linear Systems via Cholesky and LU Factorizations.  
    INFORMS J. on Computing 27, 4 (November 2015), 677–689. https://doi.org/10.1287/ijoc.2015.0653

[5] David Dureisseix. Generalized fraction-free LU factorization for singular systems with kernel extraction.
Linear Algebra and its Applications, 2012, 436 (1), pp.27-40. (https://doi.org/10.1016/j.laa.2011.06.013). [hal-00678543](https://hal.science/hal-00678543/document)

[6] W. Zhou, D.J. Jeffrey, Fraction-free matrix factors: new forms for LU and QR factors, Front. Comput. Sci. China 2 (1) (2008) 1–13. doi: https://doi.org/10.1007/s11704-008-0005-z (https://www.uwo.ca/apmaths/faculty/jeffrey/pdfs/FFLUQR.pdf)

https://dl.acm.org/doi/pdf/10.1145/1838599.1838602

[[arXiv:2504.20305](https://arxiv.org/pdf/2504.20305)] Fast LDL factorization for dense and sparse symmetric matrices over an arbitrary field

[8] U. Erlingsson, E. Kaltofen, and D. Musser. Generic Gram—Schmidt orthogonalization by exact division.  
In International Symposium on Symbolic and Algebraic
Computation, pages 275–282. ACM press, 1996.

### Определение минора $A_{i,j}^{(k)}$

Минор $A_{i,j}^{(k)}$ для ($i > k ,  j > k$) является определителем подматрицы $A$ порядка $k+1$:
```math
A_{i,j}^{(k)} = \det \begin{pmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1,k} & a_{1,j} \\
a_{2,1} & a_{2,2} & \cdots & a_{2,k} & a_{2,j} \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
a_{k,1} & a_{k,2} & \cdots & a_{k,k} & a_{k,j} \\
a_{i,1} & a_{i,2} & \cdots & a_{i,k} & a_{i,j}
\end{pmatrix}
```
где $a_{i,j}$ — элементы матрицы $A$.

- Он соответствует определителю подматрицы, образованной первыми $k$ строками и столбцами и  дополненой $i$-й строкой и $j$-м столбцом.

**Базовые случаи**:
- $A_{i,j}^{(1)} = a_{1,1} a_{i,j} - a_{1,j}a_{i,1}$
- $A_{i,j}^{(0)} = a_{i,j}$ (элементы самой матрицы).

**Реккурентная формула** (Bareiss):
```math
A_{i,j}^{(k)} = \frac{ A_{k,k}^{(k-1)} \cdot A_{i,j}^{(k-1)} - A_{k,j}^{(k-1)} \cdot A_{i,k}^{(k-1)} }{ A_{k-1,k-1}^{(k-2)} }
```
дает деление без остатка, для $k \geq 2$, $i > k , j > k$.

Эта рекуррентная формула лежит в основе *fraction-free исключения* и позволяет выразить разложение  $L D^{-1} U$ в целых числах.

В результате разложения матрицы A на миноры получаем упакованную структуру $[L \backslash U]$ из миноров


<!-- 
```math
PA = \begin{bmatrix}
a_{1,1} & a_{1,2} & a_{1,3} & \cdots & a_{1,n} & \cdots & a_{1,m} \\
a_{2,1} & a_{2,2} & a_{2,3} & \cdots & a_{2,n} & \cdots & a_{2,m} \\
a_{3,1} & a_{3,2} & a_{3,3} & \cdots & a_{3,n} & \cdots & a_{3,m} \\
\vdots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
a_{n,1} & a_{n,2} & a_{n,3} & \cdots & a_{n,n} & \cdots & a_{n,m}
\end{bmatrix}
```
 -->
```math
[L \backslash U] = \begin{bmatrix}
A_{1,1}^{(0)} & A_{1,2}^{(0)} & A_{1,3}^{(0)} & \cdots & A_{1,n}^{(0)} & \cdots & A_{1,m}^{(0)} \\
A_{2,1}^{(0)} & A_{2,2}^{(1)} & A_{2,3}^{(1)} & \cdots & A_{2,n}^{(1)} & \cdots & A_{2,m}^{(1)} \\
A_{3,1}^{(0)} & A_{3,2}^{(1)} & A_{3,3}^{(2)} & \cdots & A_{3,n}^{(2)} & \cdots & A_{3,m}^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
A_{n,1}^{(0)} & A_{n,2}^{(1)} & A_{n,3}^{(2)} & \cdots & A_{n,n} & \cdots & A_{n,m}
\end{bmatrix}
```

Элементы разложения матрицы $A = LD^{-1}U$ выражаются напрямую через миноры [6]:
  - $L_{i,k} = A_{i,k}^{(k-1)}$ (для $i > k$)
  - $U_{k,j} = A_{k,j}^{(k-1)}$ (для $j > k$)
  - $D_{k,k} = A_{k,k}^{(k-1)} A_{k-1,k-1}^{(k-2)}$ (для $k \geq 1$)

Матрица миноров [L\U] обратима по шагам (backward step reversible), т.е. используя обратный шаг рекуррентной формулы можно восстановить исходную матрицу A. 

При вычислении обратных матриц и используется обратный ход
```math
\delta_{i,j}^{(k)} = \frac{ A_{k,k}^{(k)} \cdot \delta_{i,j}^{(k-1)} - A_{i,k}^{(k)} \cdot \delta_{k,j}^{(k-1)} }{ A_{k-1,k-1}^{(k-1)} }
```

### Fraction-Free Gauss-Jordan (FFGJ) и Inversion (FFGI)

В дополнение к одномерному fraction-free Gauss elimination (Bareiss) в библиотеке реализован *FF Gauss-Jordan* и *инверсия* по алгоритму 2 из статьи [2].

**ff_gauss_jordan(a, b, n, lda)**  
- Приводит квадратную матрицу `a` к **диагональному** виду.  
- Все промежуточные деления точны (exact division) благодаря Sylvester’s identity.  
- На выходе `b` содержит масштабированное решение системы `A x = b`.  
- Возвращаемое значение — последний ведущий главный минор (обычно `±det(A)`).  
- Соответствует **Algorithm 2 (FFGJ)** из Nakos et al. (1997).

**ff_gauss_inversion(a, b, n, lda)**  
- Решает матричное уравнение `A X = I` (вычисляет обратную матрицу).  
- На входе `b` не используется — внутри автоматически заполняется единичной матрицей.  
- На выходе `b` содержит **масштабированную** обратную матрицу: `A·b = d·I`, где `d = det(A)`.  
- Очень удобен для проверки обратимости и вычисления `det(A)` в целых числах.

Обе функции работают **только при ненулевых ведущих главных минорах**. При необходимости можно добавить pivoting (как в FFGE).  

Все операции сохраняют **целочисленность** и позволяют работать с очень большими матрицами без перехода в поле рациональных чисел.

### Fraction-Free LU- и LUP-decomposition

{дать описание}

### Fraction-Free QR- decomposition

{дать описание}

## U-LoRA: Унимодулярные низкоранговые проекции

*(интеграция концепции fraction-free алгоритмов и обобщённых блочных трансформаций Гаусса)*

Операцию LoRA можно представить как outer-product в форме сложной трансформации Гаусса, произведения матриц трансформации Гаусса. При этом мы накладываем условие чтобы результат удовлетворял требованиям унимодулярности (или ортогональности в $R^n$). Алгоритм построения U-LoRA должен строится в целых числах.

Классическая LoRA имеет вид:
```math
\Delta W = B A, \quad \operatorname{rank}(A) = \operatorname{rank}(B) = r \ll d, \quad W' = W + \Delta W.
```

**Проекция в форме $W(I + uv^T)$**  
При выборе $v^Tu = 0$ матрица $Q = I + uv^T$ является унимодулярной ($\det Q = 1$). Тогда $W' = WQ$ представляет собой правое умножение на унимодулярную матрицу, что является частным случаем обобщённой блочной трансформации Гаусса. Такая форма естественным образом интегрируется в fraction-free алгоритмы и позволяет строить U-LoRA-проекции полностью в целочисленном домене.

### U-LoRA — Унимодулярная Low-Rank Adaptation

**U-LoRA** — это низкоранговая унимодулярная адаптация матрицы, которая может быть получена, как **композиция обобщённых матриц трансформации Гаусса** (Gauss transformation matrices / Frobenius matrices) двух типов:

1. **Левое умножение** (операция над строками):  
   Матрица с **ненулевыми элементами только в одном столбце** (единицы на диагонали).  
```math
   F_L = I + \mathbf{u} \mathbf{e}_k^T, \quad u_k = 0
```
   (ненулевые элементы находятся в **k-м столбце**).

2. **Правое умножение** (операция над столбцами):  
   Матрица с **ненулевыми элементами только в одной строке** (единицы на диагонали).  
```math
   F_R = I + \mathbf{e}_k \mathbf{v}^T, \quad v_k = 0
```
   (ненулевые элементы находятся в **k-й строке**).

Обе матрицы *унимодулярны*: $\det(F_L) = \det(F_R) = 1$.

### Как это использовать в KV-кэше

- Query и каждый ключ проходят свою **U-LoRA-проекцию** (композицию \(F_L\) и \(F_R\)).
- Similarity вычисляется уже после этих унимодулярных преобразований.
- Это позволяет делать **персональные вращения** для разных кластеров/persona без нарушения обратимости и точности.


## KV-кэш ассоциативная память в целых квантах

**Преимущества U-LoRA в контексте RNS-KV-кэша**

- **fraction-free** — нет перехода в рациональные числа.
- **Reversible** (существует целочисленная обратная матрица).
- **Sponge-construction** (компрессия тензоров на операциях NTT-friendly).
- **MoE-style routing** по дереву: каждая ветвь/кластер получает свою лёгкую U-LoRA-персону поверх *общего* KV-кэша (как в архитектуре Grok multi-agent).
- Низкий overhead: ранжирование и топ-k выполняются только в выбранных кластерах.

**Алгоритм построения U-LoRA (псевдокод)**

{описание}

## Разложение матриц в конечном поле

Давайте чётко систематизируем понятие *вращения* в контексте конечных полей и унимодулярных преобразований. Понятие вращения аналогично классическому вращению Гивенса.

**1. Классическое «вращение» как отправная точка**
- **Givens rotation** (над ℝ или ℂ): ортогональная матрица 2×2, сохраняющая евклидову норму, обнуляет один элемент вектора.

```math
U = \begin{pmatrix} с & -s \\ s & c \end{pmatrix}, \quad 
U \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} g \\ 0 \end{pmatrix}~.
```
Свойства: $\det(U) = \pm 1$, выбор $g = \sqrt{a^2 + b^2}$, $c = a/r$, $s = b/r$.

[[2406.02750](https://arxiv.org/pdf/2406.02750)] A Fast Compensated Algorithm for Computing Givens Rotations

- **Householder reflection**: более мощный аналог (отражение относительно гиперплоскости), тоже сохраняет норму.

В конечных полях (или PIR) понятие евклидовой нормы теряет смысл или заменяется квадратичной формой, поэтому мы переходим к *унимодулярности* (det U — единица кольца/поля).

**2. «Вращение» в конечном поле (унимодулярное 2×2 преобразование)**

Это **Gcdex** (по Storjohann):
```math
U = \begin{pmatrix} s & t \\ u & v \end{pmatrix}, \quad 
U \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} g \\ 0 \end{pmatrix}, \quad sv - tu = 1 \ (или\ единица).
```

- Здесь g = gcd(a, b) (с точностью до единицы поля).
- U порождается комбинацией элементарных операций (трансвекций + перестановок).
- Это *верхнее вращение* (pivot сверху, ноль снизу).

- Специальный случай: $b|a$ then $s = v = 1, t=0$, $U = \begin{pmatrix} 1 & 0 \\ -b/a & 1 \end{pmatrix}$

**3. Два типа «вращения» на базе трансвекций**

Мы вводим понятие верхнего и нижнего вращения, оба основаны на трансвекциях (shear matrices) и унимодулярности:

**Тип A — Верхнее вращение (Upper-Gcdex)**  
Цель: привести вектор $[a \ b]$ к виду $[g \ 0]^T$ (pivot в первой позиции).
```math
U \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} g \\ 0 \end{pmatrix}.
```

**Тип B — Нижнее вращение (Lower-Gcdex)**  
Цель: привести вектор $[a \ b]$ к виду $[0 \ g]^T$ (pivot во второй позиции, ноль сверху). При последовательном применении это элементарное преобразование формирует нули сверху.
```math
L \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} 0 \\ g \end{pmatrix}
```

Эти двe трансформации вместе позволяют строить companion-блоки (1 на субдиагонали + коэффициенты в последнем столбце), необходимые для *рациональной канонической формы* (Frobenius canonical form). трансформация Upper-вращения позволяет привести к ступенчатой форме (row-echelon) или форме Хессенберга, операция Lower-вращения позволяет из формы хессенберга получить форму матрицы-компаньона. 

- Специальный случай: $a|b$ then $s = v = 1, u=0$, $L = \begin{pmatrix} 1 & -a/b \\ 0 & 1 \end{pmatrix}$

Предлагается специальная форма  $L = \begin{pmatrix} r & -1 \\ 1 & 0 \end{pmatrix}$, $U = \begin{pmatrix} 0 & 1 \\ -1 & q \end{pmatrix}$, где $r = ba^{-1}$, $q = qb^{-1}$. Варианты соответствуют условиям делимости $a|b$ или $b|a$, и $\det(U)=\pm 1$.

https://en.wikipedia.org/wiki/Kalman_decomposition

## Вычисление характеристического полинома 

[1] Clément Pernet and Arne Storjohann. 2007. Faster algorithms for the characteristic polynomial. In Proceedings of the 2007 international symposium on Symbolic and algebraic computation (ISSAC '07). Association for Computing Machinery, New York, NY, USA, 307–314. https://doi.org/10.1145/1277548.1277590
(https://cs.uwaterloo.ca/~astorjoh/issac077f.pdf)

[2] W. Keller-Gehrig. 1985. Fast algorithms for the characteristic polynomial. Theoretical computer science 36 (1985), 309–317.  
(https://doi.org/10.1016/0304-3975(85)90049-0)

[3] Clément Pernet Computing the Kalman form, (2006)
[arXiv:cs/0510014](https://arxiv.org/pdf/cs/0510014)

[4] Jean-Guillaume Dumas, Clément Pernet, and Zhendong Wan. 2005. Efficient computation of the characteristic polynomial.  
In Proceedings of the 2005 international symposium on Symbolic and algebraic computation (ISSAC '05). Association for Computing Machinery, New York, NY, USA, 140–147. 
(https://doi.org/10.1145/1073884.1073905); [arXiv:cs/0501074](https://arxiv.org/pdf/cs/0501074)

[5] Vincent Neiger, Clément Pernet, and Gilles Villard. 2024. Computing Krylov iterates in the time of matrix multiplication. 
In International Symposium on Symbolic and Algebraic Computation (ISSAC '24), July 16--19, 2024, Raleigh, NC, USA. ACM, New York, NY, USA 10 Pages. 
(https://doi.org/10.1145/3666000.3669715)

## Rank-revealing decomposition

Задача: Krylov nullspace,  
LSP, CUP, PLE, PLUQ, LEU, Bruhat, PLDUQ ...

[6] Claude-Pierre Jeannerod, Clément Pernet, Arne Storjohann, Rank-profile revealing Gaussian elimination.  
(https://doi.org/10.48550/arXiv.1112.5717) [arXiv:1112.5717](https://arxiv.org/pdf/1112.5717) 
(https://cs.uwaterloo.ca/~astorjoh/ple.pdf) -- LSP и PLE

[7] Fast Computation of the Rank Profile Matrix and the Generalized Bruhat Decomposition
[arXiv:1601.01798](https://arxiv.org/pdf/1601.01798)

[8] Gennadi Malaschonok. Fast generalized Bruhat decomposition. 
[arXiv:1702.07242]() https://doi.org/10.48550/arXiv.1702.07242

[9] Jean-Guillaume Dumas, Clément Pernet and Ziad Sultan. Simultaneous computation of the row and column rank profiles
[arXiv:1301.4438](https://arxiv.org/pdf/1301.4438)

[10] Rank-profile revealing Gaussian elimination and the CUP matrix decomposition.  
[arXiv:1112.5717](https://arxiv.org/pdf/1112.5717)

[11] Martin R. Albrecht, Gregory V. Bard, Clément Pernet. Efficient Dense Gaussian Elimination over the Finite Field with Two Elements.  
[arXiv:1111.6549](https://arxiv.org/pdf/1111.6549) [github:M4RI](https://github.com/malb/m4ri)

[12] Martin R. Albrecht, Clément Pernet. Efficient Decomposition of Dense Matrices over GF(2)  
[arXiv:1006.1744](https://arxiv.org/pdf/1006.1744)

[1702.07243](https://arxiv.org/pdf/1702.07243) 

## Метод Крылова и циклические подпространства



## RNS-представление 