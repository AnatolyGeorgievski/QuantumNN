_Георгиевский А.М._

В обзоре собрана информация по визуальным моделям нейросетей. Целью работы является сравнение изображений по семантическим признакам. А также совмещение задачи классификации и описания изображений с задачей отслеживания движения объектов. Рассмотрены базовые методы сравнения в пространстве семантических признаков и комбинация методов сравнения для использования в процессе обучения.

Практическое использование методов сравнения изображений в пространстве семантических признаков может быть использовано для поиска и сравнения изображений в базе данных. В наших тестах методы используются для анализа видео контента для поиска границ смены сцены, для сравнения опорных кадров и для классификации. Также мы планируем реализовать поиск по семантическим признакам опорного кадра в видео потоке. 


**LLM** - Large Language Models, языковые модели с архитектурой Transformer.\
**MLLM** - Multimodal Large Language Models\
**VLM и VLA** - Visual Language and Actions модели\
**ViT** - Vision Transformer архитектура\
**CNN** - Convolutional Neural Networks\
**RNN** - Recurrent Neural Networks: GRU, LSTM, STU\
**PINN** - Physic-informed Neural Networks\
**U-Net** - архитектура сверточнной сети для сегментации изображений\
**KSN** - Kolmogorov-Spline Network архитектура\
**KAN** - Kolmogorov-Arnold Network архитектура


## VLM и VLA - Visual Language and Actions модели

VLA модели применяются в задачах для управления роботами ассистентами и в задачах управления беспилотным летательным аппаратом. Метод создания собственной модели может быть основан на разных сетях, прежде всего это сети CNN для детектирования объектов и для сегментации изображения. С использованием CNN сетей возможно обучить сети с архитектурой ViT для более эффективного решения задачи. Архитектура сети может быть основана на слое встраивания каналов изображения. Встраивание наиболее эффективно решается на архитектуре KSN (сплайновых сетях Колмогорова). Построение сплайновых сетей - современное направление, которое может заменить MLP на KAN. Почему?! Потому что есть прямой путь между проектированием цифровых фильтров и представлением в виде сверточной сети. 

Технология VLA основана на применении двух (или более) визуальных энкодеров для выделения движения, для сегментации и для обработки изображений. Обработка команд выполняется языковой моделью (LLM). Выход языковой модели содержит декодер команд управления приводами. 
Ниже привожу подборку статей по визуальным моделям для встраивания в конечное устройство. Среди моделей выделяются SmolVL2 и InternVL содержащие компактный визуальный энкодер. 

В обзоре уделяется внимание методам сравнения двух изображений в семантическом пространстве. Тут можно выделить методы сравнения растровых изображений, основанные на каскадном применении сверточных сетей CNN (т.н пирамиды), результатом является построение масок изображений, Feature map, сегментация. Сравнение фрагментов изображений должно выполняться с использованием масок изображений. Маски изображений выделяются на основе motion estimation, HOG и различных контрастных фильтров. Ряд изображений полученных может использоваться совместно как дополнительные каналы изображения. 

* [[1911.05722](https://arxiv.org/pdf/1911.05722)] Momentum Contrast for Unsupervised Visual Representation Learning
* [[2103.00020](https://arxiv.org/pdf/2103.00020)] Learning Transferable Visual Models From Natural Language Supervision
* [[2304.07193](https://arxiv.org/pdf/2304.07193)] DINOv2: Learning Robust Visual Features without Supervision
* [[2502.19645](https://arxiv.org/pdf/2502.19645)] Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success

Визуальные сети и энкодеры:
* [[2406.09246](https://arxiv.org/pdf/2406.09246)] OpenVLA: An Open-Source Vision-Language-Action Model
* [[2409.12191](https://arxiv.org/pdf/2409.12191)] Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution
* [[2504.05299](https://arxiv.org/pdf/2504.05299)] SmolVLM: Redefining small and efficient multimodal models
* [[2504.07491](https://arxiv.org/pdf/2504.07491)] Kimi-VL Technical Report
* [[2504.10479](https://arxiv.org/pdf/2504.10479)] InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models
* [[2505.04601](https://arxiv.org/pdf/2505.04601)] OpenVision : A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning
* [[2505.22159](https://arxiv.org/pdf/2505.22159)] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
* [[2506.01844](https://arxiv.org/pdf/2506.01844)] SmolVLA: A vision-language-action model for affordable and efficient robotics
* [[2506.07900](https://arxiv.org/pdf/2506.07900)] MiniCPM4: Ultra-Efficient LLMs on End Device


* (https://ucsc-vlaa.github.io/OpenVision)
* (https://huggingface.co/openvla/openvla-7b)
* (https://learnopencv.com/smolvla-lerobot-vision-language-action-model/)

Компьютерное зрение представлено библиотекой OpenCV, которая работает с бэкендом FFmpeg и GStreamer для ввода изображений и потокового видео. OpenCV включает возможности выделения ключевых точек (feature detection), выделения контуров, вычисления гомографии, вычисления глубины изображения. Документация по OpenCV содержит множество примеров использования библиотеки.
Мы выделяем несколько направлений для тестирования и построения алгоритмов ориентации в пространстве. Предполагается, что методы компьютерного зрения будут использоваться для построения алгоритмов ориентации в пространстве совместно с визуальными моделями нейросетей. 

С использованием методов OpenCV и CNN нейросетей решаются следующие задачи:

1. [Ориентированные рамки](#ориентированные-рамки) - позволяют выделить объекты (классы) на изображении с учетом их ориентации. Классы могут быть фиксированные или произвольные в зависимости от типа модели нейросети. 
2. [Принципы сравнения изображений](#принципы-сравнения-изображений) и сравнение изображений в семантическом пространстве.
3. Motion estimation Выделение масок на основе оценки движения. 
4. [Semantic Segmentation](#semantic-segmentation) Сегментация изображений на основе семантических признаков.
5. [Visual Embeddings](#visual-embeddings)

## Ориентированные рамки

    *OBB* - oriented bounding box, ориентированная рамка границ изображения
    *GBB* - Gaussian bounding box, гауссова рамка границ изображения

Маски вероятностей для GBB представляют собой гауссово распределение, описывающее вероятность нахождения объекта в определенной области изображения.

* [[2007.09584](https://arxiv.org/pdf/2007.09584)] PIoU Loss: Towards Accurate Oriented
Object Detection in Complex Environments
* [[2106.06072](https://arxiv.org/pdf/2106.06072)] Gaussian Bounding Boxes and Probabilistic
Intersection-over-Union for Object Detection

* (https://github.com/zhen6618/RotaYolo)

* (https://arxiv.org/pdf/2101.11952) Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss

{Маски вероятностей, как получить из изображения? (L)GBB. Дать определение и описание}
{Причем тут регрессия? Дать определение и пояснить на примере}

Регрессия используется для предсказания параметров гауссовой рамки $(\mu_x, \mu_y, \sigma_x, \sigma_y, \theta)$ на основе входного изображения. Например, нейросеть (обычно с архитектурой CNN или ViT) предсказывает координаты центра, размеры и угол поворота рамки. Это отличается от классификации, где предсказывается дискретный класс объекта. Пример: в задаче выделения объектов регрессия применяется для точного определения координат и ориентации рамки вокруг объекта, как описано в [2007.09584]. 

## Гомография изображений и отслеживание объектов

Гомография может быть построена по набору ключевых точек на двух изображениях. Ключевые точки могут быть выделены с помощью методов OpenCV: SURF, ORB, AKAZE. 

SURF, несмотря на высокую эффективность, имеет патентные ограничения, поэтому в современных приложениях чаще используются ORB или AKAZE, которые являются открытыми и оптимизированы для реального времени. 

Совмещение ключевых точек изображений выполняется методом RANSAC. Гомография может быть использована для вычисления глубины изображения. 

* (https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
* (https://docs.opencv.org/4.x/dc/d16/tutorial_akaze_tracking.html)

## Feature Extraction
* [[1408.5093](https://arxiv.org/pdf/1408.5093)] Caffe: Convolutional Architecture for Fast Feature Embedding

* [[1504.06066](https://arxiv.org/pdf/1504.06066)] Object Detection Networks on Convolutional Feature Maps
* [[1505.04597](https://arxiv.org/pdf/1505.04597)] U-Net: Convolutional Networks for Biomedical Image Segmentation
>*U-Net Network Architecture*\
В работе разобрана структура сети (fig.1). It consists of a contracting
path (left side) and an expansive path (right side). The contracting path follows
the typical architecture of a convolutional network. It consists of the repeated
application of two 3x3 convolutions (unpadded convolutions), each followed by
a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
for downsampling. At each downsampling step we double the number of feature
channels. Every step in the expansive path consists of an upsampling of the
feature map followed by a 2x2 convolution (“up-convolution”) that halves the
number of feature channels, a concatenation with the correspondingly cropped
feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in
every convolution. At the final layer a 1x1 convolution is used to map each 64-
component feature vector to the desired number of classes. 

* [[1504.08083](https://arxiv.org/pdf/1504.08083)] Fast R-CNN
* [[1506.01497](https://arxiv.org/pdf/1506.01497)] Faster R-CNN: Towards Real-Time Object
Detection with Region Proposal Networks
* [[1612.03144](https://arxiv.org/pdf/1612.03144)] Feature Pyramid Networks for Object Detection
* [[1612.06370](https://arxiv.org/pdf/1612.06370)] Learning Features by Watching Objects Move
* [[1703.06870](https://arxiv.org/pdf/1703.06870)] Mask R-CNN\
-- мне нравится результат - это комбинация двух сверточных сетей: для детектирования и для выделения маски
* [[2112.09133](https://arxiv.org/pdf/2112.09133)] Masked Feature Prediction for Self-Supervised Visual Pre-Training\
 -- Histograms of Oriented Gradients (HOG)
* [[2408.00714](https://arxiv.org/pdf/2408.00714)] SAM 2: Segment Anything in Images and Videos

* (https://learnopencv.com/histogram-of-oriented-gradients/)

HOG: Метод использует простые свертки (производные) $[-1,0,1]$ и $[-1,0,1]^{\mathsf{T}}$ для вычисления градиентов ${g_x, g_y}$. Представление маски может быть в виде цветовой схемы HSV (Magnitude + angle).

Представленные методы имеют сверточную структуру сети типа U-Net или FPN(feature pyramid network). В структуре сети, которая выглядит как пирамида, в которой чередуются слои `Conv 3x3`, `MaxPool 2x2`, И пирамида обратного преобразования, в которой используется `UpSample` и `Conv 2x2`. Подобная структура может быть получена из строго математических соображений, путем синтеза фильтров. 

Принципы синтеза фильтров двумерных фильтров рассматриваются в [курсе DSP](#) - цифровой обработки изображений. Особенность нашего подхода - разложение фильтров на секции второго порядка и на базисные полиномы.

Путь для освоения и применения в своих разработках, в моем видении, лежит через разбор формата ONNX в результате экспорта моделей YOLO в ONNX. Этот метод используется в Edge- устройствах на базе RockChip [сетях RKNN](https://docs.ultralytics.com/ru/integrations/rockchip-rknn/). 

Модели YOLO могут быть дообучены и встроены в программу с использованием RKNN Toolkit.
```sh
yolo export model=yolo11n.pt format=rknn name=rk3588 #
```

Проектирование новых архитектур для feature extraction я бы начинал с иных позиций. 
1. Свертка может быть представлена HOG - производными и В-сплайнами. Двуменрное преобразование раскладывается на функции одного переменного (bi-сплайны) по Колмогорову. Простейший пример [билинейное и бикубическое преобразование](bicubic.c). 
2. Свертки в пирамиде могут быть получены аналитически и дообучены. 
3. Исследования должны быть направлены на представление в архитектуре KSN/KAN и совмещать в себе распознавание физики, в простейшей архитектуре подобной GRU или State-Space модели (SSM). Переход от сверточных моделей к State-Space модели выполняется через использование time-mix оператора. 

## Принципы сравнения изображений

The goal of text embedding models is to convert variable length text into a fixed vector, encoding the text semantics into a multidimensional vector in such a way that semantically close texts are close in the vector space, while dissimilar texts have a low similarity.

* [[2502.20204](https://arxiv.org/pdf/2502.20204)] Granite Embedding Models, IBM team, 2025
* [[2506.18902](https://arxiv.org/pdf/2506.18902)] jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval

Пример. описание родственных моделей LLM для задачи генерации текстов и для задачи Embedding и Reranking. Отправная точка - использование векторов Embedding модели для поиска и сравнения изображений в базе данных или на потоке. На данный момент открыты модели для семантического сравнения текстовых документов.

* [[2505.09388](https://arxiv.org/pdf/2505.09388)] Qwen3 Technical Report
* [[2506.05176](https://arxiv.org/pdf/2506.05176)] Qwen3 Embedding: Advancing Text Embedding and
Reranking Through Foundation Models

Базовая идея построения LLM-сетей а архитектуре Трансформеры сводятся к многокаскадному использованию функции softmax, которая являясь частным случаем сравнения по схожести, similarity, выполняют сравнение в пространстве семантических признаков с контекстом. Семантическая схожесть (semantic similarity) векторов ключевой информации (K) и вектора запроса (Q) используется, как весовые коэффициенты для выбора вариантов ответов из значений V. Это и есть принцип работы ассоциативной памяти, механизм Attention.

Embedding модели выполняют преобразование в пространство семантических признаков, получаются вектора определенной длины для последующего сравнения методом семантической схожести. 

Задача Reranking модели - выделить и отсортировать по релевантности из предложенного списка документов. В частности - оценка релевантности документа по отношению к запросу. 

Выход произвольной Embedding модели можно отобразить в пространство семантических признаков другой модели с использованием матриц (Embedding matrix). Так преобразование токенов в языковой модели выполняется с использованием матрицы Embedding слоя сети и словаря. 

Вектора семантических признаков приведенные в единое семантическое пространство можно сравнивать используя определение скалярного произведения (_inner product_) и построенного на его базе косинусной схожести, проекции одного вектора на другой. Скалярное произведение определяется как в евклидовом пространстве через операцию _dot product_.

Может быть иной способ сравнения, который включает нормализацию (приведение в вероятностное пространство) и балансировку весов семантических признаков. Ряд критериев включают смешивание двух и более критериев сравнения с использованием метода mix (линейная интерполяция):
```math
\text{mix}(A, B, \mu) = \mu \cdot A + (1-\mu )\cdot B
```
Скалярный параметр ($\mu$) или вектор параметров $\mu$ определяет вес каждого из критериев. Вес может принимать значения в диапазоне [0,1] и требует активации - отображения вектора входных значений в вероятностное пространство.
```math
\mu = \sigma(W \cdot x + b)
```
Базовые идеи: 
1. Если входное значение уже находится в вероятностном пространстве а матрица представляет собой аффинное (ортогональное) преобразование, функции активации может не быть. Такое происходит при использовании сетей KAN. Сеть построенная исключительно на операциях mix окажется сетью KAN. Для векторов в вероятностном пространстве следует применять операцию скалярного произведения. Если пространство обладает нелинейным распределением, то в состав операции скалярного произведения может входить оконная функция (весовые коэффициенты). Примером является построение пространства (сети KAN) с использованием полиномов Чебышева и Якоби.
2. Операция `softmax` используется для преобразования в вероятностное пространство
Вместо скалярного произведения, которое является естественным решением для вероятностного пространства в этом случае используется _cross entropy loss_. При каких условиях операцию скалярного произведения можно заменить на функцию потерь? Очевидно что для этого используется правила логарифмов произведения и логарифма отношения. 
3. Вероятностное пространство - Гильбертово пространство. 
4. Построение перехода между вероятностной логикой и тернарной логикой. Квантование моделей и метод обучения квантизованных весов с использованием квантизации по времени. 

Метод сравнения может содержать рекурсию, и таким образом учитывать значения с предыдущего шага. Например, в случае сравнения двух векторов в разный момент времени это работает как производная - разница векторов на временном интервале (time-mix). Рекурсия такого рода порождает матричную экспоненту. Матричная экспонента в моделях отсылает к функции _softmax_. 

Техника time-mix отсылает к моделям рекуррентных нейронных сетей (RNN) и позволяет более точно описывать физические модели. В терминах time-mix можно выражать уравнения динамики в обобщенных координатах и в State-space (SSM). Это открывает возможность для разработки динамических моделей в обобщенных координатах.

Следует отметить, возможность представления сети FFN в виде сплайновой сети KAN. Сплайновые сети Колмогорова-Арнольда KSN лучше подходят для задач обработки изображений. Сети KAN могут быть представлены на операции mix и не требуют дополнительной активации, поскольку свойство полиномов Бернштейна (B-сплайнов) таково. Цифровые фильтры изображений могут быть построены на сплайнах. В тоже время могут быть построены и на свертках. Таким образом, интерес представляет возможно представления моделей CNN и MLP в виде сплайновых сетей KAN. 

* [[2309.04664](https://arxiv.org/pdf/1606.08415)] GAUSSIAN ERROR LINEAR UNITS (GELU)\
    -- We could use the CDF of $N (µ, σ^2)$ and have $µ$ and $σ$ be learnable parameters, but throughout this work we simply let $µ = 0$ and $σ = 1$. 
    $$\text{GELU}(x) = xP(X ≤ x) = x\Phi(x)$$ 

**Gaussian distribution, Normal distribution**
In probability theory and statistics, a Normal distribution or Gaussian distribution is a type of continuous probability distribution for a real-valued random variable. 
```math
N(x)= \frac {1}{\sqrt {2\pi \sigma ^{2}}} e^{-{\cfrac {(x-\mu )^{2}}{2\sigma ^{2}}}}
```
The parameter ⁠$\mu$ is the mean or expectation of the distribution (and also its median and mode), while the parameter $\sigma^{2}$ is the variance. 

Заметим, что статистически нормальным будет распределение случайной величины при бесконечном числе опытов. Формула получается как предельный переход (Предельная теорема Муавра-Лапласа). При конечном числе опытов имеет место биномиальное распределение, которое соответствует статистике испытаний Бернулли. По аналогии с формулой биномиального распределения вводятся полиномы Бернштейна, на которых можно строить KSN (сплайновые сети Колмогорова). Формула Бернулли, как и полиномы Бернштейна выводятся через рекуррентное соотношение, с фиксированной вероятностью реализации события при каждом испытании.

**Cumulative distribution function**

The Cumulative Distribution Function (CDF) of the standard normal distribution
```math
\Phi (x)={\frac {1}{\sqrt {2\pi }}}\int _{-\infty }^{x}e^{-t^{2}/2}~dt
```

**Error function**

The related error function $\text{erf}(x)$ gives the probability of a random variable, with normal distribution of mean 0 and variance 1/2 falling in the range ⁠$[-x,x]$⁠.
```math
\text{erf} (x)={\frac {1}{\sqrt {\pi }}}\int _{-x}^{x}e^{-t^{2}}~dt
```

* [[1807.03748](https://arxiv.org/pdf/1807.03748)] Representation Learning with
Contrastive Predictive Coding

* [[2211.14438](https://arxiv.org/pdf/2211.14438)] BERN-NN: Tight Bound Propagation For Neural Networks Using Bernstein Polynomial Interval Arithmetic

* [[2309.04664](https://arxiv.org/pdf/2309.04664)] Compact: Approximating Complex Activation Functions for Secure Computation

* [[2505.24293](https://arxiv.org/pdf/2505.24293)] Large Language Models are Locally Linear Mappings
* [[2406.03865](https://arxiv.org/pdf/2406.03865)] Semantic Similarity Score for Measuring Visual Similarity at Semantic Level

1. Косинусное сходство (Cosine Similarity):

Метод сравнения двух embedding векторов (векторов в пространстве семантических признаков). Вычисляет (проекцию) косинус _угла_ между двумя векторами в n-мерном евклидовом пространстве, нормируя их длину. Значение варьируется от -1 (полностью противоположные) до 1 (идентичные).
Формула:
$$\text{Cosine Similarity} = \frac{\langle A , B\rangle}{\|A\|_2 \|B\|_2} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \cdot \sqrt{\sum_{i=1}^n B_i^2}}$$
где $A$ и $B$ — два эмбеддинга (вектора), $A_i$ и $B_i$ — компоненты их векторов, $\langle \cdot,\cdot\rangle$ — скалярное произведение, $\|A\|_2$ — 2-норма (длина) вектора. 

2. Скалярное произведение нормированных векторов (Dot Product):

Метод сравнения двух embedding векторов, который вычисляет скалярное произведение двух векторов. Значение варьируется от -1 (противоположные) до 1 (идентичные).

Если эмбеддинги предварительно нормированы (2-норма), то формула упрощается до скалярного произведения двух векторов:
$$\text{Dot Product} = \langle A, B \rangle = \sum_{i=1}^n A_i B_i$$

3. Евклидово расстояние (Euclidean Distance):

Измеряет прямое расстояние между двумя точками в пространстве (длину прямой линии). Чем меньше расстояние, тем ближе эмбеддинги.

$$\text{Euclidean Distance} = \sqrt{\sum_{i=1}^n (A_i - B_i)^2}$$

4. Манхеттенское расстояние (Manhattan Distance)

Суммирует абсолютные разницы (SAD) между компонентами векторов. Также называется L1-расстоянием.
Формула:
$$\text{Manhattan Distance} = \sum_{i=1}^n |A_i - B_i|$$

Почему Манхеттенское расстояние? Сколько нужно заплатить за такси, чтобы перейти от точки A до точки B с учетом того что все улицы пересекаются под прямым углом. Такое планирование улиц свойственно для Манхеттена и Барселоны. 

*MSE* is the simplest and most commonly used image similarity metric. Given two images of the same size, $𝑋$ and $𝑌$, where the pixel values are denoted as 𝑥𝑖 and $𝑦_𝑖$ respectively
and the total number of pixels is 𝑁, the MSE can be expressed as:
```math
MSE(𝑋, 𝑌) ={1 \over 𝑁}\sum_{i=1}^N (𝑥_𝑖 − 𝑦_𝑖)^2
```

*PSNR* is to convert *MSE* into the commonly used decibel form in signal processing. From a digital perspective, it has more distinguishing power compared to *MSE*. 
```math
PSNR(𝑋, 𝑌) = 10 \lg \frac{L^2}{MSE(𝑋, 𝑌)}
```
где $L$ - максимальное значение (динамический диапазон значений), $L=255$ для 8-битных изображений. 

5. Индекс структурного сходства Structure-level Metrics

Структурное сходство - это сравнение признаков с использованием масок, причем маски формируются из пиксельный свойств того же изображения. При наложении масок действует правило умножения как операция И в нечеткой логике. 

* [[wang03b](https://www.cns.nyu.edu/pub/eero/wang03b.pdf)]: MULTI-SCALE STRUCTURAL SIMILARITY FOR IMAGE QUALITY ASSESSMENT. Zhou Wang, Eero P. Simoncelli and Alan C. Bovik (2003)

Индекс структурного сходства (SSIM от англ. structure similarity) является одним из методов измерения схожести между двумя изображениями [[2406.03865](https://arxiv.org/pdf/2406.03865)].
*SSIM* метрика рассчитана на различные размеры окна. Разница между двумя окнами 
$x$ и $y$ имеющими одинаковый размер $N×N$:

```math
{\text{SSIM}}(x,y)={\frac {(2\mu _{x}\mu _{y}+c_{1})(2\sigma _{xy}+c_{2})}{(\mu _{x}^{2}+\mu _{y}^{2}+c_{1})(\sigma _{x}^{2}+\sigma _{y}^{2}+c_{2})}},
```
где $\mu_{x}$ — среднее $x$, $\mu_{y}$ — среднее $y$, 
$\sigma_{x}^{2}$ — дисперсия $x$,
$\sigma_{y}^{2}$ — дисперсия $y$,
$\sigma_{xy}$ — ковариация $x$ и $y$, \
$c_{1}=(k_{1}L)^{2}$, 
$c_{2}=(k_{2}L)^{2}$,
$c_{3}=с_{2}/2$, — две константы:
$L$ — динамический диапазон пикселей ($2^{\text{(bits)}}-1$), \
$k_{1}=0{,}01$ и $k_{2}=0{,}03$ — эмпирические стабилизирующие константы, подобраны чтобы избежать деления на ноль при целочисленной арифметике.

В более общем виде предложен критерий сравнения
```math
\text{SSIM}(𝑥, 𝑦) = 𝑙(𝑥, 𝑦)^𝛼 𝑐(𝑥, 𝑦)^𝛽 𝑠(𝑥, 𝑦)^\gamma
```
включает $l(x,y)$ - luminance similarity, 
$c(x,y)$ - contrast  similarity, 
$s(x,y)$ - structure similarity
```math
\begin{aligned}
l(𝑥, 𝑦) &= \frac{2𝜇_𝑥 𝜇_𝑦 + c_1}{𝜇_x^2 + 𝜇_y^2 + c_1} \\
c(𝑥, 𝑦) &= \frac{2𝜎_𝑥 𝜎_𝑦 + c_2}{𝜎_x^2 + 𝜎_y^2 + c_2} \\
s(𝑥, 𝑦) &= \frac{  𝜎_{𝑥𝑦} + c_3}{𝜎_𝑥 𝜎_𝑦 + c_3}
\end{aligned}
```
Константы $c_1, c_2, c_3$, добавлены чтобы снизить ошибку связанную с дискретизацией и избежать деления на ноль.

```math
\mu_x = {1 \over n} \sum^n_{i=1}x_i~, \quad \sigma_x^2 = { {1 \over n} \sum^n_{i=1}(x_i − µ)^2}
```
Ковариация $\sigma_{xy}$ вводится как
```math
\sigma_{xy} = { {1 \over n} \sum^n_{i=1}(x_i − µ_x)(y_i - \mu_y)}
```
По сути терм $s(x,y)$ является коэффициентом корреляции или "косинусной схожестью" двух векторов, считается как проекция в евклидовом пространстве:
```math
s(x,y) = \frac{\sigma_{xy}}{\sigma_x \sigma_y} = \frac{x^\mathsf{T} y}{\|x\|_2 \cdot \|y\|_2}
```

На практике изображение может быть разбито на несколько фрагментов или сравниваться несколько фильтров. Для сравнения можно использовать усреднение (MSSIM) index to evaluate the overall image quality:
```math
MSSIM(X, Y) = {1 \over M} \sum^M_{j=1} SSIM(x_j , y_j),
```

Далее можно перейти к более сложным критериям структурной схожести, которые выражаются через каскад фильтров (MS-SSIM, multiscale -Structural Similarity). Каскад фильтров можно выразить через операцию свертки. Следует обратить внимание на архитектуру CNN (сверточной) сети для сравнения изображений, в частности на архитектуру Feature Pyramid. 

Финальную функцию многокаскадного сравнения MS-SSIM можно трактовать как произведение структурной схожести (s) с учетом маски контрастной схожести(с). К результату (может) применяется маска luminance-similarity. 
```math
\text{MS-SSIM}(𝑥, 𝑦) = 𝑙(𝑥, 𝑦)^𝛼 \prod^N_{k=1}{𝑐(𝑥_k, 𝑦_k)^{\beta_k} 𝑠(𝑥_k, 𝑦_k)^{\gamma_k}}
```
Я не сторонник степенных функций. Однако, если первично составление функции потерь с использованием логарифма от функции структурной схожести, такой критерий преобразуется в сумму с весовыми коэффициентами и оправдывает использование сверточных сетей. 

* [[1910.07467](https://arxiv.org/pdf/1910.07467)] Root Mean Square Layer Normalization

В основе лежит дискуссия о методах нормализации и влиянии нормализации на результат.

**Нормализация слоя** (LayerNorm) призвана улучшить сходимость модели. Прежде всего модель оказывается чувствительной по отношению к функции активации, для нормализации используем базовую идею
```math
\bar{a}_i = \frac{a_i - \mu}{\sigma}g_i~,\quad y_i = f(\bar{a}_i + b_i) 
```
где $\bar{a}_i$ - нормализованное значение, $g_i$ - gain, $b_i$ - bias, $f(\cdot)$ - некоторая функция активации. В статье предполагается что центрирование не столь существенно влияет на сходимость, как масштабирование, и предлагается заменять метод LayerNorm на RMSNorm.  
```math
\bar{a}_i = \frac{a_i}{RMS(a)}g_i~,\quad {RMS}(a) = \sqrt{ {1 \over n} \sum^n_{i=1} a_i^2}
```
* [[2104.14294](https://arxiv.org/pdf/2104.14294)] Emerging Properties in Self-Supervised Vision Transformers
* [[2111.07832](https://arxiv.org/pdf/2111.07832)] IBOT: IMAGE BERT PRE-TRAINING WITH ONLINE TOKENIZER

При обучении моделей с использованием iBot и DINO применяется понятие Cross-Entropy Loss или вариант knowledge distillation loss
```
Alg. Input:
gs, gt ; // student and teacher network
 C, C0 ; // center on [CLS] token and patch tokens
τs, τt ; // temperature on [CLS] token for student and teacher network
τs0,τt0; // temperature on patch tokens for student and teacher network
   l   ; // momentum rate for network
 m, m0 ; // momentum rates for center on [CLS] token and patch tokens

``` 
$u_{[CLS]}$, $u_{patch}$ = g(u, return all tok=true) ; // [n, K], [n, S^2, K]
Результат преобразования с использованием computation graph.

```
The function H(s, t, c, τs, τt) takes five inputs:
    s: Student logits (predictions from a student model).
    t: Teacher logits (predictions from a teacher model).
    c: A centering parameter (likely a scalar or vector).
    τs: Temperature parameter for the student softmax (scalar).
    τt: Temperature parameter for the teacher softmax (scalar).
```

```py
## Algorithm 1 DINO PyTorch pseudocode w/o multi-crop.
# gs, gt: student and teacher networks
# C: center (K)
# tps, tpt: student and teacher temperatures
# l, m: network and center momentum rates
gt.params = gs.params

for x in loader: # load a minibatch x with n samples
    x1, x2 = augment(x), augment(x) # random views
    s1, s2 = gs(x1), gs(x2) # student output n-by-K
    t1, t2 = gt(x1), gt(x2) # teacher output n-by-K
    loss = H(t1, s2)/2 + H(t2, s1)/2
    loss.backward() # back-propagate
    # student, teacher and center updates
    update(gs) # SGD
    gt.params = l*gt.params + (1-l)*gs.params
    C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)

def H(t, s):
    t = t.detach() # stop gradient
    s = softmax(s / tps, dim=1)
    t = softmax((t - C) / tpt, dim=1) # center + sharpen
    return - (t * log(s)).sum(dim=1).mean()
```
The output of the teacher network is centered with a mean computed over the batch.
Each networks outputs a $K$ dimensional feature that is normalized
with a temperature *softmax* over the feature dimension. Their
similarity is then measured with a _cross-entropy loss_. We apply a
stop-gradient operator on the teacher to propagate gradients
only through the student. The teacher parameters are updated with
an _exponential moving average_ (EMA) of the student parameters.

**Knowledge Distillation**

[[1503.02531](https://arxiv.org/pdf/1503.02531)]:
Neural networks typically produce class probabilities by using a *softmax* output layer that converts the logit, $z_i$, computed for each class into a probability, $q_i$, by comparing $z_i$ with the other logits.
```math
q_i = \frac{\exp(z_i/\tau)}{\sum_{j} \exp(z_j/\tau)}
```
where $\tau$ is a temperature that is normally set to $1$.

Парадигма обучения сети сводится к поиску минимума функции перекрестной энтропии, $H(a,b) = -a\cdot \log(b)$
```math
\min_{θ_s} H(P_t(x), P_s(x)),
```
Для нормализации выходных значений сети $g$ с набором параметров $\theta_s$и перехода в вероятностное пространство используется функция `softmax`.
```math
P_s(x)_{i} = \frac{\exp(z_i/\tau_s - z_{max})}{\sum^K_{k=1} \exp(z_{k}/\tau_s - z_{max})}~,\quad \mathbf{z} = g[θs](\mathbf{x})
```
чтобы значения экспоненты не зашкаливали, не вызывали переполнение при расчете, в функцию добавляется смещение `z_max`. Смещение может быть предварительно подобранной константой или вычисляться иначе, например как среднее. В математическом плане функция инвариантна относительно смещения.

* [[2304.07288](https://arxiv.org/pdf/2304.07288)] Cross-Entropy Loss Functions:
Theoretical Analysis and Applications

Обобщенная функция потерь вводится через функцию распределения $\Phi_2(.)$ - не возрастающая, $\Phi_1(.)$ не убывающая. 
```math
\ell_{}(x,y) = \Phi_1 \left( \sum^N_{y\neq y'} \Phi_2(h(x,y)-h(x,y')) \right)
```
```math
\Phi_1^{τ}(u) =
\begin{cases}
\frac{1}{1−τ}
((1 + u)^{1−τ} − 1)~, \quad τ ≥ 0, τ ≠ 1 \\
\log(1 + u)~,\quad τ = 1.
\end{cases}
```
Принимаем $\Phi_2(u) = \exp(−u)$. Функция $\Phi^{τ}_1(u)$ удовлетворяет следующему уравнению
```math
{∂Φ^{τ} \over ∂u} (u) =\frac{1}{(1 + u)^τ}~, \quad Φ^{τ}(0) = 0~.
```

For a single sample $ i $ (row in the batch):

Student softmax:
$$s_{i,j} = \frac{\exp(s_{i,j} / \tau_s)}{\sum_{k=1}^C \exp(s_{i,k} / \tau_s)}, \quad j = 1, \dots, C$$
$ s_i $ is a probability distribution over $ C $ classes.

Teacher softmax:
$$t_{i,j} = \frac{\exp((t_{i,j} - c_j) / \tau_t)}{\sum_{k=1}^C \exp((t_{i,k} - c_k) / \tau_t)}, \quad j = 1, \dots, C$$
$ t_i $ is a probability distribution over $ C $ classes.
Loss:
$$H_i = -\sum_{j=1}^C t_{i,j} \cdot \log(s_{i,j})$$
This is the cross-entropy loss for sample $ i $. For the batch, the output is a vector $ [H_1, H_2, \dots, H_N] $. 


* [[2104.08821](https://arxiv.org/pdf/2104.08821)] SimCSE: Simple Contrastive Learning of Sentence Embeddings

Следует обратить внимание на метод составления функции ошибки для обучения модели, эти же функции могут использоваться для сравнения эмбеддингов. Базовый принцип - contrastive learning.

Contrastive learning используется для эффективного близких по семантическим признакам 

**Определение.**\
It assumes a set of paired examples $D = \{(x_i, x_i^{+})\}^m_{i=1}$, where $x_i$ and $x^{+}_i$ are semantically related. We follow the contrastive framework in Chen et al. (2020) and take a cross-entropy objective with in-batch negatives (Chen et al., 2017; Henderson et al., 2017): 
let $h_i$ and $h^{+}_i$ denote the representations of $x_i$ and $x^{+}_i$, the training objective
for $(x_i, x^{+}_i)$ with a mini-batch of $N$ pairs is:
```math
\mathcal{L}_i = − \log \frac{e^{\mathrm{sim}(h_i,h^{+}_i)/\tau}}{
\sum^{N}_{j=1} e^{\mathrm{sim}(h_i, h^{+}_j)/\tau}}~, 
```
where $\tau$ is a temperature hyperparameter and $\mathrm{sim}(h_1, h_2)$ is the _cosine similarity_ 
```math
\mathrm{sim}(h_1, h_2) = \frac{\langle h_1, h_2 \rangle}{\|h_1\|·\|h_2\|}~.
```
При использовании этого принципа, все равно следует учитывать необходимость повышения контрастности за счет центрирования, вычитания среднего значения как в случае нормализации слоя `LayerNorm` (см. [Centered kernel functions](#)). Дальнейший путь повышения контрастности - динамическое отслеживание среднего уровня в пределах сегмента изображения, рассмотрим ниже в приложении сравнения embedding векторов визуальных сетей, принцип встречается в DINO.

При составлении функции ошибки подобно cross-entropy loss из разных функций, используется выражение вида
```math
\mathcal{L}_{tot} = \alpha_1\mathcal{L}_1 + \alpha_2\mathcal{L}_2 + ... + \alpha_{N} \mathcal{L}_N~,
```
где $\alpha_i$ - весовые коэффициенты, удовляетворяющие требованию $\sum_{i=1}^N \alpha_i = 1$.

Для подбора весовых коэффициентов используется рекурсивный метод `mix(L_1, mix(L_2, ..., b2),b1)`:
```math
\mathcal{L}_{tot} = (1-\beta_1) \cdot \mathcal{L}_1 + \beta_1 \cdot ((1-\beta_2) \cdot \mathcal{L}_2 + \beta_2 \cdot (...))
```
где $\beta_i$ - весовые коэффициенты, удовлетворяющие требованию $\beta_i \in [0,1]$.

## Visual Embeddings

**Сравнение эмбеддингов в SigLIP:**

Как и в случае с текстовыми эмбеддингами, для сравнения эмбеддингов из SigLIP чаще всего используется косинусное сходство (cosine similarity). Это позволяет оценить, насколько близки текст и изображение (или текст и текст, изображение и изображение) в семантическом пространстве.

> *Семантическое пространство* — это математическое пространство, в котором объекты (например, слова, фразы, предложения, изображения или другие данные) представлены в виде числовых векторов таким образом, что расстояние или угол между векторами отражает их семантическую близость или схожесть по смыслу.

Retrieval-Augmented Generation (RAG) is a technique that enhances the capabilities of Large Language Models (LLMs) by integrating them with external knowledge sources. Instead of relying solely on their pre-trained data, RAG allows LLMs to retrieve relevant information from specified documents, databases, or the web before generating a response.

Техника RAG позволяет на основе сравнения embedding векторов от документов или изображений выбрать для генерации несколько документов подходящих к запросу по семантической схожести вектора запрос и вектора от документа. Так например можно выполнить выборку из базы данных и составить промпт запроса с использованием выбранных документов. 

Модели трансформеров строятся с использованием кодирования-декодирования. Этап кодирования переводит промпт в пространство семантических признаков. 

BERT (Bidirectional encoder representations from transformers)

In summary, BERT is an encoder-only transformer model consisting of 4 main parts:

* Tokenizer: chops up texts into sequences of integers.
* Embedding: the module that converts discrete tokens into vectors.
* Encoder: a stack of transformer blocks with self-attention.
* Task head: when encoder is finished with the representations, this task-specific head handles them for token generation or classification tasks.

https://datahacker.rs/005-how-to-create-a-panorama-image-using-opencv-with-python/#Detecting-distinctive-keypoints-

* CLIP (Contrastive Language-Image Pretraining): Trained on image-text pairs, CLIP maps both modalities into a shared embedding space, enabling cross-modal tasks.
* DINO and SimCLR: Self-supervised contrastive learning models that generate robust visual embeddings without labeled data.

* [[DeViSE](https://static.googleusercontent.com/media/research.google.com/ru//pubs/archive/41473.pdf)] DeViSE: A Deep Visual-Semantic Embedding Model
* [[2303.15343](https://arxiv.org/pdf/2303.15343)] Sigmoid Loss for Language Image Pre-Training
* [[2304.07193](https://arxiv.org/pdf/2304.07193)] DINOv2: Learning Robust Visual Features without Supervision
* [[2405.20204](https://arxiv.org/pdf/2405.20204)] JINA CLIP: Your CLIP Model Is Also Your Text Retriever
-- развитие идей DeViSE
* [[2406.03865](https://arxiv.org/pdf/2406.03865)] Semantic Similarity Score for Measuring Visual Similarity at Semantic Level
* [[2406.18587](https://arxiv.org/pdf/2406.18587)] Nomic Embed Vision: Expanding the Latent Space
* [[2502.14786](https://arxiv.org/pdf/2502.14786)] SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Feaures
* [[2503.08723](https://arxiv.org/pdf/2503.08723)] Is CLIP ideal? No. Can we fix it? Yes!
-- в работе применяется метод Dense Cosine Similarity Maps (DCSMs)

На современном этапе развития методов сравнения, наиболее эффективным является *ClipScore* или методы построенные на его основе. 

* [[2406.03865](https://arxiv.org/pdf/2406.03865)]: Semantic Similarity Score for Measuring Visual Similarity at Semantic Level, 2024


*ClipScore* is based on the multimodal model `CLIP` [17], which aligns images and natural language descriptions, enabling it to ”understand” images at a higher level. As a result, it is the most robust to changes weakly correlated with image
semantics. Therefore, it has been widely used in research fields related to visual semantic communication [[2406.03865](https://arxiv.org/pdf/2406.03865)].
```math
d_{CLIP} (𝑋, 𝑌) = 1 − \frac{\langle 𝑒(𝑋) , 𝑒(𝑌) \rangle}{\|𝑒(𝑋)\|_2 \|𝑒(𝑌)\|_2}
```
Where $𝑒(·)$ is the image encoder (output) of the CLIP model, and ClipScore is the *cosine similarity* between the image encodings.

* [[2103.10697](https://arxiv.org/pdf/2103.10697)] ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases
* [[2105.01601](https://arxiv.org/pdf/2105.01601)] MLP-Mixer: An all-MLP Architecture for Vision
* [[2108.08810](https://arxiv.org/pdf/2108.08810)] Do Vision Transformers See Like Convolutional Neural Networks?

    см. Representation Similarity and CKA (Centered Kernel Alignment) 
* [[1203.0550](https://arxiv.org/pdf/1203.0550)] Algorithms for Learning Kernels Based on Centered Alignment\
-- эта работа хороша в плане математических основ для сравнения. Все методы сводятся к скалярному произведению (косинусной схожести). Методы распространяются на матричные функции - kernels. Центрирование повышает контрастность сравнения. Функция вычисления центра распределения может быть представлена как вычисление математического ожидания, среднее значение случайной величины.

**2.1 Centered kernel functions**

Let $D$ be the distribution according to which training and test points are drawn. A feature mapping $\Phi: X → H$ is centered by subtracting from it its expectation, that is forming it by $Φ−\mathsf{E}_x[Φ]$, where
$\mathsf{E}_x$ denotes the expected value of $\Phi$ when $x$ is drawn according to the distribution $D$. Centering a positive definite symmetric (PDS) kernel function $K: X × X → \mathbb{R}$ consists of centering any feature mapping $\Phi$ associated to $K$. Thus, the centered kernel $K_c$ associated to $K$ is defined for all $x, x′ ∈ X$ by 
$$K_c(x, x′) = (Φ(x) − \mathsf{E}_x[Φ])^⊤(Φ(x′) − \mathsf{E}_{x′}[Φ])$$

Я провел эксперимент с функцией cosine similarity -- работает плохо. И в следующем приближении предлагается использовать функцию `softmax` в качестве нормировки $\Phi$ и вместе с тем считать ожидаемое среднее значение по серии кадров или на потоке, как это дается в функции потерь. И функцию косинусной схожести считать как представлено ниже.

В статье дается два определения для сравнения центрированных ядер (kernels) в вероятностном пространстве и в матричной форме
```math
A = \frac{\mathsf{E}[KK']}{\sqrt{\mathsf{E}[K^2]\cdot \mathsf{E}[K'^2]}}, \quad
A_b = \frac{{\langle K, K' \rangle}_{F}}{\|K\|_F \|K'\|_F}
```
В определении используется норма матрицы Фробениуса и произведение Фробениуса (след матрицы) согласованный с определением скалярного произведения. Матричная норма Фробениуса индуцирована определением скалярного произведения Фробениуса.\
Let $⟨·, ·⟩_F$ denote the Frobenius inner product and $\| · \|_F$ the Frobenius norm defined by
```math
\forall A, B \in \mathbb{R}^{n×n}, \quad ⟨A, B⟩_F = \text{Tr}[A^⊤ B], \quad \|A\|_F = \sqrt{⟨A, A⟩_F}~.
```
* [[2311.15419](https://arxiv.org/pdf/2311.15419)] в комплексном пространстве скалярное произведение будет выражаться через произведение матриц:
```math
\forall A, B \in \mathbb{C}^{n×n}, \quad ⟨A, B⟩_{\mathsf{F}} = \text{Tr}[A^{\mathsf{H}} B], \quad \|A\|_{\mathsf{F}} = \sqrt{⟨A, A⟩_{\mathsf{F}}}~.
```
Норма и скалярное произведение Фробениуса выполняется также как если бы матрица была составлена в виде единого вектора-столбца из столбцов исходной матрицы. Таким образом для перехода нужна функция векторизации матрицы, после чего скалярное произведение Фробениуса сводится к операции _dot product_. 

```math
\langle \mathbf {A} ,\mathbf {B} \rangle _{\mathsf{F} }={\overline {\mathrm {vec} (\mathbf {A} )}^{\mathsf{T}}}\mathrm {vec} (\mathbf {B} )\,.
```

Выражения относятся к симметричным положительно (неотрицательно) определенным матрицам.
An $n\times n$ symmetric real matrix $M$ is said to be positive-semidefinite (PSD) or non-negative-definite if $\mathbf {x} ^{\mathsf {T}}M\mathbf {x} \geq 0$ for all 
$\forall \mathbf {x} \in \mathbb{R} ^{n}$. Formally,
```math
M{\text{ positive semi-definite}}\quad \iff \quad \mathbf {x} ^{\mathsf {T}}M\mathbf {x} \geq 0{\text{ for all }}\mathbf {x} \in \mathbb {R} ^{n}
```

* [[1905.00414](https://arxiv.org/pdf/1905.00414)] Similarity of Neural Network Representations Revisited

* [[1912.11370](https://arxiv.org/pdf/1912.11370)] Big Transfer (BiT): General Visual Representation Learning
* [[2010.11929](https://arxiv.org/pdf/2010.11929)] AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

Мейнстрим работ по визуальным сетям представляет собой переход от светочных сетей таких как ResNet к архитектуре Visual Transformers. При этом все крупные игроки такие как Google и Meta прежде всего развивали модели ResNet и использовали их для обучения моделей ViT. Сети ViT показывают лучше результат и производительность чем CNN. 

Архитектура визуального энкодера может быть гибридной, которая включает несколько каналов _feature map_ подготовленных с использованием CNN сетей.

Мы склонны при этом рассматривать заменену традиционной архитектуры CNN на сплайновые сети KSN, а элемент сети MLP на KAN.

**Visual Transformer** (ViT) представляет собой ту же архитектуру Self-Attention Transformers c активацией GELU в блоке MLP. 

и работает с векторами определенной|фиксированной длины D. В данном контексте важно, что из себя представляет вектор выходных значений, которые служат embedding для последующей обработки. На входе модели присутствует линейное разложение на фрагменты ("патчи"), которые формируются путем разбивки нормализованного изображения на фрагменты размером $P\times P$. Исходное изображение имеет разрешение $H\times W \times C$ (высота, ширина, каналы цветности), разбивается и раскладывается на фрагменты $N \times (P^2 \cdot C)$ (см. операцию `im2col`).

Transformer использует постоянный размер скрытого вектора D во всех своих слоях, поэтому мы
сгладить участки и сопоставить их с размерами D с помощью обучаемой линейной проекции (уравнение 1). Мы ссылаемся на выходные данные этой проекции в виде серии патчей (embedding patch).
```math
z_0 = [x_{class}; x^1_p E; x^2_p E; · · · ; x^N_p E] + E_{pos}~, \quad E ∈ \mathbb{R}^{(P^2\cdot C)\times D},~~ E_{pos} ∈ \mathbb{R}^{(N+1)×D}
```
Архитектура слоев ViT включает: MSA - Multi-head Self-Attention, MLP - multilayer perceptron содержит два уровня с функцией активации GELU:
```math
\begin{aligned}
z_\ell'&= \text{MSA}(LN(z_{\ell−1})) + z_{\ell−1}~,\quad \ell = 1 . . . L \\
z_\ell &= \text{MLP}(LN(z_\ell')) + z_\ell'~,\quad \ell = 1 . . . L  \\
y      &= \text{LN} (z_L) 
\end{aligned}
```
*Hybrid Architecture*. As an alternative to raw image patches, the input sequence can be formed
from feature maps of a CNN. In this hybrid model, the patch embedding
projection $E$ (Eq. 1) is applied to patches extracted from a CNN feature map.

*Model Variants*. Варианты модели складываются из размерностей: размер патча $(16\times 16)$, число слоев модели (L), размер нормализованного изображения на входе $D = HW/P^2$ , размерность MLP слоя $d_{ff}$, число каналов цветности (С).
Число параметров модели считается, как произведение $L \cdot D \cdot d_{ff} \cdot C$, 


Метод сравнения *ViTscope* применяется для двух наборов *feature vectors*.
```math
R = {1 \over N} \sum\limits_{i=1}^N \left( \max\limits_{j} x_i^T y_j\right)~,\quad
P = {1 \over M} \sum\limits_{j=1}^M \left( \max\limits_{i} x_i^T y_j\right)
```
Далее применяется усреднение обратных величин $(1/R + 1/P)/2  = 1/V$:
```math
ViTscope = 2 \frac{R\cdot P}{R+P}
```
-- аналогично тому как складываются проводимости при вычислении сопротивления проводников. 
{Мне хочется сравнить результат с методом "contrastive".} 


* [[1908.10396](https://arxiv.org/pdf/1908.10396)] Accelerating Large-Scale Inference with Anisotropic Vector Quantization

## Semantic Segmentation

    *SAM* - Segment Anything 
    *SGG* - Scene Graph Generation
    *SeSS* - Semantic Similarity Score
    *CLIP* - Contrastive Language-Image Pre-training 

Полная архитектура SeSS. Изображения сначала проходят через модель SAM для семантической сегментации. SAM разбивает изображение на набор масок объектов. Этот результат сегментации затем вводится в модель генерации графа сцены SGG, которая выполняет построение отношений между сегментированными объектами, генерируя граф объектов-отношений. Графы объектов-отношений двух изображений, а также исходные изображения и модель CLIP, затем используются для сопоставления графов для вычисления конечного показателя семантической схожести изображений. Этот процесс соответствует процессу восприятия сложных изображений, как это воспринимают люди.

Мы планируем использовать что-то традиционное

* [[1807.10221](https://arxiv.org/pdf/1807.10221)] Unified Perceptual Parsing for Scene Understanding
* [[2111.09883](https://arxiv.org/pdf/2111.09883)] Swin Transformer V2: Scaling Up Capacity and Resolution
* [[2106.13797](https://arxiv.org/pdf/2106.13797)] PVT v2: Improved Baselines with Pyramid Vision Transformer
* [[2302.06378](https://arxiv.org/pdf/2302.06378)] Semantic Image Segmentation: Two Decades of Research
* [[2304.02643](https://arxiv.org/pdf/2304.02643)] Segment Anything
* [[2304.03284](https://arxiv.org/pdf/2304.03284)] SegGPT: Segmenting Everything In Context
* [[2305.08196](https://arxiv.org/pdf/2305.08196)] A comprehensive survey on segment anything model for vision and beyond
* [[2408.00714](https://arxiv.org/pdf/2408.00714)] Sam 2: Segment anything in images and videos, 2024.
* [[2506.18807](https://arxiv.org/pdf/2506.18807)] PicoSAM2: Low-Latency Segmentation In-Sensor for Edge Vision Applications
* (https://docs.ultralytics.com/models/sam/) Сегментация изображений

<!-- * (https://habr.com/ru/articles/801885/) KNN метод ближайших соседей {заменить ссылку на статью}
-->
* (https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

* (https://segmentation-models.readthedocs.io/en/latest/tutorial.html)


* [[2003.12039](https://arxiv.org/pdf/2003.12039)] RAFT: Recurrent All-Pairs Field Transforms for
Optical Flow 


## Simple Recurrent Unit

{Мы обращаемся к моделям реккурентных сетей типа GRU и SRU для демонстрации математических принципов. Раскрыть связь с KAN}

* [[1709.02755](https://arxiv.org/pdf/1709.02755)] Simple Recurrent Units for Highly Parallelizable Recurrence

> We present and explain the design of Simple Recurrent Unit (SRU) in this section. A single layer
of SRU involves the following computation:
```math
\begin{aligned}
f_t &= σ (W_f x_t + v_f \odot c_{t−1} + b_f ) \\
c_t &= f_t \odot c_{t−1} + (1 − f_t) \odot (W x_t) \\
r_t &= σ (W_r x_t + v_r \odot c_{t−1} + b_r) \\
h_t &= r_t \odot c_t + (1 − r_t) \odot x_t
\end{aligned}
```
Тут присутствует рекурсия, которая позволяет описать State Space Model С двумя гейтами (управляющими сигналами): foget и reset. сигнал foget по сути позволяет переключаться между значениями со входа и предыдущими значениями, работает как Enable для входного сигнала. Сигнал reset позволяет сбросить значение к заданному или обойти использование блока. 

При проектировании цифровой электроники существует парадигма, что любая цифровая схема может быть выражена, как каскад триггеров с комбинаторной логикой (FFN) и обратной связью (рекурсией). Триггер - это элемент снабженный входами set и reset. Аналогия может быть использована для перехода к вероятностному пространству, когда управляющие сигналы представлены вещественными числами [0,1], а комбинаторная логика преобразуется в вероятностную логику.

Представление управляющих сигналов $r_t$ и $f_t$ в форме 
```math
r_t = \sigma(W_r x_t + b_r)
```
бесспорно. Дополнительный терм - логика сброса может зависеть от предыдущего состояния системы требует ввести второй сигнал сброса, который какой-то функцией должен быть связан с первым. 
```math
r_t = \sigma(W_r [x_t,c_{t-1}] + b_r)
```

* [[1611.01576](https://arxiv.org/pdf/1611.01576)] QUASI-RECURRENT NEURAL NETWORKS
* [[1705.07393](https://arxiv.org/pdf/1705.07393)] Recurrent Additive Networks

## Линейные трансформеры и рекуррентные нейронные сети

В контексте .. мне кажется уместным провести параллель для представления современных нейросетей типа трансформеры в виде RNN (рекуррентных нейронных сетей) и в частности переход к дифференциальным уравнениям. Трансформеры развиваются, но это не значит, что это верный путь развития. Нужна прямая связь между физическими моделями и архитектурой нейросетей. Должно существовать преобразование приближенное которое позволяет вернуться к математическим основам. В математике (теории вероятностей, теории функции) основой является разложение по базису функций. В частности, разложение по нормированным ортогональным функциям. В теории вероятностей это разложение по независимым событиям. В частности, разложение по базисным векторам в евклидовом пространстве. В приложении нейросетей можно говорить про разложение произвольного вектора в семантическом пространстве по базисным нормированным векторам.
Разложение по базису выполняется с помощью функции скалярного произведения векторов. Скалярное произведение векторов в n-мерном евклидовом пространстве - операция dot product.

Ближайшим представителем нейросетей, которые развиваются в направлении "вернуться к математическим основам" является линейный трансформер и сети типа RWKV, Mamba (SSM, State-Space Models).
* (https://en.wikipedia.org/wiki/State-space_representation) 

* [[2111.00396](https://arxiv.org/pdf/2111.00396)] Efficiently Modeling Long Sequences with Structured State Spaces
* [[2305.13048](https://arxiv.org/pdf/2305.13048)] RWKV: Reinventing RNNs for the Transformer Era
* [[2312.00752](https://arxiv.org/pdf/2312.00752)] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
* [[2401.04081](https://arxiv.org/pdf/2401.04081)] MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts
* [[2403.19887](https://arxiv.org/pdf/2403.19887)] Jamba: A Hybrid Transformer-Mamba Language Model
* [[2405.21060](https://arxiv.org/pdf/2405.21060)] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
* [[2505.18975](https://arxiv.org/pdf/2505.18975)] FastMamba: A High-Speed and Efficient Mamba
Accelerator on FPGA with Accurate Quantization

* (https://habr.com/ru/articles/786278/) Mamba. От начала до конца (обзор)

* [[RWKV-6](https://openreview.net/pdf?id=soz1SEiPeq)] Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence
* [[2404.05892](https://arxiv.org/pdf/2404.05892)] Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence
* [[2404.19756](https://arxiv.org/pdf/2404.19756)] KAN: Kolmogorov–Arnold Networks
* [[2503.14456](https://arxiv.org/pdf/2503.14456)] RWKV-7 "Goose" with Expressive Dynamic State Evolution

В сетях RWKV используется разложение блока Attention к виду рекуррентной нейронной сети. По сути сохраняется некоторая преемственность архитектуры трансформеров, которая включает два блока: $FFN_{ReLU2}$ и RWKV- оператор с рекуррентной связью. Рекуррентная связь в чем-то схожа с рекуррентными нейронными сетями с элементом GRU (Gated Recurrent Unit), SRU (Simple Recurrent Unit).

Я предлагаю рассматривать блок FFN как некоторый класс, который может быть представлен с понижением разрядности вплоть до тернарной логики и с подменой функции активации. Любая логическая функция может быть представлена как комбинация логических операций AND, OR, NOT. Разложение дается в нормальной форме ДНФ или Алгебраической нормальной форме. Такая форма может быть разложена на маски - матрицы коэффициентов в тернарной логике. см. [Тернарная логика](#). Любая логическая функция должна иметь разложение в тернарной логике. Кроме того, класс FFN должен выражать функции в нечеткой вероятностной логике. Приведение к вероятностной логике - функция активации со значениями в диапазоне от 0 до 1.
Другое значимое представление MLP - разложение вероятностной логики в базисе полиномов Бернштейна, представление в виде B-сплайнов. Свойство базисных полиномов - соответствие статистике и теории вероятностей [KAN]. Благодаря свойствам полиномов Бернштейна, не требуется использование функций активации (не требуется искусственно вводить насыщение). В общем случае может быть использован любой набор базисных функций, есть работы которые показывают применимость разложения по ортогональным полиномам Чебышева и Якоби.

Рассмотрению подлежит схема многослойной сети использующая блоки Kolmogorov–Arnold Networks (KAN) и блока обыкновенных дифференциальных уравнений в матричной форме приведенной к архитектуре нейросети (neural ODEs). В кукой то мере эти принципы восходят к математическим основам заложенным в математике Колмогорова и Арнольда. Я не являюсь сторонником, рассматриваю как некоторые упражнения Арнольда, которое в общем виде сводится к математическим основам квантовой физики и теории операторов. В тоже время важно в общем виде рассматривать матричные операции, как разложение на симметричные и антисиметрические компоненты, гамильтоновы матрицы. Доказательством состоятельности модели нейросети может быть возможность описания динамики движения в гамильтоновой динамике и представление дифференциальных уравнений в матричной форме. 

* [[1806.07366](https://arxiv.org/pdf/1806.07366)] Neural Ordinary Differential Equations
* [[2506.16392](https://arxiv.org/pdf/2506.16392)] State-Space Kolmogorov Arnold Networks for Interpretable Nonlinear System Identification
-- Следует обратить внимание на предложенную модель. 
Использование SS-KAN для моделирования физических процессов, таких как движение объектов на видео, где модель аппроксимирует траектории.

* (https://github.com/SynodicMonth/ChebyKAN/)

* [[1907.06732](https://arxiv.org/pdf/1907.06732)] PADÉ ACTIVATION UNITS: END-TO-END LEARNING
OF FLEXIBLE ACTIVATION FUNCTIONS IN DEEP NETWORKS\
 -- В работе рассматривается возможность представления функций активации в виде рациональных функций. От себя хочется добавить возможность разложения степенных рядов по полиномам Бернштейна и полиномам Тейлора.

```math
F(x) = \frac{P(x)}{Q(x)}
= \frac{ \sum_{j=0}^m a_jx^j }{1 + \sum_{k=1}^n b_kx^k }
```
* [[1704.07483](https://arxiv.org/pdf/1704.07483)] Continuously Differentiable Exponential Linear Units (ELU, CELU)
* [[2410.10084](https://arxiv.org/pdf/2410.10084)] POINTNET WITH KAN VERSUS
POINT NET WITH MLP FOR 3D CLASSIFICATION AND SEGMENTATION OF POINT SETS
* [[2505.22686](https://arxiv.org/pdf/2505.22686)] Localized Weather Prediction Using
Kolmogorov-Arnold Network-Based Models and Deep RNNs
* [[2506.06644](https://arxiv.org/pdf/2506.06644)] Spark Transformer: Reactivating Sparsity in FFN and Attention
* [[2506.14802](https://arxiv.org/pdf/2506.14802)] SS-MAMBA: SEMANTIC-SPLINE SELECTIVE STATE-SPACE MODEL
* [[2506.18339](https://arxiv.org/pdf/2506.18339)] Structured Kolmogorov-Arnold Neural ODEs for Interpretable Learning and Symbolic Discovery of Nonlinear Dynamics

* (https://arxiv.org/pdf/1502.03167) - описание операции нормировки блоков
* [[2006.16236](https://arxiv.org/pdf/2006.16236)] Transformers are RNNs
* [[2409.18747](https://arxiv.org/pdf/2409.18747)] COTTENTION: LINEAR TRANSFORMERS WITH COSINE ATTENTION

```math
ATTN_l(x) = V' = \text{softmax }\left(\frac{Q\cdot K^{T}}{\sqrt{D}}\right) V
```
Equation 2 implements a specific form of self-attention called *softmax* attention where the _similarity score_ is the exponential of the dot product between a Query and a Key.
Given that subscripting a matrix with i returns the i-th row as a vector, we can write a generalized attention equation for any similarity function as follows,
```math
V'_i = \frac{\sum_{j=1}^N {\text{sim }(Q_i, K_j) V_j}}{\sum_{j=1}^N {\text{sim }(Q_i, K_j)}}
```
Эти два уравнения эквиваленты при подстановке 
$$\text{sim }(q, k) = \exp \left(\frac{q^T k}{\sqrt{D}}\right)$$


* [[2506.06941](https://arxiv.org/pdf/2506.06941)] The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity

**KSN/KAN:**\
В современной научной литературе сети KAN позиционируются, как замена традиционных MLP с сохранением механизма Attention. MLP представляют собой последовательность слоев линейных преобразований и нелинейных функций активации, такой как `ReLU` или `GELU`. MLP не могут быть интерпретированы, а также не могут быть использованы для построения физических моделей. В некоторых случаях модели KAN могут быть интерпретированы через производные и конечные разности, квантизация поверх KAN имеет физическую интерпретацию, использование базовых полиномов Бернштейна или ортогональных функций приводит модель в Гильбертово или Вероятностное пространство без использования Clamp и Softmax и нелинейной активации. 

Сплайновые сети (KSN) и KAN используют B-сплайны для представления нелинейных зависимостей, что делает их интерпретируемыми и подходящими для задач обработки изображений [Igelnik03][2404.19756]. Цифровые фильтры могут быть представлены B-сплайнами и имеют физическую и геометрическую интерпретацию.

* [K57] A. N. Kolmogorov, “On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition, ” Dokl. Akad. Nauk SSSR, pp. 953–956, vol. 114, 1957.
* [[Igelnik03](https://ieeexplore.ieee.org/document/1215392)] B. Igelnik, at al. Kolmogorov’s Spline Network, IEEE Transactions on Neural Networks ( Volume: 14, Issue: 4, July 2003)
* [[2410.04096](https://arxiv.org/pdf/2410.04096)] Sinc Kolmogorov-Arnold Network and Its Applications on Physics-informed Neural Networks

-- Не могу согласиться, использование разного рода полиномов кроме ортогонльных полиномов и базисных полиномов, на мой взгляд, уводит в сторону от физики. Можно согласиться использовать ортогональные полиномы Эрмита, Чебышева, Якоби в задачах физических и так называемых PINN (Physic-informed Neural Network), таких как описание волн на поверхности или описание динамики частиц. В данной работе надо как-то обозначить возможность представление физики с использованием базисных полиномов и конечных разностей подобно фильтрам с бесконечной импульсной характеристикой. В оригинальной работе KAN я не могу согласиться с использованием SILU активации. По сути предложенная архитектура не является полностью КАN. Вся сеть должна строится на принципах KAN. Необходимо использовать модель основанную на фильтрах с бесконечной-импульсной характеристикой и с физической интерпретацией модели в Z-пространстве, возможно использовать архитектуру SSM (State-Space Models) и рекуррентные модели (RNN такие как GRU и STU).

