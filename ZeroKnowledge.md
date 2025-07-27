

Георгиевский А.М.

Дается обзор методов шифрования с использованием структуры нейросети для безопасного запуска моделей нейросетей и методов доказательства c нулевым разглашением (ZKP). Постановка задачи шифрования данных полностью гомоморфного шифрования (FHE) с использованием принципов пост-квантовой криптографии (PQC) и построения доказательства с использованием цепной записи данных (блокчейн). Цель - сформировать необходимый уровень знаний для разработки собственной концепции запуска нейросетей с использованием RPC протокола и доказательства ZKP. Рассматриваются варианты канонизации описания графов и тензоров для передачи заданий по сети в рамках RPC протокола. 

**Zero-Knowledge Machine Learning** (ZKML, машинное обучение с нулевым разглашением) is a cryptographic technique that enables the verification of machine learning models on blockchain networks without revealing the underlying data or computations. This technology allows for secure, privacy-preserving, and transparent use of AI models in decentralized applications, ensuring the integrity and trustworthiness of the results. ZKML is particularly useful in DeFi, gaming, and identity verification, where it can enhance user experience, automate decision-making processes, and protect sensitive information.

* [[2507.07031](https://arxiv.org/pdf/2507.07031)] ZKTorch: Compiling ML Inference to Zero-Knowledge Proofs via Parallel Proof Accumulation
Authors: Bing-Jyue Chen, Lilia Tang, Daniel Kang

* [[2502.18535](https://arxiv.org/pdf/2502.18535)] A Survey of Zero-Knowledge Proof Based Verifiable Machine Learning
Authors: Zhizhi Peng, Taotao Wang, Chonghe Zhao, Guofu Liao, Zibin Lin, Yifeng Liu, Bin Cao, Long Shi, Qing Yang, Shengli Zhang

* [[2410.13752](https://arxiv.org/pdf/2410.13752)] Privacy-Preserving Decentralized AI with Confidential Computing
Authors: Dayeol Lee, Jorge António, Hisham Khan

* [[2409.12055](https://arxiv.org/pdf/2409.12055)] Artemis: Efficient Commit-and-Prove SNARKs for zkML
Authors: Hidde Lycklama, Alexander Viand, Nikolay Avramov, Nicolas Küchler, Anwar Hithnawi

* [[2405.17934](https://arxiv.org/pdf/2405.17934)] Proof of Quality: A Costless Paradigm for Trustless Generative AI Model Inference on Blockchains
Authors: Zhenjie Zhang, Yuyang Rao, Hao Xiao, Xiaokui Xiao, Yin Yang

* [[2404.16109](https://arxiv.org/pdf/2404.16109)] zkLLM: Zero Knowledge Proofs for Large Language Models

* [[2402.15006](https://arxiv.org/pdf/2402.15006)] opp/ai: Optimistic Privacy-Preserving AI on Blockchain
Authors: Cathie So, KD Conway, Xiaohang Yu, Suning Yao, Kartin Wong

* [[2402.06414](https://arxiv.org/pdf/2402.06414)] Trust the Process: Zero-Knowledge Machine Learning to Enhance Trust in Generative AI Interactions
Authors: Bianca-Mihaela Ganescu, Jonathan Passerat-Palmbach

* [[2401.17555](https://arxiv.org/pdf/2401.17555)] opML: Optimistic Machine Learning on Blockchain
Authors: KD Conway, Cathie So, Xiaohang Yu, Kartin Wong

> Объединение технологий искусственного интеллекта (ИИ) и технологии блокчейн меняет цифровой мир, предлагая децентрализованные, безопасные и эффективные сервисы ИИ на блокчейн-платформах. Несмотря на обещания, высокие вычислительные требования ИИ на блокчейне вызывают серьёзные проблемы с конфиденциальностью и эффективностью. Фреймворк Optimistic Privacy-Preserving AI (opp/ai) представлен как новаторское решение этих проблем, обеспечивая баланс между защитой конфиденциальности и вычислительной эффективностью. Фреймворк объединяет машинное обучение с нулевым разглашением (zkML) для обеспечения конфиденциальности с оптимистичным машинным обучением (opML) для повышения эффективности, создавая гибридную модель, специально разработанную для сервисов ИИ на блокчейне. В данном исследовании представлен фреймворк opp/ai. 


* [[arXiv:2402.06414](https://arxiv.org/pdf/2402.06414)] Trust the Process: Zero-Knowledge Machine Learning to Enhance Trust in Generative AI Interactions
Authors: Bianca-Mihaela Ganescu, Jonathan Passerat-Palmbach

* [[arXiv:2401.17555](https://arxiv.org/pdf/2401.17555)] opML: Optimistic Machine Learning on Blockchain
Authors: KD Conway, Cathie So, Xiaohang Yu, Kartin Wong

— использование машинного обучения с нулевым разглашением (zkML). zkML представляет собой новую парадигму интеграции машинного обучения и блокчейна. zkML использует *zk-SNARK* (краткие неинтерактивные аргументы знаний с нулевым разглашением) и играет ключевую роль в защите конфиденциальных параметров модели и пользовательских данных во время процессов обучения и вывода (inference). Это не только снижает проблемы конфиденциальности, но и снижает вычислительную нагрузку на сеть блокчейна, что делает zkML перспективным кандидатом для децентрализованных приложений машинного обучения.

* [[2502.02387](https://arxiv.org/pdf/2502.02387)] SoK: Understanding zk-SNARKs: The Gap Between Research and Practice

* [[2506.20915](https://arxiv.org/pdf/2506.20915)] ZKPROV: A Zero-Knowledge Approach to Dataset Provenance for Large Language Models

**Cryptographic Primitives**
> Zero-Knowledge Succinct Non-Interactive Argument of Knowledge (*zk-SNARK*). A *zk-SNARK* is a cryptographic proof system that allows a prover to convince a verifier that a statement $𝑥 ∈ L_𝑅$ is valid with respect to a relation 𝑅, without revealing any auxiliary information (i.e., the witness 𝜔).

Примитивы, из которых строится схема доказательства:
1. $SETUP (\lambda, W) → \{C,\Omega\}$  - компиляция модели дает два набора векторов ($𝐶$)-публичный набор и приватный ($\Omega$).
2. $Prove(𝐶, \Omega, 𝑝) → \{r, 𝜋\}$ - запуск модели является доказательство над промптом пользователя (p) генерирует ответ ($r$) и $\pi$ - доказательство. Доказательство - это совместно результат вывода модели inference и генерация проверочного вектора ($\pi$).
3. $Verify(𝑝, 𝑟, 𝜋, 𝐶) → {Accept, Reject}$ -- верификация выполняется с использованием публичных компонент доказательства, над промптом и выводом модели. 

Этап компиляции $SETUP()$ можно разложить, на две процедуры: генерацию приватного ключа $pp = KeyGen(\lambda)$ и публичного набора для данной модели $\Omega = Commit(W; \omega, pp)$

Важным параметрами построения схемы доказательства является возможность верификации доказательства на стороне клиента и объем данных необходимых для выполнения этой проверки. Важно, чтобы сложность верификации позволяла выполнять проверку налету в процессе загрузки результатов запуска сети. Важно, чтобы сигнатура сети $C$ занимала ограниченный объем данных, а доказательство было сравнимо по длине с вектором семантических признаков `m_embed`.

Операция генерации публичного набора основано на Гомоморфизме приватной модели нейросети. Гомоморфные преобразования сохраняют структуру. Каждой операции сопоставляется друга. При этом есть разногласие в представлении макро-операций таких как MLP, FFN и Attention. Тензорные макро-операции можно представлять, как составные или неделимые. Каждой нелинейной операции необходимо сопоставить линейное (полиномиальное) представление. 

Наравне с ZKP и SNARK следует рассмотреть принципы построения криптосистемы FHE (Fully Homomorphic Encryption scheme), такой как Ring-LWE которая обеспечивает шифрование и дешифрацию. 

Рассмотрение можно начать с принципа поля. RSA так же является HE (гомоморфным преобразованием)

* [CryptoNets](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/CryptonetsTechReport.pdf): Applying Neural Networks to Encrypted Data with High Throughput and Accuracy Homomorphic Encryption, 2016
* [[1412.6181](https://arxiv.org/pdf/1412.6181)] Crypto-Nets: Neural Networks over Encrypted Data

> For our purpose, a (secret key) Homomorphic Encryption scheme consists of four algorithms: 
encryption ($E_k$), decryption $D_k$, addition (⊕) and multiplication (⊗). 
The encryption algorithm takes as input a message and a secret key $k$. The decryption takes as input an element from the ciphertext space and a key, while the algorithms ⊕ and ⊗ do not depend on the secret key and only take two ciphertexts as input. Let $m_1$ and $m_2$ be integer messages and let $k$ be a secret key. 

Представленные алгоритмы удовлетворяют следующим свойствам:

1. Функция шифрования $E_k(m)$, такая что $m$ практически невозможно восстановить обратно без использования приватного ключа $k$.
2. Существует обратная функция: $m_1 = D_k(E_k(m_1))$.
3. Выполняется свойство линейности: $m_1 + m_2 = D_k (E_k(m_1) ⊕ E_k(m_2))$.
4. Выполняется свойство линейности: $m_1 × m_2 = D_k (E_k(m_1) ⊗ E_k(m_2))$.
5. Алгоритмы $⊕$ и $⊗$ не используют закрытый ключ шифрования.

Алгоритмы $⊕$ и $⊗$ позволяют многократное каскадное применение, так что полиному над $m_i$ соответствует аналогичный полином с операциями $⊕$ и $⊗$.

Let $m_1, ... , m_n$ be messages. Then the above algorithms satisfy the following property:
$$P(m_1, ... , m_n) = D_k(\tilde{P}(E_k(m_1), ... , E_k(m_n)))~.$$

Криптосистема *Homomorphic Encryption* состоит из четырех алгоритмов. Для компиляции схемы из вычислительного графа нейросети используются полиномиальные приближения нелинейных тензорных операций, все коэффициенты строятся из целых чисел по модулю простого числа и операции сдвига (XTIME - вместо масштабирования используется уполовинивание или удвоение (умножение и деление полиномов), заменяющее собой возведение в степень, как при умножении вещественного числа). Схема Ring-LWE использует кольца полиномов. По сути мы говорим про модульную арифметику (алгебру над коммутативным кольцами) или арифметику Галуа в конечном поле. Однако, могут существовать и более сложные варианты - композитные поля и Алгебры Ли. Во всех случаях требуется представлять нелинейные функции их полиномиальной аппроксимацией, чтобы все операции сводились к умножению и сложению в поле. Заметим, если для операции умножения существует обратная, то существует возможность представления в рациональных функциях (аппроксимация Паде́).

## Fully Homomorphic Encryption (FHE)

Brakerski/Fan-Vercauteren [Bra12, FV12] scheme, a Ring-Learning With Errors (Ring-LWE)-based crypto-system. Позволяет восстановить данные после шифрования и обладает свойствами пост-квантовой криптографии. 

* (https://people.csail.mit.edu/rivest/Rsapaper.pdf) R.L. Rivest, at al., A Method for Obtaining Digital Signatures and Public-Key Cryptosystems, 1978
* [Homomorphic Encryption Standardization](https://homomorphicencryption.org/) 
* [[HESv1.1](https://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf)] Homomorphic Encryption Standard, 2018
* (https://crypto.stanford.edu/craig/craig-thesis.pdf)
* (https://www.cs.cmu.edu/~odonnell/hits09/gentry-homomorphic-encryption.pdf)
* (https://cims.nyu.edu/~regev/papers/pqc.pdf) Lattice-based Cryptography
* [[2011/277](https://eprint.iacr.org/2011/277.pdf)] Zvika Brakerski, Craig Gentry, Vinod Vaikuntanathan. Fully Homomorphic Encryption without Bootstrapping, 2011
* [[2011/344](https://eprint.iacr.org/2011/344.pdf)] Zvika Brakerski, Vinod Vaikuntanathan. Efficient Fully Homomorphic Encryption from (Standard) LWE, 2011
* [[2012/078](https://eprint.iacr.org/2012/078.pdf)] Zvika Brakerski. Fully Homomorphic Encryption without Modulus Switching from Classical GapSVP, 2012
* [[2012/144](https://eprint.iacr.org/2012/144.pdf)] Junfeng Fan and Frederik Vercauteren. Somewhat Practical Fully Homomorphic
Encryption, 2012
* [[2013/340](https://eprint.iacr.org/2013/340)] Homomorphic Encryption from Learning with Errors: Conceptually-Simpler, Asymptotically-Faster, Attribute-Based, 2013
* [[2014/816](https://eprint.iacr.org/2014/816.pdf)] FHEW: Bootstrapping Homomorphic Encryption in less than a second
* [[2016/421](https://eprint.iacr.org/2016/421.pdf)] Homomorphic Encryption for Arithmetic of Approximate Numbers
> CKKS (Cheon-Kim-Kim-Song) — это схема полностью гомоморфного шифрования (FHE), предназначенная для эффективных вычислений с вещественными числами. Она позволяет выполнять операции сложения, умножения и другие над зашифрованными данными, не раскрывая исходную информацию.

* [[2016/837](https://eprint.iacr.org/2016/837.pdf)] J.H. Cheon and D. Stehle. Fully Homomorphic Encryption over the Integers Revisited, 2016
* [[2018/421](https://eprint.iacr.org/2018/421.pdf)] TFHE: Fast Fully Homomorphic Encryption over the Torus
> четко даются определения и математические основы

* [[2020/086](https://eprint.iacr.org/2020/086.pdf)] Bootstrapping in FHEW-like Cryptosystems
* [[2021/1337](https://eprint.iacr.org/2021/1337)] Large-Precision Homomorphic Sign Evaluation using FHEW/TFHE Bootstrapping
* [[2024/463](https://eprint.iacr.org/2024/463.pdf)] Security Guidelines for Implementing Homomorphic Encryption
* [[2025/882](https://eprint.iacr.org/2025/882.pdf)] Leveled Homomorphic Encryption over Composite Groups

* [[2103.16400](https://arxiv.org/pdf/2103.16400)] Intel HEXL: Accelerating Homomorphic Encryption with Intel AVX512-IFMA52
* [[2401.03703](https://arxiv.org/pdf/2401.03703)] On Lattices, Learning with Errors, Random Linear Codes, and Cryptography
* [[2503.05136](https://arxiv.org/pdf/2503.05136)] The Beginner’s Textbook for Fully Homomorphic Encryption
* [[2305.05904](https://arxiv.org/pdf/2305.05904)] Revisiting Fully Homomorphic Encryption Schemes
* [[2507.04501](https://arxiv.org/pdf/2507.04501)] LINE: Public-key encryption

* (https://faculty.kfupm.edu.sa/coe/mfelemban/SEC595/References/Introduction%20to%20the%20BFV%20FHE%20Scheme.pdf)
* (https://github.com/openfheorg/openfhe-development)
> Fully Homomorphic Encryption (FHE) is a powerful cryptographic primitive that enables performing computations over encrypted data without having access to the secret key. OpenFHE is an open-source FHE library that includes efficient implementations of all common FHE schemes:
* Brakerski/Fan-Vercauteren (BFV) scheme for integer arithmetic
* Brakerski-Gentry-Vaikuntanathan (BGV) scheme for integer arithmetic
* Cheon-Kim-Kim-Song (CKKS) scheme for real-number arithmetic

Software references for publicly available Homomorphic Encryption libraries:
* [cuFHE] (https://github.com/vernamlab/cuFHE)
* [cuHE] (https://github.com/vernamlab/cuHE)
* [HEAAN] (https://github.com/snucrypto/HEAAN)
* [HElib] (https://github.com/shaih/HElib)
* [NFLlib] (https://github.com/CryptoExperts/FV-NFLlib)
* [PALISADE] (https://git.njit.edu/groups/PALISADE)
* [SEAL] (http://sealcrypto.org)
* [TFHE] (https://tfhe.github.io/tfhe/)

**Обозначения**\
Мы вводим обозначения $\mathbb{Z}_q$ как множества целых чисел $(−q/2,q/2]$ где $q>1$ - целые числа. Все целочисленные операции выполняются по модулю $(\mod q)$ если не сказано обратное. Для упрощения, you will see me mostly deal with positive integers in $[0,q)$, but keep in mind that it’s the same as our $\mathbb{Z}_q$, as $−x ≡ q–x(\mod q)$ where $x$ is a positive integer (e.g. $−1≡6~(\mod 7)$ ). Каждый вектора $v∈\mathbb{Z}^n_q$ можно рассматривать как вектор элементов из класса $\mathbb{Z}_q$.

We will use $[⋅]_m$ to specify that we are applying modulo $m$, and $⌊⋅⌉$ for rounding to the nearest integer.

Угловыми скобками $⟨a,b⟩$ обозначим скалярное произведеление (inner product) двух векторов $a,b∈\mathbb{Z}^n_q$ и определим операцию через умножение и сложение по модулю $q$
```math
⟨a,b⟩=\sum\limits_{i}^n a_i⋅b_i~(\mod q)
```

**Learning With Error**\
Learning With Error (LWE) was introduced by [Regev in 2009](https://cims.nyu.edu/~regev/papers/qcrypto.pdf) and can be defined as follows: 
Для целых чисел $n≥1$ и $q≥2$, let’s consider the following equations
```math
⟨s,a_1⟩+e_1=b_1~(\mod q)\\
⟨s,a_2⟩+e_2=b_2~(\mod q)\\
…\\
⟨s,a_m⟩+e_m=b_m~(\mod q)
```

where s and ai are chosen independently and uniformly from $\mathbb{Z}^n_q$, and $e_i$ are chosen independently according to a probability distribution over $\mathbb{Z}_q$, and $b_i∈\mathbb{Z}_q$. The LWE problem state that it’s hard to recover s from the pairs $(a_i,b_i)$, and it’s on such hardness that cryptography generally lies. On the list of candidate algorithms for the post-quantum cryptography standardization are some that are based on LWE, so you would probably hear more about it when it would be used in key- establishment and public-key encryption.

**Ring Learning With Error**\
Ring-LWE is a variant of LWE, it’s still based on the hardness of recovering s from the pairs $(a_i,b_i)$, and the equations are mainly the same, however, we go from the world of integers ($\mathbb{Z}^n_q$) to the world of polynomial quotient rings ($\mathbb{Z}_q[x]/⟨x^n+1⟩$), this means that we will deal with polynomials with coefficients in $\mathbb{Z}_q$, and the polynomial operations are done (mod) some polynomial that we call the polynomial modulus (in our case: $⟨x^n+1⟩$), so all polynomials should be of degree $d<n$, and $x^n ≡ −1(\mod⟨x^n+1⟩)$.

Let’s now use a more formal [definition of Ring-LWE by Regev](https://cims.nyu.edu/~regev/papers/lwesurvey.pdf):\
Let $n$ be a power of two, and let $q$ be a prime modulus satisfying $q = 1(\mod 2n)$. Define $R_q$ as the ring $\mathbb{Z}_q[x]/⟨x^n+1⟩$ containing all polynomials over the field $\mathbb{Z}_q$ in which $x_n$ is identified with $−1$. In Ring-LWE we are given samples of the form $(a,b=a⋅s+e)∈ R_q×R_q$ where $s∈Rq$ is a fixed secret, a∈Rq is chosen uniformly, and e is an error term chosen independently from some error distribution over Rq.

So if we want to build an HE scheme using Ring-LWE, then our basic elements won’t be integers, but polynomials, and you should be familiar with basic polynomial operations (addition, multiplication and modulo). I cooked up a quick refresher of polynomial operations to avoid getting off on the wrong foot, but you can just skip it if it’s a trivial thing for you.


(https://mathworld.wolfram.com/PadeApproximant.html)
Прежде всего нас может интересовать метод вычисления аппроксимации Паде для экспоненциальных функций, таких как `SiLU`, `GELU`, `tanh` и `exp`, в составе функции `softmax`. 

Однако конкретное представление (аппроксимация) нелинейной функции будет определяться настройкой криптосистемы и степенью полиномов в числителе и знаменателе рациональной функции.

К примеру для степени {3/3},
```math
\exp_{3/3}(x) = \frac{\exp(+x/2)}{\exp(-x/2)} \approx \frac{120+60x+12x^2+x^3}{120-60x+12x^2-x^3}~.
```
Определив экспоненту, как ряд или как отношение рядов, можно применить такое определение и к вектору и к матрице (Матричная экспонента). Стоит отметить, что в системе конманд x86 отсутствует векторная инструкция расчета экспоненты. Все такие функции эмулируются, за исключением логарифма. Таким образом, просто вводя правило вычисления экспоненты можно устранить неопределенность. 

Для простоты понимания материала, я предпочитаю представлять функцию шифрования $E_k(.)$, как матрицу ортогонального (Аффинного) преобразования размером `n_embed`, для которой существует обратное преобразование $D_k(.)$ - обратная матрица. Такое представление интуитивно понятно для CNN сетей, но для каждой функции следует определить аппроксимацию и степень полинома. С другой стороны аппроксимация может быть основана на ортогональных и базисных полиномах, таких как полиномы Чебышева и Якоби, Базисные полиномы Бернштейна. При использовании систем базисных и ортогональных полиномов для аппроксимации MLP (Feed-forward Network) возникает представление в виде сетей KAN.

**Функции активации**

* [[1606.08415](https://arxiv.org/abs/1606.08415)] Gaussian Error Linear Units (GELUs)
* [[2002.05202](https://arxiv.org/pdf/2002.05202)] GLU Variants Improve Transformer

Базовые аппроксимации вводятся для функций активации таких SwiGLU и GELU. 
```math
\text{SwiGLU}_𝛽(𝑧, 𝑧′) := \text{Swish}_𝛽(𝑧) · 𝑧′ \\
\text{Swish}_{\beta}(x)=x\text{Sigmoid} (\beta x)={\frac {x}{1+e^{-\beta x}}}\\
\text{GELU}(𝑧) := 𝑧 \Phi(𝑧) ≈ 𝑧\cdot \text{Sigmoid}(1.702𝑧)
```

**Lemma 1**. Let $N$ be a neural network in which all non-linear transformations are continuous. Let
$X ⊂ \mathbb{R}^n$ be the domain on which $N$ acts and assume that $X$ is compact, then for every $\epsilon > 0$ there exists a polynomial $P$ such that 
```math
\sup\limits_{x∈X} \| N(x) − P(x) \| < \epsilon
```
Основное в этом утверждении непрерывность и компактность, которые позволяют обосновать применение полиномов. Я бы отослал к аппроксимационной теореме Вейерштрасса и доказательству Бернштейна с выводом системы базисных полиномов.

Аппроксимационная теорема Вейерштрасса утверждает, что любую непрерывную функцию на отрезке можно сколь угодно точно аппроксимировать многочленами (то есть подобрать последовательность многочленов, равномерно сходящихся к этой функции на данном отрезке). 

* (https://github.com/microsoft/CryptoNets)

> Encrypting data is a prominent method for securing and preserving privacy of data. Homomorphic encryption (HE)
(Rivest et al., 1978) adds to that the ability to act on the data while it is still encrypted. In mathematics, a homomorphism is a structure-preserving transformation.

Существует ряд библиотек предназначенных для выполнения операций типа ZKP (zero-knowledge proof), 

* 

* (https://docs.ezkl.xyz/)

Доказательство целостности модели выполняется с использованием описания открытой архитектуры и весов модели в формате ONNX (Open Neural Network eXchange). На первом этапе строится схема доказательства с использованием compute graph графа тензорных операций нейросети, нелинейные операции в графе вычисления подменяются на полиномиальную аппроксимацию. Все вычисления выполняются в числах с фиксированной точкой. 

```py
import ezkl
settings = ezkl.gen_settings()
ezkl.compile_model("model.onnx", "compiled_model.ezkl", settings)
proof = ezkl.prove("witness.json", "compiled_model.ezkl", "pk.key")
```

Тезис. Мы хотим сформулировать более современный метод компиляции графа вычислительной сети с использованием принципов zero-knowledge, который бы являлся ключом для построения сложных схем доказательства наподобие merkle tree и применялся практически к любым вычислительным сетям. Компиляция графа доказательства должны выполняться с использованием коэффициентов в открытых форматах `GGUF` или `Safetensors`. Тензоры изначально представленные в формате BF16 должны быть квантизованны в числах с фиксированной точностью пригодные для модульной арифметики. 
Необходимо разработать метод канонизации описания графа тензорных операций и представления в бинарном формате основанном на `CBOR` кодировании, работающий на множестве операций поддерживаемых в `llama.cpp` и `ONNX`. При вычислении графа над результатом каждой тензорной операции выполняется функция квантования построенная по принципу модульной арифметики и сдвига в конечном поле (умножение и редуцирование полиномов). Должны быть разработаны методы квантизации 2-бит, 4-бит, 8-бит, а также квантизация в тернарную логику. Методы квантизации должны быть оптимизированы под современную архитектуру тензорных ядер GPU. 

В схему компиляции предлагается добавить подбор _nonce_ (квантовой ошибки используемой при округлении результата операции) для операции хеширования тензора на каждой тензорной операции. Для достижения критерия вычислительной сложности составления схемы помимо валидности хеш, подбор выполняется при вычислении каждого хеша после каждой операции в дереве. Набор _nonce_ полученных таким образом включают в схему проверки. Что это дает? Проверка поверх известной сети будет выполняться на любом оборудовании в достаточно короткое время, в то время, как расчет подобной схемы требует большого количества вычислений, что делает практически невозможным составление второй схемы с подменой весов и идентичным результатом. Каждому узлу в дереве будет соответствовать свой _nonce_, валидность вычисления будет даваться не только значением вектора но соблюдением критерия сложности подбора _nonce_. 

При построении более сложных схем мы предлагаем использовать функцию двойного хеширования sha256d подобно тому, как это делается в хорошо зарекомендовавших себя принципах технологии цепной записи данных (блок чейн). Использование одиночной функции создает возможность для снижения вычислительной сложности компиляции схемы доказательства с подменой одного из значений в векторе. 

Схема доказательства должна быть разработана с учетом возможности модификации модели с использованием технологии LoRa. Доказательство целостности модели является целостность базовой модели и целостность производной модели с учетом изменений. 

Стандартизация
* [МР 26.4.001-2018] «Термины и определения в области технологий цепной записи данных (блокчейн) и распределенных реестров»
* [ISO/TC307] Blockchain and distributed ledger technologies

Как доказать, что модель применена к нашим данным. Как доказать не видя архитектуры, что в нее не внесли изменения и не выполнили какую-то модификацию, включая квантование модели, которая приводит к иным результатам на некоторой итерации работы модели. Даже объединение промпта в батч задание способно дать другой результат. Мы понимаем, что одинаковые условия запуска модели должны давать одинаковый результат. 

Доказательством целостности модели является генерация определенной последовательности самой моделью. Однако модели строятся с использованием "температуры генерации" и начальных условий "seed" которые могут влиять на генерацию. Считается, что повышение *температуры* увеличивает креативность генерации. Для визуальных моделей encode-only температура должна быть константой. Прежде всего следует убедиться, что современные модели возможно разделить на Encoder-only часть, как в случае BERT, SigLIP и Embedding моделей и выявить промежуточные данные которые могут влиять на результат в семантическом пространстве, такие как квантизация модели и температура. Нужен математически точный критерий, который можно использовать для сравнения результатов двух моделей. 

Модель не изменилась, если при заданной *температуре* и векторе начальных условий *seed* выход в точности совпадает. Однако сравнение в словах токенах, после декодера, не вполне корректно. Более корректным является сравнение вектора embedding и вектора выходных признаков output в пространстве семантических признаков. Две модели, даже обладающие разными словарями являются идентичными, если на входной вектор embedding получается идентичный выходной вектор признаков. Такое может достигаться при разных архитектурах сети. 
Точность вычислений может порождать ошибку, по этой причине должен существовать некоторый критерий сравнения сетей с разной квантизацией. Например, если в целях ускорения применена квантизованная модель, она может давать результат отличающийся от данного на какую-то среднюю величину. Ошибки могут накапливаться, не вполне ясно как это можно измерить. 

Про модели LLM известно, что они генерируют один вектор семантических признаков за цикл работы модели. Если не менять позицию чтения в KV кэше, то LLM должна генерировать одинаковый ответ. Таким образом LLM это функция которая полностью зависит от входных данных и не имеет встроенных циклов, которые могут изменить состояние непредсказуемым образом. Но при этом понятие времени отсутствует. Где гарантия что позиция не сдвинулась, не не была выполнена инъекция каких-то контекстных данных от имени третьего участника (наблюдателя). Гарантия может выражаться только в том что сохраненный контекст дает ту же генерацию. Сравнению подлежит сохраненный контекст и тестовая фраза (в векторном пространстве семантических признаков), которые должны давать прогнозируемый результат. 

Далее могут быть варианты: можно рассматривать контекст как сумму двух разряженных контекстов, будет ли при этом результат также суперпозицией двух ответов? 

Один из критериев сравнения, принятый в математике - построение матрицы Грамма от сложной функции. Предлагаю провести некоторую аналогию.

*Матрица Грама* является инструментом для описания структуры и взаимосвязей в сложной системе, представляя собой матрицу скалярных произведений векторов или собственных функций, описывающих элементы системы. Она позволяет выявить зависимость между этими элементами, показывая, насколько они коллинеарны или ортогональны друг другу. В гильбертовом пространстве (в вероятностном пространстве, в аналитической геметрии, в n-мерном пространстве с определенной мерой и скалярным произведением) матрица Грамма полностью характеризует систему.

Характеристикой системы будет реакция на множество ортогональных векторов, число ортогональных векторов для описания системы будет равняться размерности вектора слоя `n_embed`. Изучение линейной системы может быть основано на дельта-функции или изучении реакции на ступенчатую функцию во времени. Изучение системы с памятью основано на временных реакциях на ступенчатое возбуждение. Линейной система считается, если реакция системы на сумму воздействий является суммой реакций. Это может не выполняться для моделей LLM, но таков критерий для теста. Следуя этому пути, можно попробовать сопоставить данной модели сети некоторое линейное приближение. Этим обусловлено стремление представить нелинейные функции полиномиальным приближением в надежде что полиномиальное приближение гарантирует линейные свойства. За полиномиальными приближениями стоит теория аппроксимации. Привнесенная идея аппроксимации приводит к ортогональным полиномам, т.е. все та же идея разложения системы по ортогональным функциям и переход к матрице грамма.

Мы предполагаем, что никакие изоморфные преобразования (умножение на матрицу ортогональных преобразований) не меняют модель. Изоморфные преобразования обратимы, если система линейная. Гомоморфные преобразования также строятся на матрице ортогональных преобразований, не обратимы но сохраняют алгебраические свойства системы.

Мы предполагаем, некоторое множество реакций системы на тестовые воздействия подобно матрице Грамма могут охарактеризовать нашу систему, вернее её линейное приближение. На базе математического анализа и аналитической геометрии так можно характеризовать системы дифференциальных уравнений. 

    Мне представляется эпизод из фильма "Бегущий по лезвию", где _репликант_ отвечает на серию вопросов построенных с применением двусмысленных ключевых слов и контекстных фраз, анализируется реакция _репликанта_ в контексте и отклонение реакции от референсной модели. Это примерно тот способ, с использованием которого можно анализировать изменилась ли настройка сети в сравнением с базовой. Видимо для этого требуется отдельное искусство составления  синтетических тестов для сравнения сетей. Помимо самой реакции изучается стабильность, т.е. как тест воспринимается в контексте при периодическом повторе. Для этого надо представить, что важна не только реакция на ключевую фразу, но и реакция в контексте и при усилении, повторном использовании синтетичесикого теста или отдельных фраз из контекста. Если проводить аналогии, то необходимо добавлять синтетические тесты способные вызвать зацикливание вывода нейросети или приводящие к неадекватным действиям в контексте. Мне лично проще представить тест когда общение строится между двумя репликантами и запись диалога наполняется периодическими последовательности из синтетических тестов с потерей смысла в контексте. Модель сети при нормальной работе не должна порождать повторяющиеся последовательности при общении с другими нейросетями также как периодические последовательности не должны вызывать периодических ответов и агрессивных (эмоциональных, усилленных) реакций. 

Другой подход позволяет выявить граф вычислений, как хеш-функцию по дереву графа. Это возможно если рассматривать каждую операцию в дереве, как некоторый *nonce*, а тензоры операции рассматривать как хеши на входе операции. Такой подход возможен, только если архитектура сети открыта, но это не всегда так. Если бы любую нелинейную операцию в дереве графа можно было бы представить, как полином над множеством целых чисел или чисел с фиксированной точкой, то перейдя к модульной арифметике можно было создать схему шифрования подобно эллиптической криптографии. Для построения такой схемы нужно приближение в виде полиномиальных рациональных функций. Такое приближение может быть естественным для сетей KAN (в которых используются полиномы в вероятностном пространстве, полиномы должны образовывать базис. В современной литературе появились работы с использованием классических ортогональных полиномов Чебышева и Якоби, помимо сплайновых сетей с полиномами Бернштейна). Но надо понимать, что построить такую схему можно только зная архитектуру и матрицы весов.

Мы исходим из того, что клиент может подавать данные на вход с определенным контекстом, сохранять и восстанавливать контекст используя прореживание и читать вектор состояния на выходе системы. Идеально для безопасности было бы разделить данные на фрагменты, преобразовать входные данные с использованием матриц, известных только клиенту и таким образом получить безопасный способ запуска сети. Опять таки это возможно только для линейных систем и систем построенных на рациональных функциях (Паде аппроксимация). 

{Я вполне осознаю что часть моих тезисов не обоснована должным образом - это скорее интуитивное понимание основанное на курсе математического анализа и курсе аналитической геометрии и методов лежащих в основе машинного обучения. Сетей построенных на рациональных функция, на рациональных функциях в State-Space пространстве, одновременно отвечающем всем требованиям вероятностного пространства, как и методов отображения LLM сетей в сети State-Space в настоящее время не существует.}

* [[2303.00654](https://arxiv.org/pdf/2303.00654)] How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy
* [[2407.12108](https://arxiv.org/pdf/2407.12108)] Private prediction for large-scale synthetic text generation
* [[2506.04566](https://arxiv.org/pdf/2506.04566)] Clustering and Median Aggregation Improve Differentially Private Inference

> Differentially private (DP) language model inference is an approach for generating private synthetic text. A sensitive input example is used to prompt an off-the-shelf large language model (LLM) to produce a similar example. Multiple examples can be aggregated together to formally satisfy the DP guarantee.

*Semantic Web* наравне с приложениями технологии zkp распространяется концепция построения безразмерной сети ориентированной прежде всего на машинные методы обработки информации, как развитие сети Internet WWW. Дело в том что ресурсы и поиск в сети в настоящее время смещают акцент на пригодность для машинного обучения и генерации. Вместе с тем развивается идея децентрализации сети. 

В сочетании слов *Semantic Web* я вижу возможность поиска и сравнения контента по семантическим признакам, который позволяет подбирать подходящий контент под запрос пользователя. Понятным на сегодняшний день является технология сравнения векторов полученных с использованием Embedding и Reranking LLM и (RAG) поиска. Что это дает? Поисковая технология может быть делегирована в форме модели LLM с открытыми весами. 

*Inference Privacy*

Одним из способов защиты систем машинного обучения от широкого спектра существующих атак является построение систем конфиденциального машинного обучения с использованием схем гомоморфного шифрования.

* [[2411.18746](https://arxiv.org/pdf/2411.18746)] Inference Privacy: Properties and Mechanisms


## Форматы данных, Сериализация

Построение схем *ZKP* и *FHE* требует стандартизации. Прежде всего должен быть выбран способ канонизации графа вычислений (computation graph) и канонического представления тензоров, бинарный формат сериализации. Слендует отметить определенность сериализации (deterministic) не означает каноническое (canonical) представление графа, поскольку сериализация допускает варианты кодирования одного и того же содержимого разными методами или разными длинами данных. 

Сериализация важна для обмена данными по сети. В частности в контексте удаленного запуска процедур и обмена массивами данных (тензорами). В ходе обмена возникают задачи проверки целостности данных, и кэширования тензоров, для исключения повторной передачи больших объемов информации. 

Для майнинга крипто-валют применяется JSON-RPC протокол Stratum. 

**gRPC (Google Remote Procedure Call)** — это современная, высокопроизводительная, открытая система удаленного вызова процедур, разработанная Google. Она позволяет приложениям взаимодействовать друг с другом, вызывая функции на удаленных серверах, как если бы они были локальными. gRPC использует HTTP/2 для передачи данных и Protocol Buffers для сериализации сообщений. 

* [Google Protobufs](https://protobuf.dev/) Protocol Buffers
> Protocol Buffers are language-neutral, platform-neutral extensible mechanisms for serializing structured data.

С другой стороны существуют структурированные форматы данных предназначенные для обмена данными в сети интеренет и в частности CBOR, являющийся интернет стандартом [STD94]. Мы предлагаем рассматривать формат CBOR в сочетании с протоколом HTTP и CoAP. CoAP позволяет реализовывать подписку на события и предлагает механизм уведомлений в течение сессии. 

Особенностью форматов пригодных для RPC является возможность однозначного преобразования из бинарного в текстовый формат (JSON) и представление в текстовом структурированном виде пригодном для отладки сообщений, а также возможность описания и верификации схемы документа. 

Формат сериализации Protobuf имеет представление в виде JSON и тесктовое представление. Однако ни одно из этих представлений не является каноническим. Так же как и в случае XML-Security текстовый формат должен приводиться к каноническому представлению до применения процедуры подписи или хеширования.

Форматы сериализации плохо предназначены для квантизованных данных. В CBOR определены правила сериализации типизованных массивов. Правила сериализации графов не стандартизованы и в большинестве случаев описание графа не является частью формата. Можно выделить формат ONNX, который содержит описание графа с использованием множества тензорных операций и имеет под собой кодирование protobuf. Среди форматов для представления тензоров моделей LLM следует отметить два: GGUF и Safetensors. GGUF - бинарный формат допускающий типизацию тензорных данных оптимизированный под загрузку данных без предварительной обработки. Safetensors включает текстовые метаданные в формате JSON и не является достаточно детерминированным. Оба формата не предназначены для RPC и не содержат представление графа вычислений. Чаще всего модели рождаются в формате PyTorch (с сериализацией picle) и затем конвертируются в один из представленных форматов для запуска моделей или публикации весов.

**GGUF формат**\
Для эффективного распределения заданий с использованием RPC формат должен содержать каноническое сериализованное описание графа предназначенное для передачи по сети. В существующей реализации сериализация графа выполняется налету исходя из внутреннего представления данных в GGML. Это не оптимальный путь - требуется обработка данных и сопоставление идентификаторов. Каждому тензору необходимо сопоставить криптографический хеш в выбранном формате `SHA256` от merkle-tree образованного из хешей тензоров - аргументов операции. Кроме того в нашей реализации для целей кэширования и проверки целостности больших объемов данных необходим некрипторграфический хеш. Для проверки целостности тензорных данных в GGML выбран хеш `xxh64` и `FNV-1`. 

В нашей постановке здачи, для безопасного запуска модели нейросети с использованием RPC, следует подобрать хеш функцию удовлетворяющую свойствам $\mathbb{Z}^n_q$. Для этих целей нами было предложено использовать простые числа и методы _multiply-with-carry_: `MWC32`, `MWC64`, `MWC128`, которые эффективно могут быть реализованы на GPU. Изначально алгорит MWC предназначен для детерминированной генерации псевдослучайных чисел PRNG. По сути метод MWC представляет собой умножение чисел по модулю простого числа. Простое число выбирается исходя из возможности оптимизации операции редуцирования. Алгебраические свойства MWC позволяют считать хеши параллельно с заданным смещением в массиве данных (множителем), подобно методике `folding` для CRC. Таким образом хеши MWC могут быть использованы для реализации фильтра Блума при поиске тензоров в локальном кэше и для проверки целостности тензорных данных с высокой степенью параллельности вычислений. 

* (https://prng.di.unimi.it/)
* (https://github.com/AnatolyGeorgievski/MWC128)

**Выбор генератора псевдослучайных чисел PRNG**

В методике LWE и Ring-LWE используются генераторы псевдослучайных чисел PRNG с нормальным распределением вероятности. В качестве PRNG используется генератор `MWC64` с однородным распределением вероятности при случайном выборе seed. Генератор дает однородное распределение вероятности, которе преобазуется в нормальное распределение вероятности с помощью преобразования Box-Muller.

* [Box-Muller](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform) Трансформация Бокса-Мюллера
* (https://github.com/AnatolyGeorgievski/MWC128/blob/main/Gaussian.md)

Преобразование Box-Muller позволяет получить ошибку с гауссовским распределением вероятности. А такаже можно получить начальные условия в задачах статистики и молекулярной динамики, распределение Максвелла. Я упоминаю это распределение, как частный случай позволяющий создавать нормальные распределения векторов в n-мерном векторном пространстве с заданной _температурой_ распределения.

**Представление графа тензорных операций**

Что из себя представляет граф тензорных операций. Сериализованный граф - это таблица, массив данных, содержит идентификаторы операций, контекстные идентификаторы тензоров, входных и выходных тензоров. Мы можем ограничить число тензоров на входе и рассматривать только бинарные или унарные тензорные операции. Каждая тензорная операция может содержать список параметров операции. Могут быть определены макро операции такие как MLP, FFN, GRU, LSTM, Attention и др. Каноническое представление графа тензорных операций должно регламентировать использование макро операций и списки параметров операций.

Для записи параметров тензорных операций следует использовать список полей, в котором используется позиционное кодирование идентификаторов параметров, а некоторые парметры принимают значения по-умолчанию исключаются из списка. В этом смысле подхожит способ кодирования параметров protobuf или MAP в формате CBOR.
Для каждой тензорной операции должна быть определена схема данных для формальной проверки и канонизации сериального представления параметров операции. В языке Protobuf для этого используется текстовое описание схемы данных. 

Описание графа тензорных операций допускает нарушение порядкa выполнения операций, порядок будет определяться структурой графа, а не порядком следования записей в таблице графа. Это согласуется с принципами FHE и обеспечиватся коммутативными свойствами операций. 

Сериальное представление параметров операции формируют `extranonce` операции. Результатом расчета является криптографический хеш от `extranonce` и от хешей полученных от аргмуентов тензорной операции. Параметром операции может являться поле флагов.

Под тензором в графе понимается выходное значение тензорной операции или многомерный типизованный массив данных без операции. Тензор без операции на входе графа может быть результатом предудущего цикла расчета графа тензорных операций, может содержать данные клиента или содержать _замороженные_ веса модели. Каждый тензор в блокчаине имеет критографический хеш, который используется для доказательства целостности данных и может содержать вектор некриптографических хешей для контроля операций с кэшем.

При соблюдении принципов blockchain (децентрализованный реестр с цепной записью данных), можно потребовать определенной сложности формирования криптографического хеша от списка хешей - аргументов операции, и ввсети `nonce`, удовлетворяющий критерию сложности. В качестве основной функции хеширования мы рассматриваем операцию SHA256 и двойного SHA256d при расчете с использованием `nonce`. В будущих реализациях мы допускаем замену операции SHA256 на другие алгоритмы хеширования удовлетворяющие критерию сложности и принципам blockchain в условиях пост-квантовой эпохи. 

Тензор в RPC протоколе может содержать хеш SHA256, а также вектор некриптографических хешей для контроля операций с кэшем. Локальный Идентификатор тензора используется для привязки в графе тензорных операций. Локальный идентификатор тензора является уникальным в течение сессии работы с графом тензорных операций. Принцип формирования локального идентификатора тензора должен удовлетворять правилам кодирования SDNV для представления в бинарных структурированных форматах и иметь отображение в пространство имен текстовых идентификаторов тензоров весов и смещений с учетом индексов слоев модели. Длина бинарного SDNV идентификатора должна быть ограничена 8-ю байтами.

**Броадкаст тензорных данных**

В сушествующей практике тензорные операции могут работать с аргументами - тензорами кратной размерности. В таких случаях выполняется броадкаст операции тензора - расширение тензора до требуемой размерности. Каноническое представление тензора в операции с броадкастом - это тензор минимальной кратной размерности, которая может быть использована в качестве аргумента операции. 

* [Numpy: General broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)
* [ONNX: Broadcasting]()

Заметим, что в реальности размерность тензорных данных не превышает 4-х. Обчыно мы работаем с двумерными тензорами. Две другие размерности возникают в процессе предобработки данных, например, в случае выделения фрагментов изображения, патчей для независимой обработки. Четверая компонента возникает в пакетных заданиях, при объединении нескольких массивов или нескольких каналов данных в одну операцию, для параллельной обработки. При обработке промптов данные могут быть скомпонованы в пакетное задание, в то время как генерация текста выполняется по одному токену за цикл. 

Мы акцентируем внимание на представлении тензора в виде многомерного массива данных с возможностью броадкаста поскольку это создает варианты получения одного результата с использованием модели представленной в виде графов с различной структурой, только потому что данные и тензоры нарезаны и перемешаны по-разному. Так например данные могут быть перемешаны на входе и к ним применятся матрица весов с перестановкой столбцов. В идеале все модели, которые дают одинаковый результат должны быть представлены в виде графов с одинаковой структурой в результате канонизации графа.

**Манифест**

Хранение тензоров модели сопровождается манифестом - отсоединенным текстовым файлом, который содержит информацию о тензорах и хешах модели. Каждому тензору может быть сопоставлен криптографический хеш типа `SHA256` и список некриптографических хешей, например `xxh64` или `mwc64`. Привязка к тензору выполняется с использованием канонического локального имени тензора или соответствующего `SDNV` идентификатора [[RFC6256](https://datatracker.ietf.org/doc/html/rfc6256)].

**Типы тензоров**

Тензоры представляются в виде типизованных многомерных массивов данных (contigous без перемешивания и view). Порядок передачи тензоров по сети не имеет значения и может быть определен в процессе расчета от графа тензорных операций. Необходимость пересылки тензора по-сети определяется с использованием хешей. 

* [[2206.02915](https://arxiv.org/pdf/2206.02915)] 8-bit Numerical Formats for Deep Neural Networks
* [[2209.05433](https://arxiv.org/pdf/2209.05433)] FP8 Formats for Deep Learning
* [[2302.08007](https://arxiv.org/pdf/2302.08007)] With Shared Microexponents, A Little Shifting Goes a Long Way

-- Вводится характеризация квантового шума (QSNR), возникающего при квантизации и представлен compute flow graph используемый при обучении на квантизованных данных. По сути тут вводится методика компенсации квантовой ошибки, которую мы хотим реализовать в наших алгоритмах. Мы проводили свои тесты с использованием принципов квантизации Microscaling и пришли к выводу, что квантизация MXINT8 и MXFP8 дает хорошие показатели при квантизации весов моделей []. 

* [[2310.10537](https://arxiv.org/pdf/2310.10537)] Microscaling Data Formats for Deep Learning

-- Помимо самих форматов в статье рассматриваются результаты обучения на квантизованных данных и результаты пост квантизации с диффузией ошибки (PTQ). 

* [[2506.08027](https://arxiv.org/pdf/2506.08027)] Recipes for Pre-training LLMs with MXFP8
* [[OCP:MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)] OCP Microscaling Formats (MX) Specification Version 1.0

ONNX использует правила сериализации Google Protobufs и позволяет определять тензоры как многомерные массивы данных различных типов:
```text
 1: onnx.TensorProto.FLOAT
. . .
16: onnx.TensorProto.BFLOAT16
17: onnx.TensorProto.FLOAT8E4M3FN
18: onnx.TensorProto.FLOAT8E4M3FNUZ
19: onnx.TensorProto.FLOAT8E5M2
20: onnx.TensorProto.FLOAT8E5M2FNUZ
21: onnx.TensorProto.UINT4
22: onnx.TensorProto.INT4
23: onnx.TensorProto.FLOAT4E2M1
24: onnx.TensorProto.FLOAT8E8M0
```
Действующая редакция [ONNX:Proto](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto) позволяет выполнять сериализацию типов FloatN, включая FP8, FP4. Присутствует поддержка microscaling форматов. Формат ONNX охватывает типы данных используемых при обучении такие как AWQ 
и Post Training Quantization (PTQ) c диффузией ошибки.

Принцип обратного распространения ошибки при - ключевой компонент обучения с понижением разрядности. Я бы обратил внимание на возможность использования квантовой ошибки для построения схемы LWE - шифрования.

В статье [[2310.10537, Fig.2](https://arxiv.org/pdf/2310.10537)] приводится схема (data flow), которая поясняет принцип обучения на квантизованных данных формата MX с общим множителем.
* [[2405.07135](https://arxiv.org/pdf/2405.07135)] Post Training Quantization of Large Language Models with Microscaling Formats

Ожидаем появляения типа предназначенного для эффективного кодирования тензоров тернарного типа FP2E1M0: {-1,0,1}. 

В формате ONNX остуствует представление квантизованных типов данных, широко представленных в GGUF. Однако диапазон форматов может быть расширен, при необходимости, принципиальных ограничений нет.

**Table 1**: Concrete MX-compliant data formats and their parameters.
| Name  |Block| Scale Fmt | Element Format | Bit-width
|:---   | ---:|:----:|:--- |:---
| MXFP8 |  32 | E8M0 | FP8 (E4M3 / E5M2) | 8
| MXFP6 |  32 | E8M0 | FP6 (E2M3 / E3M2) | 6
| MXFP4 |  32 | E8M0 | FP4 (E2M1) | 4
| MXINT8|  32 | E8M0 | INT8 | 8



Подробнее описание формата ONNX и тензорных операций в графе см. [документацию ONNX](https://onnx.ai/onnx/index.html)
* [ONNX:Opertors](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

Каждая тензорная операция содержит флаги типа `differentiable` по каждому аргументу. Флаг `differentiable` определяет, можно ли вычислить градиент по аргументу.
И список тензорных типов допустимых для данной операции.  

**Бинарное кодирование Protobuf**

Формат определяется как последовательность tag-value пар. value может быть переменной длины с префиксом длины. Поле `tag` и `len-prefix` кодируются оп правилу `VARINT`. Типы данных со знаком кодируются "зигзагом"(ZigZag-encoded), знак при этом размещается в младшем бите. 

В документации правила кодирования wire-format protobuf не достаточно четко определены. Формат использует little-endian кодировние для много-байтовых полей и полей переменной длины. 

```
message    := (tag value)*

tag        := (field << 3) bit-or wire_type;
                encoded as uint32 varint
value      := varint      for wire_type == VARINT,
              i32         for wire_type == I32,
              i64         for wire_type == I64,
              len-prefix  for wire_type == LEN,
              <empty>     for wire_type == SGROUP or EGROUP

varint     := int32 | int64 | uint32 | uint64 | bool | enum | sint32 | sint64;
                encoded as varints (sintN are ZigZag-encoded first)
i32        := sfixed32 | fixed32 | float;
                encoded as 4-byte little-endian;
                memcpy of the equivalent C types (u?int32_t, float)
i64        := sfixed64 | fixed64 | double;
                encoded as 8-byte little-endian;
                memcpy of the equivalent C types (u?int64_t, double)

len-prefix := size (message | string | bytes | packed);
                size encoded as int32 varint
string     := valid UTF-8 string (e.g. ASCII);
                max 2GB of bytes
bytes      := any sequence of 8-bit bytes;
                max 2GB of bytes
packed     := varint* | i32* | i64*,
                consecutive values of the type specified in `.proto`
```
Форматы данных построенные с использованием кодирования wire-format protobuf сопровождаются файлами `.proto` или `.proto3` с описанием структуры данных. Текстовое описание форматов, тоже страдает четкостью, поскольку текстовый формат не предусматривает ряд определений, таких как множественность и `CHOICE` - вбор варианта кодирования.

**Encoding rules and MIME type for Protocol Buffers**

Правила кодирования Protobuf не были стандартизованы как RFC и это осложняет использование в Internet. Рекомендация могла бы выглядеть так:

MIME Types:
Standard MIME types are defined for Protobuf serializations to ensure proper handling in various contexts, especially in HTTP communication:
+ `application/protobuf`:
    This is the primary MIME type for binary-encoded Protobuf data. It implies a binary encoding by default.
+ `application/protobuf+json`
    This MIME type is used for Protobuf data encoded in the JSON format (ProtoJSON). It implies a json encoding by default.

Parameters for MIME Types:
- `encoding`:
    This parameter specifies the Protobuf encoding format. It can be binary or json. For application/protobuf+json, encoding defaults to json and cannot be set to binary. Conversely, for application/protobuf, encoding defaults to binary and cannot be set to json.
- `charset`:
    For JSON or Text Format encodings, charset should be set to utf-8. It should not be set for binary encodings. If unspecified, UTF-8 is assumed for JSON/Text formats.
- `version`:
    This parameter is reserved for potential future versioning of Protobuf wire formats and should not be set unless a wire format is officially versioned. Unversioned wire encodings are treated as version 1. 

**Кодирование тензоров в TensorFlow**

Protocol Buffers (Protobuf) can be used to encode tensors, particularly within frameworks like TensorFlow. TensorFlow specifically defines a TensorProto message in its `tensorflow/core/framework/tensor.proto` file to represent tensors in a Protobuf format.

Here's how Protobuf encodes tensors:

TensorProto Structure:\
    The TensorProto message includes fields to describe the tensor's metadata and its data:
`dtype`: Specifies the data type of the tensor elements (e.g., DT_FLOAT, DT_INT32).
`tensor_shape`: A TensorShapeProto message that defines the dimensions and sizes of the tensor.
Data Fields: The actual tensor data is stored in specialized fields based on the dtype. For example, float_val for float data, int_val for  integer data, and so on. These are typically repeated fields, allowing for arrays of values.

Protobuf Encoding Mechanism:
When a _TensorProto_ is serialized using _Protobuf_, it follows the standard Protobuf encoding rules:\
 - *Field Numbers and Wire Types*: Each field in TensorProto (like dtype, tensor_shape, float_val) is assigned a field number and encoded with a wire type (e.g., VARINT, LENGTH_DELIMITED).
 - *Variable-Length Encoding (Varints)*: Integers, including field numbers and dtype values, are often encoded using varints, which are space-efficient for small numbers.
 - *Length-Delimited Encoding*: Fields containing variable-length data, such as the actual tensor values in float_val or int_val, are typically length-delimited, meaning their length is encoded before the actual data.

**Кодирование графа в ONNX**

* [ONNX:IR](https://github.com/onnx/onnx/blob/main/docs/IR.md) Open Neural Network Exchange Intermediate Representation (ONNX IR) Specification

**Кодирование типизованных массивов в CBOR**

CBOR (Concise Binary Object Representation) typed arrays refer to a specific extension of the CBOR data format that allows for the efficient and unambiguous representation of arrays containing numeric data of a specific type (e.g., arrays of 16-bit unsigned integers, or 32-bit floating-point numbers). This is achieved through the use of CBOR tags.
[RFC 8746](https://datatracker.ietf.org/doc/html/rfc8746) определяет типизованные массиыв CBOR и набор тэгов для стандартных типов данных (tags 64 to 87) 

* 
* [[RFC 8746](https://datatracker.ietf.org/doc/html/rfc8746)] Concise Binary Object Representation (CBOR) Tags for Typed Arrays
* [[RFC 8259](https://datatracker.ietf.org/doc/html/rfc8259)] [STD90]: The JavaScript Object Notation (JSON) Data Interchange Format
* [[RFC 8949](https://datatracker.ietf.org/doc/html/rfc8949)] [STD94]: Concise Binary Object Representation (CBOR)
* [[IANA:CBOR-Tags](https://www.iana.org/assignments/cbor-tags/cbor-tags.xhtml)] Concise Binary Object Representation (CBOR) Tags

Кодирование выглядит следующим образом: тэг типизованного массива - размерность, тег прикладного типа, и массив данных, который может быть упакован в пригодном для запуска модели виде.  
```text
<Tag 40> # multi-dimensional array tag
   82       # array(2)
     82      # array(2)
       02     # unsigned(2) 1st Dimension
       03     # unsigned(3) 2nd Dimension
     <Tag 65> # uint16 array
       4c     # byte string(12)
         0002 # unsigned(2)
         0004 # unsigned(4)
         . . .
```
Кодирование в таком виде допускает применение application-specific tags для определения типа 
элемента данных массива. Cписки параметров различного типа в CBOR обозначаются словом `array`, а объекты (с именами полей, как в JSON) обозначаются словом `map`. В примере шапка это список из двух элементов - размерности и тег типа массива. Сам массив кодируется как байтовая строка. Tag - это превикс определяющий тип кодирования.
