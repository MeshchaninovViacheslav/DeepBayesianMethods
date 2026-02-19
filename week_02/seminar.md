# Seminar: Doubly Stochastic Variational Inference

## 1. Log-derivative trick vs. Reparameterization trick

### Task Definition
Let

- $x \sim q(x\mid \theta)$,
- $f(x)$ be a scalar objective,

and consider

$$
J(\theta) = \mathbb E_{x\sim q(x\mid\theta)}[f(x)].
$$

We want to compute $\nabla_\theta J(\theta)$ and construct a Monte-Carlo stochastic gradient estimator.

---

### 1.1 Log-derivative trick (score-function)

Start from

$$
\nabla_\theta J(\theta) = \nabla_\theta \int q(x\mid\theta) f(x)\,dx.
$$

Assuming regularity conditions (interchange derivative and integral),

$$
\nabla_\theta J(\theta)
= \int \nabla_\theta q(x\mid\theta)\, f(x)\,dx.
$$

Use the identity

$$
\nabla_\theta q(x\mid\theta) = q(x\mid\theta)\,\nabla_\theta \log q(x\mid\theta).
$$

Then

$$
\nabla_\theta J(\theta)
= \int q(x\mid\theta)\, \nabla_\theta \log q(x\mid\theta)\, f(x)\,dx
= \mathbb E_{x\sim q(x\mid\theta)}\big[f(x)\,\nabla_\theta \log q(x\mid\theta)\big].
$$

**Monte-Carlo estimator (1 sample):**

$$
\widehat{g}_{\text{LDT}} = f(x)\,\nabla_\theta \log q(x\mid\theta),
\quad x\sim q(x\mid\theta).
$$

---

### 1.2 Reparameterization trick (pathwise)

Assume we can sample from $q(x\mid\theta)$ via a differentiable transformation:

$$
\varepsilon \sim p(\varepsilon) \quad (\text{independent of }\theta),
\qquad x = g(\varepsilon,\theta).
$$

Then

$$
J(\theta) = \mathbb E_{\varepsilon\sim p(\varepsilon)}\big[f(g(\varepsilon,\theta))\big].
$$

Differentiate inside the expectation:

$$
\nabla_\theta J(\theta)
= \mathbb E_{\varepsilon\sim p(\varepsilon)}\big[\nabla_\theta f(g(\varepsilon,\theta))\big]
= \mathbb E_{\varepsilon}\Big[\nabla_x f(x)\,\frac{\partial x}{\partial \theta}\Big]_{x=g(\varepsilon,\theta)}.
$$

**Monte-Carlo estimator (1 sample):**

$$
\widehat{g}_{\text{RT}} = \nabla_\theta f(g(\varepsilon,\theta)),
\quad \varepsilon\sim p(\varepsilon).
$$

## 2. Variance comparison on a Gaussian toy problem

### Task Definition
Let

$$
X \sim \mathcal N(\mu,1),
\qquad f(x) = x^2,
\qquad J(\mu) = \mathbb E[X^2].
$$

We want to optimize $J(\mu)$ w.r.t. $\mu$ using SGD, and compare the variance of gradient estimators.

Note that analytically

$$
J(\mu) = \mu^2 + 1
\qquad \frac{d}{d\mu}J(\mu)=2\mu
$$

---

### 2.1 LDT estimator

For a Normal with unit variance,

$$
\frac{\partial}{\partial\mu}\log \mathcal N(x\mid\mu,1) = x-\mu.
$$

So the 1-sample LDT gradient estimator is

$$
\widehat g_{\text{LDT}} = x^2(x-\mu),\quad x\sim\mathcal N(\mu,1).
$$

Compute the variance

$$
\operatorname{Var}(\widehat g_{\text{LDT}})
= \mathbb E\big[x^4(x-\mu)^2\big] - \big(\mathbb E[x^2(x-\mu)]\big)^2.
$$

Write $x = \mu + z$ with $z\sim\mathcal N(0,1)$. Then

$$
\widehat g_{\text{LDT}} = x^2(x-\mu) = (\mu+z)^2 z.
$$

Expand:

$$
(\mu+z)^2 z = (\mu^2 + 2\mu z + z^2)z = \mu^2 z + 2\mu z^2 + z^3.
$$

Unbiasedness (check the mean): using $\mathbb E[z]=0$, $\mathbb E[z^2]=1$, $\mathbb E[z^3]=0$,

$$
\mathbb E[\widehat g_{\text{LDT}}]
= \mu^2\mathbb E[z] + 2\mu\mathbb E[z^2] + \mathbb E[z^3]
= 2\mu.
$$

Second moment:

$$
\widehat g_{\text{LDT}}^2 = (\mu^2 z + 2\mu z^2 + z^3)^2
= \mu^4 z^2 + 4\mu^3 z^3 + 6\mu^2 z^4 + 4\mu z^5 + z^6.
$$

Now use standard normal moments:
$\mathbb E[z^2]=1$, $\mathbb E[z^3]=0$, $\mathbb E[z^4]=3$, $\mathbb E[z^5]=0$, $\mathbb E[z^6]=15$.
Hence

$$
\mathbb E[\widehat g_{\text{LDT}}^2]
= \mu^4\cdot 1 + 4\mu^3\cdot 0 + 6\mu^2\cdot 3 + 4\mu\cdot 0 + 15
= \mu^4 + 18\mu^2 + 15.
$$

Finally,

$$
\operatorname{Var}(\widehat g_{\text{LDT}})
= \mathbb E[\widehat g_{\text{LDT}}^2] - (\mathbb E[\widehat g_{\text{LDT}}])^2
= (\mu^4 + 18\mu^2 + 15) - (2\mu)^2
= \mu^4 + 14\mu^2 + 15.
$$

---

### 2.2 Baseline for variance reduction

We may subtract a constant baseline $b$:

$$
\widehat g = (f(x)-b)\,\nabla_\mu \log q(x\mid\mu).
$$

Since $\mathbb E[x-\mu]=0$, the expectation of the estimator does not change, while the variance may. Let's consider $b=\mu^2$:

$$
\widehat g_{\text{LDT,base}} = (x^2-\mu^2)(x-\mu).
$$

Reparameterize $x=\mu+z$, $z\sim\mathcal N(0,1)$:

$$
\widehat g_{\text{LDT,base}}
= \big((\mu+z)^2-\mu^2\big)z
= (2\mu z+z^2)z
= 2\mu z^2 + z^3.
$$

Unbiasedness:

$$
\mathbb E[\widehat g_{\text{LDT,base}}]
= 2\mu\mathbb E[z^2] + \mathbb E[z^3]
= 2\mu.
$$

Second moment:

$$
\widehat g_{\text{LDT,base}}^2
= (2\mu z^2 + z^3)^2
= 4\mu^2 z^4 + 4\mu z^5 + z^6.
$$

Using $\mathbb E[z^4]=3$, $\mathbb E[z^5]=0$, $\mathbb E[z^6]=15$:

$$
\mathbb E[\widehat g_{\text{LDT,base}}^2]
= 4\mu^2\cdot 3 + 15
= 12\mu^2 + 15.
$$

Therefore

$$
\operatorname{Var}(\widehat g_{\text{LDT,base}})
= (12\mu^2 + 15) - (2\mu)^2
= 8\mu^2 + 15.
$$

The baseline removes the $\mu^4$ term but the variance is still much larger than the reparameterization estimator.

---

### 2.3 Reparameterization estimator

Reparameterize:

$$
X = \mu + \varepsilon,\quad \varepsilon\sim\mathcal N(0,1).
$$

Then

$$
\widehat g_{\text{RT}} = \frac{\partial}{\partial\mu}(\mu+\varepsilon)^2 = 2(\mu+\varepsilon).
$$

Its variance is

$$
\operatorname{Var}(\widehat g_{\text{RT}})=\operatorname{Var}(2\varepsilon)=4.
$$


---

## 3. Discrete example: Bernoulli and why RT breaks

### Task Definition
Let

$$
X\sim\operatorname{Bern}(\theta),\qquad f(x)=x,
\qquad J(\theta)=\mathbb E[X]=\theta.
$$

We try to compute $\nabla_\theta J(\theta)$ using RT and see what goes wrong.

---

### 3.1 Attempted reparameterization via thresholding

A common sampling scheme is

$$
U\sim\mathcal U[0,1],\qquad X = \mathbb I[U < \theta].
$$

Then formally

$$
\nabla_\theta J(\theta) = \nabla_\theta\,\mathbb E_U\big[\mathbb I[U<\theta]\big].
$$

But the mapping $x=\mathbb I[U<\theta]$ is **discontinuous** in $\theta$.

For almost every $U$, the derivative is zero:

$$
\frac{\partial}{\partial\theta}\,\mathbb I[U<\theta]=0\quad\text{a.e.}
$$

So a naive pathwise gradient yields a **zero gradient** (and is useless).

---

### 3.2 What to do instead

- Use **LDT / REINFORCE** (unbiased but high variance).
- Or use a **continuous relaxation** (e.g., Gumbel-Softmax).

This issue is one reason classical GAN training and discrete latent-variable models require additional tricks.

---

## 4. RT via inverse CDF and implicit differentiation

### Task Definition
We want a more general reparameterization for 1D continuous distributions.

Let $F_\theta$ be the CDF of $q(x\mid\theta)$.

---

### 4.1 Inverse-CDF reparameterization

If we can compute the inverse CDF, we can sample as

$$
U\sim\mathcal U[0,1],\qquad X = F_\theta^{-1}(U).
$$

Then

$$
\nabla_\theta\,\mathbb E[f(X)]
= \mathbb E_U\Big[\nabla_x f(X)\,\frac{\partial}{\partial\theta}F_\theta^{-1}(U)\Big].
$$

A direct numerical derivative of $F_\theta^{-1}$ via finite differences can be unstable, especially near the tails.

---

### 4.2 Implicit differentiation (more stable)

Instead of differentiating the inverse explicitly, use the defining equation

$$
U = F_\theta(X).
$$

Differentiate both sides w.r.t. $\theta$:

$$
0 = \frac{\partial}{\partial\theta}F_\theta(X) + \frac{\partial}{\partial x}F_\theta(X)\,\frac{\partial X}{\partial\theta}.
$$

Solve for $\partial X/\partial\theta$:

$$
\frac{\partial X}{\partial\theta}
= -\frac{\partial_\theta F_\theta(X)}{\partial_x F_\theta(X)}.
$$

Since $\partial_x F_\theta(X)=q(X\mid\theta)$ (the PDF), let us obtain the final gradient formula:

$$
\nabla_\theta\,\mathbb E[f(X)]
= \mathbb E\Big[\nabla_x f(X)\,
\Big(-\frac{\partial_\theta F_\theta(X)}{q(X\mid\theta)}\Big)\Big].
$$

---