# Seminar: Statistical Proofs

## 1. Cramér-Rao (Rao-Cramer) Bound

### Task Definition

Let $X$ be a random variable with density $p(x \mid \theta)$, where $\theta$ is a parameter. Let $\hat{\theta}(X)$ be an unbiased estimator of $\theta$, i.e., $\mathbb{E}_X[\hat{\theta}(X)] = \theta$.

Define:
- **Score function**: $U(x, \theta) = \frac{\partial \log p(x \mid \theta)}{\partial \theta}$
- **Fisher information**: $I(\theta) = \mathbb{E}_X[U^2(x, \theta)] = \text{Var}(U)$

**Assumptions:**
- $\text{Var}(U) < \infty$ (equivalently, $I(\theta) < \infty$)
- Regularity conditions allow differentiation under the integral sign

**Statement:** For any unbiased estimator $\hat{\theta}(X)$ of $\theta$:

$$I(\theta) \cdot \text{Var}(\hat{\theta}) \geq 1$$

or equivalently:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

### Proof

**Step 1: Properties of the score function** (was on lecture)

Since $\int p(x \mid \theta) dx = 1$, differentiating both sides with respect to $\theta$:

$$\frac{\partial}{\partial \theta} \int p(x \mid \theta) dx = \int \frac{\partial p(x \mid \theta)}{\partial \theta} dx = 0$$

Rewriting using $\frac{\partial p}{\partial \theta} = p \cdot \frac{\partial \log p}{\partial \theta} = p \cdot U$:

$$\int U(x, \theta) \cdot p(x \mid \theta) dx = \mathbb{E}_X[U(x, \theta)] = 0$$

**Step 2: Variance of the estimator**

Since $\hat{\theta}(X)$ is unbiased, we have:

$$\mathbb{E}_X[\hat{\theta}(X)] = \int \hat{\theta}(x) \cdot p(x \mid \theta) dx = \theta$$

Differentiating both sides with respect to $\theta$:

$$\frac{\partial}{\partial \theta} \int \hat{\theta}(x) \cdot p(x \mid \theta) dx = 1$$

$$\int \hat{\theta}(x) \cdot \frac{\partial p(x \mid \theta)}{\partial \theta} dx = 1$$

$$\int \hat{\theta}(x) \cdot U(x, \theta) \cdot p(x \mid \theta) dx = 1$$

$$\mathbb{E}_X[\hat{\theta}(X) \cdot U(x, \theta)] = 1$$

**Step 3: Apply Cauchy-Schwarz inequality**

Since $\mathbb{E}[U] = 0$, we have:

$$\text{Cov}(\hat{\theta}, U) = \mathbb{E}[\hat{\theta} \cdot U] - \mathbb{E}[\hat{\theta}] \cdot \mathbb{E}[U] = 1 - \theta \cdot 0 = 1$$

By the Cauchy-Schwarz inequality:

$$|\text{Cov}(\hat{\theta}, U)|^2 \leq \text{Var}(\hat{\theta}) \cdot \text{Var}(U)$$

Substituting our results:

$$1 = |\text{Cov}(\hat{\theta}, U)|^2 \leq \text{Var}(\hat{\theta}) \cdot I(\theta)$$

Therefore:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

or equivalently:

$$I(\theta) \cdot \text{Var}(\hat{\theta}) \geq 1$$

**Conclusion:** The variance of any unbiased estimator is bounded below by the reciprocal of the Fisher information. This lower bound is called the **Cramér-Rao lower bound (CRLB)**. $\square$

---

## 2. Equivalence of Fisher Information Definitions

### Task Definition

Let $X$ be a random variable with density $p(x \mid \theta)$, where $\theta \in \mathbb{R}$ is a scalar parameter.

Define the **score function**: $U(x, \theta) = \frac{\partial \log p(x \mid \theta)}{\partial \theta}$

**Two definitions of Fisher Information:**

**Definition 1** (Square of gradient):
$$I(\theta) = \mathbb{E}_X\left[\left(\frac{\partial \log p(x \mid \theta)}{\partial \theta}\right)^2\right] = \mathbb{E}_X[U^2(x, \theta)]$$

**Definition 2** (Negative expected second derivative):
$$I(\theta) = -\mathbb{E}_X\left[\frac{\partial^2 \log p(x \mid \theta)}{\partial \theta^2}\right]$$

**Statement:** Under regularity conditions, these two definitions are equivalent.

### Proof

**Step 1: Expectation of the score is zero** (was on lecture)

From the normalization condition $\int p(x \mid \theta) dx = 1$, differentiate with respect to $\theta$:

$$\frac{\partial}{\partial \theta} \int p(x \mid \theta) dx = \int \frac{\partial p(x \mid \theta)}{\partial \theta} dx = 0$$

Using $\frac{\partial p}{\partial \theta} = p \cdot \frac{\partial \log p}{\partial \theta}$:

$$\mathbb{E}_X\left[\frac{\partial \log p}{\partial \theta}\right] = \int \frac{\partial \log p}{\partial \theta} \cdot p(x \mid \theta) dx = 0$$

**Step 2: Differentiate again**

Differentiate $\mathbb{E}_X\left[\frac{\partial \log p}{\partial \theta}\right] = 0$ with respect to $\theta$:

$$\frac{\partial}{\partial \theta} \int \frac{\partial \log p}{\partial \theta} \cdot p(x \mid \theta) dx = 0$$

Apply the product rule:

$$\int \left[\frac{\partial^2 \log p}{\partial \theta^2} \cdot p + \frac{\partial \log p}{\partial \theta} \cdot \frac{\partial p}{\partial \theta}\right] dx = 0$$

Substitute $\frac{\partial p}{\partial \theta} = p \cdot \frac{\partial \log p}{\partial \theta}$:

$$\int \frac{\partial^2 \log p}{\partial \theta^2} \cdot p \, dx + \int \left(\frac{\partial \log p}{\partial \theta}\right)^2 \cdot p \, dx = 0$$

**Step 3: Obtain equivalence**

Rewrite in terms of expectations:

$$\mathbb{E}_X\left[\frac{\partial^2 \log p}{\partial \theta^2}\right] + \mathbb{E}_X\left[\left(\frac{\partial \log p}{\partial \theta}\right)^2\right] = 0$$

Therefore:

$$\mathbb{E}_X\left[\left(\frac{\partial \log p}{\partial \theta}\right)^2\right] = -\mathbb{E}_X\left[\frac{\partial^2 \log p}{\partial \theta^2}\right]$$

**Conclusion:** The two definitions are equivalent:

$$I(\theta) = \mathbb{E}_X[U^2] = -\mathbb{E}_X\left[\frac{\partial^2 \log p}{\partial \theta^2}\right]$$

This shows that Fisher Information can be computed either as the expected square of the score or as the negative expected second derivative of the log-likelihood. $\square$

## 3. Fisher Information for Exponential Family

### Task Definition

Consider the **exponential family** of distributions with scalar parameter $\theta \in \mathbb{R}$:

$$p(x \mid \theta) = \frac{1}{h(\theta)} g(x) \exp(\theta \cdot u(x))$$

where:
- $g(x) \geq 0$ is the base measure
- $u(x)$ is the sufficient statistic
- $h(\theta) = \int g(x) \exp(\theta \cdot u(x)) dx$ is the normalization constant (partition function)

**Statement:** For the exponential family, the Fisher information has the following equivalent forms:

$$I(\theta) = \text{Var}(u(X)) = \frac{d^2 \log h(\theta)}{d\theta^2}$$

### Proof

**Step 1: Compute the log-likelihood and score**

The log-likelihood is:

$$\log p(x \mid \theta) = \log g(x) + \theta \cdot u(x) - \log h(\theta)$$

The score function is:

$$\frac{\partial \log p}{\partial \theta} = u(x) - \frac{d \log h(\theta)}{d\theta}$$

**Step 2: Compute expectation of u(x)**

Since $\mathbb{E}\left[\frac{\partial \log p}{\partial \theta}\right] = 0$ (from proof 2):

$$\mathbb{E}[u(X)] - \frac{d \log h(\theta)}{d\theta} = 0$$

Therefore:

$$\mathbb{E}[u(X)] = \frac{h'(\theta)}{h(\theta)} = \frac{d \log h(\theta)}{d\theta}$$

**Step 3: Fisher information as variance of u(x)**

Using Definition 1 of Fisher information:

$$I(\theta) = \mathbb{E}\left[\left(\frac{\partial \log p}{\partial \theta}\right)^2\right]$$

Since $\mathbb{E}\left[\frac{\partial \log p}{\partial \theta}\right] = 0$:

$$I(\theta) = \text{Var}\left(\frac{\partial \log p}{\partial \theta}\right) = \text{Var}\left(u(X) - \frac{d \log h(\theta)}{d\theta}\right)$$

Since $\frac{d \log h(\theta)}{d\theta}$ is a constant (doesn't depend on $X$):

$$I(\theta) = \text{Var}(u(X))$$

**Step 4: Fisher information as second derivative of log h**

Using Definition 2 of Fisher information:

$$I(\theta) = -\mathbb{E}\left[\frac{\partial^2 \log p}{\partial \theta^2}\right]$$

Compute the second derivative:

$$\frac{\partial^2 \log p}{\partial \theta^2} = \frac{\partial}{\partial \theta}\left[u(x) - \frac{d \log h(\theta)}{d\theta}\right] = -\frac{d^2 \log h(\theta)}{d\theta^2}$$

Therefore:

$$I(\theta) = -\mathbb{E}\left[-\frac{d^2 \log h(\theta)}{d\theta^2}\right] = \frac{d^2 \log h(\theta)}{d\theta^2}$$

**Alternative derivation using h(θ):**

From Step 2, $\mathbb{E}[u(X)] = \frac{h'(\theta)}{h(\theta)}$. Differentiate with respect to $\theta$:

$$\frac{d}{d\theta}\mathbb{E}[u(X)] = \frac{d}{d\theta}\left[\frac{h'(\theta)}{h(\theta)}\right]$$

The left side gives $\text{Var}(u(X))$ (by differentiating $\int u(x) p(x|\theta) dx$ and using the score).

The right side gives:

$$\frac{h''(\theta) \cdot h(\theta) - (h'(\theta))^2}{(h(\theta))^2} = \frac{d^2 \log h(\theta)}{d\theta^2}$$

**Conclusion:** For exponential families:

$$I(\theta) = \text{Var}(u(X)) = \frac{d^2 \log h(\theta)}{d\theta^2}$$

This elegant result shows that in exponential families, the Fisher information equals the variance of the sufficient statistic and can be computed directly from the log-partition function. $\square$

## 4. KL Divergence as a Local Metric (Natural Gradient)

### Task Definition

Let $p(x \mid \theta)$ be a parametric family of distributions, where $\theta \in \mathbb{R}$ is a scalar parameter.

The **Kullback-Leibler (KL) divergence** from $p(x \mid \theta)$ to $p(x \mid \theta + \Delta\theta)$ is:

$$D_{KL}(p(\cdot \mid \theta) \| p(\cdot \mid \theta + \Delta\theta)) = \mathbb{E}_{x \sim p(\cdot|\theta)}\left[\log \frac{p(x \mid \theta)}{p(x \mid \theta + \Delta\theta)}\right]$$

Let $I(\theta)$ be the Fisher Information:

$$I(\theta) = \mathbb{E}_X\left[\left(\frac{\partial \log p}{\partial \theta}\right)^2\right]$$

**Statement:** For small $\Delta\theta$, the KL divergence has the following second-order Taylor expansion:

$$D_{KL}(p(\cdot \mid \theta) \| p(\cdot \mid \theta + \Delta\theta)) = \frac{1}{2} I(\theta) (\Delta\theta)^2 + O(|\Delta\theta|^3)$$

This shows that the Fisher Information defines a **Riemannian metric** on the parameter space, which is the foundation of **natural gradient** methods.

### Proof

**Step 1: Taylor expansion of log-likelihood**

Expand $\log p(x \mid \theta + \Delta\theta)$ around $\theta$ using Taylor series:

$$\log p(x \mid \theta + \Delta\theta) = \log p(x \mid \theta) + \frac{\partial \log p}{\partial \theta}\bigg|_\theta \Delta\theta + \frac{1}{2} \frac{\partial^2 \log p}{\partial \theta^2}\bigg|_\theta (\Delta\theta)^2 + O(|\Delta\theta|^3)$$

**Step 2: Compute KL divergence**

Substitute into the KL divergence:

$$D_{KL} = \mathbb{E}_X\left[\log p(x \mid \theta) - \log p(x \mid \theta + \Delta\theta)\right]$$

$$= \mathbb{E}_X\left[-\frac{\partial \log p}{\partial \theta} \Delta\theta - \frac{1}{2} \frac{\partial^2 \log p}{\partial \theta^2} (\Delta\theta)^2\right] + O(|\Delta\theta|^3)$$

$$= -\Delta\theta \cdot \mathbb{E}_X\left[\frac{\partial \log p}{\partial \theta}\right] - \frac{(\Delta\theta)^2}{2} \mathbb{E}_X\left[\frac{\partial^2 \log p}{\partial \theta^2}\right] + O(|\Delta\theta|^3)$$

**Step 3: Apply known results**

From earlier proofs, we know:

1. $\mathbb{E}_X\left[\frac{\partial \log p}{\partial \theta}\right] = 0$ (score has zero expectation)
2. $\mathbb{E}_X\left[\frac{\partial^2 \log p}{\partial \theta^2}\right] = -I(\theta)$ (from proof 2)

Substituting:

$$D_{KL} = -\Delta\theta \cdot 0 - \frac{(\Delta\theta)^2}{2} \cdot (-I(\theta)) + O(|\Delta\theta|^3)$$

$$= \frac{1}{2} I(\theta) (\Delta\theta)^2 + O(|\Delta\theta|^3)$$

**Step 4: Symmetry to second order**

**Important note:** KL divergence is generally **not symmetric**: $D_{KL}(p \| q) \neq D_{KL}(q \| p)$.

However, we can show that to second order, the reversed KL divergence gives the same result.

Consider $D_{KL}(p(\cdot \mid \theta + \Delta\theta) \| p(\cdot \mid \theta))$ where the **expectation is with respect to** $p(x \mid \theta + \Delta\theta)$:

$$D_{KL}(p(\cdot \mid \theta + \Delta\theta) \| p(\cdot \mid \theta)) = \mathbb{E}_{x \sim p(\cdot|\theta + \Delta\theta)}\left[\log \frac{p(x \mid \theta + \Delta\theta)}{p(x \mid \theta)}\right]$$

Applying the same result from Steps 1-3, but now evaluated at $\theta + \Delta\theta$ instead of $\theta$:

$$D_{KL}(p(\cdot \mid \theta + \Delta\theta) \| p(\cdot \mid \theta)) = \frac{1}{2} I(\theta + \Delta\theta) (\Delta\theta)^2 + O(|\Delta\theta|^3)$$

Now, since $I(\theta)$ is a smooth function of $\theta$:

$$I(\theta + \Delta\theta) = I(\theta) + O(|\Delta\theta|)$$

Therefore:

$$I(\theta + \Delta\theta) (\Delta\theta)^2 = I(\theta) (\Delta\theta)^2 + O(|\Delta\theta|^3)$$

This gives:

$$D_{KL}(p(\cdot \mid \theta + \Delta\theta) \| p(\cdot \mid \theta)) = \frac{1}{2} I(\theta) (\Delta\theta)^2 + O(|\Delta\theta|^3)$$

**Result:**
$$D_{KL}(p(\cdot \mid \theta) \| p(\cdot \mid \theta + \Delta\theta)) = D_{KL}(p(\cdot \mid \theta + \Delta\theta) \| p(\cdot \mid \theta)) + O(|\Delta\theta|^3)$$

Both equal $\frac{1}{2} I(\theta) (\Delta\theta)^2$ to second order! This **symmetry to second order** validates that Fisher Information truly defines a metric.
