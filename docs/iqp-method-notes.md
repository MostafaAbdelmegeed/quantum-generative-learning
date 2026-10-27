# Classical IQP training: implementation notes

Checked on 23 October 2026. These notes specify a small reproduction and its numerical checks. They contain no experiment results.

## Sources and scope

The starting method is the classical correlator and unbiased MMD estimator in [Recio-Armengol, Ahmed and Bowles, arXiv:2503.02934v2](https://arxiv.org/html/2503.02934v2), sections 4.1-4.4 and appendix A. The related [IQPopt paper, arXiv:2501.04776v2](https://arxiv.org/html/2501.04776v2), describes the implementation and the distinction between estimating correlations and sampling a circuit.

The code references below pin IQPopt to commit `18ae04255ed80a274d8c49ed91cd4e6214757ccd`, dated 22 October 2026. The paper's experiment repository is pinned to [scaling-gqml commit `8e5620cfa71f9a4e4254597b2dfe1692231d3723`](https://github.com/XanaduAI/scaling-gqml/tree/8e5620cfa71f9a4e4254597b2dfe1692231d3723), dated 27 November 2025. Its [training entry point](https://github.com/XanaduAI/scaling-gqml/blob/8e5620cfa71f9a4e4254597b2dfe1692231d3723/paper/training/train_iqp.py) is a useful reproduction reference. IQPopt's current [README](https://github.com/XanaduAI/iqpopt/blob/18ae04255ed80a274d8c49ed91cd4e6214757ccd/README.md) says ongoing development has moved to PennyLane Labs; pinning the older implementation avoids silently changing the target method.

## Circuit and angle conventions

Use an all-zero input, computational-basis measurement, and commuting gates

\[
U(\theta)=\prod_{j=1}^{m}\exp(i\theta_j X_{S_j}).
\]

Represent each support by a binary row \(g_j\), with \(g_{j,r}=1\) exactly when qubit \(r\) belongs to \(S_j\). All dot products inside parity expressions are modulo two. A mask \(a\) specifies the observable \(Z_a\). Use the same qubit order in support arrays, data columns, probability-vector decoding, and Qiskit measurement strings.

For Qiskit, this positive-exponent convention requires `rx(-2*theta)` and `rxx(-2*theta)`. [IBM's RXX definition](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RXXGate) is \(R_{XX}(\alpha)=\exp(-i\alpha X\otimes X/2)\).

The pinned IQPopt [circuit implementation](https://github.com/XanaduAI/iqpopt/blob/18ae04255ed80a274d8c49ed91cd4e6214757ccd/src/iqpopt/iqp_optimizer.py) uses Hadamards around `MultiRZ(2*parameter)`, which implements the opposite global exponent sign. This yields the same output probabilities for this real commuting family: changing every angle's sign complex-conjugates the output amplitudes. Probability comparisons therefore cannot detect a consistent global sign reversal. Keep the declared convention explicit, and use a small direct statevector comparison when verifying amplitudes.

## Deriving the correlator and its gradient

Write

\[
c_j(z)=(-1)^{g_j\cdot z},\qquad
t_j(a)=g_j\cdot a\pmod 2,\qquad
\phi_a(z)=2\sum_j\theta_j t_j(a)c_j(z).
\]

Conjugating by Hadamards makes the circuit diagonal. Its basis-state phase is \(\lambda_z=\exp(i\sum_j\theta_j c_j(z))\). The expectation of \(Z_a\) is the uniform average of \(\lambda_{z\oplus a}^{*}\lambda_z\); taking its real part gives

\[
m_a(\theta)=\langle Z_a\rangle
=\mathbb E_{z\sim\mathrm{Uniform}(\{0,1\}^n)}[\cos\phi_a(z)].
\]

This is equation 13 of the generative-model paper in the support notation above. Its direct derivative is

\[
\partial_{\theta_j}m_a
=-2t_j(a)\mathbb E_z[c_j(z)\sin\phi_a(z)].
\]

Only generators that anticommute with \(Z_a\) contribute. In particular, the empty mask has moment one and zero gradient. Inactive generators have exactly zero derivative for that mask.

Parameter sharing is valid. If \(\theta_j=\sum_r B_{jr}\eta_r+\beta_j\), accumulate the individual derivatives through \(B^T\). For one parameter tied across several supports, sum those supports' contributions before applying the sine. IQPopt implements this through its `trans_par` parameter transformation in the [simulator source](https://github.com/XanaduAI/iqpopt/blob/18ae04255ed80a274d8c49ed91cd4e6214757ccd/src/iqpopt/iqp_optimizer.py).

## Fourier masks for the Hamming kernel

Define the kernel using a variance-like parameter \(b>0\):

\[
k_b(x,y)=\exp[-H(x,y)/(2b)].
\]

Let \(r=\exp[-1/(2b)]\) and \(\rho=(1-r)/2\). For each bit, independently sample \(a_i\sim\mathrm{Bernoulli}(\rho)\). The one-bit factor obeys

\[
\mathbb E_{a_i}[(-1)^{a_i(x_i\oplus y_i)}]
=\begin{cases}1,&x_i=y_i,\\1-2\rho=r,&x_i\ne y_i.\end{cases}
\]

Multiplying the factors proves

\[
k_b(x,y)=\mathbb E_a[(-1)^{a\cdot x}(-1)^{a\cdot y}],\qquad
L(\theta)=\mathrm{MMD}^2(p,q_\theta)=\mathbb E_a[(c_a-m_a)^2],
\]

where \(c_a=\mathbb E_{x\sim p}[(-1)^{a\cdot x}]\). IQPopt uses `sigma` for standard deviation, so **\(b=\sigma^2\)**. Its [mask sampler](https://github.com/XanaduAI/iqpopt/blob/18ae04255ed80a274d8c49ed91cd4e6214757ccd/src/iqpopt/gen_qml/iqp_methods.py#L93-L107) uses `(1-exp(-1/(2*sigma**2)))/2`.

Do not discard the empty mask and then average the remaining masks without a correction: conditioning on a nonempty mask changes the kernel normalization. A mixture of bandwidths is a mixture of the corresponding kernels, so sample a bandwidth and then its mask, or average their losses explicitly.

## Two meanings of an unbiased loss

For a fixed training set \(D\) of size \(N\), the exact empirical moment is

\[
c_a^D=N^{-1}\sum_{x\in D}(-1)^{a\cdot x}.
\]

There are two legitimate targets: MMD against this empirical distribution, and an estimator of population MMD that is unbiased over newly drawn training datasets. State which target the reported loss uses.

For the empirical target, independently draw uniform-bit batches A and B and compute their means \(\widehat m_a^A,\widehat m_a^B\). Then

\[
\widehat L_a=(\widehat m_a^A-c_a^D)(\widehat m_a^B-c_a^D)
\]

is unbiased for \((m_a-c_a^D)^2\). Differentiate both factors:

\[
\widehat{\nabla L_a}
=(\widehat m_a^B-c_a^D)\widehat{\nabla m_a^A}
+(\widehat m_a^A-c_a^D)\widehat{\nabla m_a^B}.
\]

This gradient is unbiased because each mean/gradient factor is independent of the factor from the other batch. Reusing a batch in both factors instead adds \(\operatorname{Var}(\widehat m_a)\) to the expected loss. That variance depends on the parameters, so the naive squared estimator also biases training gradients.

An alternative uses one batch of \(S>1\) independent values \(f_s=\cos\phi_a(z_s)\):

\[
\widehat{m_a^2}_{U}
=\frac{(\sum_s f_s)^2-\sum_s f_s^2}{S(S-1)}.
\]

Use \(\widehat{m_a^2}_U-2c_a^D\overline f+(c_a^D)^2\) for the empirical target. Its derivative can be computed without storing every per-parameter derivative:

\[
\nabla\widehat{m_a^2}_{U}
=\frac{2[(\sum_s f_s)(\sum_s\nabla f_s)-\sum_s f_s\nabla f_s]}{S(S-1)}.
\]

For a population-unbiased estimator, replace the data-only square by

\[
\widehat{c_a^2}_{U}=\frac{N(c_a^D)^2-1}{N-1},\qquad N>1.
\]

This is the estimator used by IQPopt's [loss implementation](https://github.com/XanaduAI/iqpopt/blob/18ae04255ed80a274d8c49ed91cd4e6214757ccd/src/iqpopt/gen_qml/iqp_methods.py#L35-L50). For a fixed dataset, the two target definitions differ by a parameter-independent term; their gradients agree. They can have different reported loss values, which matters when comparing a stochastic result against an exact empirical oracle.

Unbiased finite-sample estimates may be negative. Clipping, taking an absolute value, or taking a square root changes the estimator. Keep signed MMD-squared estimates in the experiment artifacts. If training uses minibatches of data, either account for the data-side square or label the objective accordingly.

## Cost and statistical precision

The following bounds follow directly from the formulas and assume independent uniform strings within each batch.

- Since \(|f_s|\leq1\), one moment has variance at most \(1/S\). A worst-case standard error of 0.01 therefore needs 10,000 uniform strings, regardless of qubit count. This bound concerns additive precision; it does not promise useful relative precision for tiny moments.
- An individual untied parameter derivative has magnitude at most two and sample-mean variance at most \(4/S\). Tying one parameter across \(r\) generators changes the simple bound to \(4r^2/S\).
- With \(K\) masks, \(m\) generators, \(S\) strings, and \(n\) qubits, dense parity construction costs \(O(nm(K+S))\). Phase evaluation costs \(O(KmS)\). Reverse differentiation of these matrix products has the same order of arithmetic cost. Computing empirical data moments directly adds \(O(NKn)\).
- The implementation needs arrays for generators, parity tables, and \(K\times S\) phases; none requires a \(2^n\) statevector. Chunking masks and strings can cap memory. “Polynomial” still allows a costly dense generator set or a large precision budget.

Sharing a uniform-string batch across masks preserves each estimator's mean but correlates their errors. A standard error obtained by dividing the standard deviation across masks by \(\sqrt K\) does not generally measure the full Monte Carlo uncertainty in that case. Use independent complete evaluations for uncertainty checks. Fresh randomness per training step prevents the optimizer from fitting one fixed random approximation; keep independent evaluation seeds.

## Small reproduction and verification plan

Start with four to six qubits and a fixed empirical target distribution. For each verification, save the seed, supports, angles, target weights, and bandwidth convention.

1. Enumerate all \(2^n\) uniform strings and all masks. Compare the resulting moments with Qiskit statevector probabilities. Include an asymmetric support set and mixed angle signs to catch qubit-order mistakes.
2. Compare the analytical derivative with central differences of the exact loss. Include a tied parameter, an inactive support, the empty mask, zero angles, and a single-qubit circuit with known moment \(\cos(2\theta)\).
3. Verify that the fully enumerated Fourier loss equals the direct probability-kernel expression \((p-q)^T K(p-q)\). This catches a bandwidth-versus-bandwidth-squared error.
4. At fixed nonzero parameters, repeat independent stochastic estimates. Check loss and gradient means against the exact oracle using Monte Carlo standard errors. Compare the corrected estimator with the naive squared-moment estimator so the bias is visible. Use the same target definition in both comparisons.
5. Train exact-oracle and stochastic versions from the same initial parameters, with the same architecture and data. Give both the same validation protocol and optimizer search. Report gradient steps, wall time, and random-sample budgets separately. Score their final small models using exact MMD, total variation, and generated samples.
6. After those checks, run a 32-qubit training demonstration with held-out moment evaluation. Record memory and runtime versus mask/string budgets. Label this as a classical estimator scalability experiment.

These steps reproduce a method at a manageable scale. The proposed extension is a controlled comparison of dense, device-compatible, and equally sparse random support sets under a fixed noise/compilation protocol. A literature search focused on that extension is still needed before calling it novel.

## What a larger run establishes

The generative-model paper reports classically trained models up to 1,000 qubits, but its large-model evaluation does not include generated quantum samples. Its classical comparisons are limited, and it does not claim state-of-the-art superiority. It also shows that this IQP family is not a universal distribution model without extra qubits. [Paper, table 3 and sections 5.3, 7 and 9](https://arxiv.org/html/2503.02934v2)

For this project, a successful 32-qubit run would demonstrate an implemented training estimator with polynomial-size working arrays. It would not establish accurate hardware generation, noise tolerance, sampling hardness of the learned instances, or superiority to classical generators. In particular, a low estimated MMD can coexist with poor properties that the chosen bandwidth rarely probes. Keep small-model sample validation and larger-model correlator validation as separate reported results.

The noise study must also remain separate from the ideal correlator derivation. Inserting arbitrary noise channels into the deployed circuit does not make the ideal estimator a noise-aware training method. Any claim of training against noise needs a justified modified estimator or explicit noisy simulation at the small scale.
