# Follow-up: give observed correlations more training weight

Frozen on 23 October 2026 before running `scaling-pairs-v1`.

The first 32-qubit experiment reached 98.5% agreement per adjacent pair, but only 81.0% of complete samples satisfied every pair. Its multiscale MMD was close to the informed classical model's score despite the difference in invalid samples. This motivates an exploratory change to the training kernel.

From the 512 training examples, calculate the absolute centered spin correlation `abs(E[Zi*Zj] - E[Zi]*E[Zj])` for every pair of columns. Select the `n/2` largest values, breaking ties lexicographically. The selector receives no target distribution, test examples, or known pair labels. The chosen count remains a deliberate hyperparameter motivated by this paired-bit task.

Use half the mask budget for ordinary multiscale Fourier masks and half for uniformly sampled masks from those selected pairs. For a fixed training set, this is the valid kernel

`K_mix(x,y) = 0.5*K_multiscale(x,y) + 0.5*mean_selected_pairs[chi_pair(x)*chi_pair(y)]`.

Each character product is a positive semidefinite kernel. The mixture keeps the original kernel's full-support component. The independent paired estimator therefore remains unbiased for MMD against the fixed empirical target under this new kernel. The kernel is data-dependent; this is not an unbiased estimate of a fixed population-kernel test statistic.

Keep all other settings from `scaling-v1`: datasets, initial parameters, 1,500 Adam updates, 128 total masks per step, 256 uniform strings per half, learning-rate schedule, final-checkpoint rule and evaluation seeds. Thus both methods perform 98,304,000 classical phase evaluations per fit. Record the small pair-selection cost separately. Do not run a hyperparameter search.

Use the original multiscale sample MMD for evaluation in both experiments, alongside complete-sample validity, pair agreement and marginal bias. Compare paired seeds at all three widths and retain every run. A better validity result would support this feature-weighting change on this toy task. It would not establish performance on other data or superiority to classical generators.

This is a kernel-design experiment within the published IQP training framework. Multiscale and adaptive kernels already have prior art: [Rudolph et al.](https://www.nature.com/articles/s41534-024-00902-0) and [adaptive-kernel quantum generative work](https://arxiv.org/abs/2510.08476). No claim of algorithmic novelty is made.
