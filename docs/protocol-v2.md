# Follow-up protocol

Frozen on 23 October 2026, before running either follow-up. The v1 records stay available.

All 18 exact QCBM fits reached the 250-iteration limit in v1. `reproduction-v2` reruns those same initializations with a 2,000-iteration cap. Data, circuits, objectives and stopping tolerances stay fixed. Report both budgets and convergence status. This follow-up addresses incomplete optimization; it is exploratory because the first results motivated it.

`iqp-mc-v1` tests classical Monte Carlo training of the same six-qubit IQP circuits used in `connectivity-v1`. It uses the correlator identity in [IQPopt, equation (8)](https://arxiv.org/html/2501.04776v1) and the Fourier form of Hamming-kernel MMD. Each step draws 64 masks and two independent sets of 256 uniform classical bitstrings. Products of the independent moment estimates give an unbiased loss and gradient against the fixed empirical training distribution. The data-only term uses the empirical squared moment, not a population U statistic. Individual loss estimates may be negative.

Run 1,500 Adam updates with an initial learning rate of 0.03, cosine decay to 0.003, beta values 0.9 and 0.999, and epsilon 1e-8. Draw fresh masks and bitstrings at every step. Use the final parameters, with no checkpoint selection. Seeds, data and initial parameters match the exact IQP study. The training function accepts binary sample matrices and allocates no statevector, probability table or full kernel matrix.

After training, compute exact six-qubit metrics and check the exported Qiskit circuit. These diagnostics never enter optimization. Compare all three datasets, all three topologies and all three seeds with the exact IQP reference. Optimizers and compute budgets differ, so this comparison tests whether the implementation learns useful distributions; it does not isolate estimator noise or establish a speed advantage.

Record classical masks, uniform bitstrings, phase evaluations, elapsed time, and zero quantum training shots. Training with classical estimates follows published work. The project contribution is the verified Qiskit implementation and the reproducible connectivity study, not a new training theorem.
