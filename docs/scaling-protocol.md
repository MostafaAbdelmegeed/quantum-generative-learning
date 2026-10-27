# Larger input demonstration

Frozen on 23 October 2026 before `scaling-v1`. This experiment checks the complete classical-training and Qiskit-sampling workflow at 8, 16 and 32 qubits. It uses a deliberately easy classical distribution so the training and sampling code can be checked at larger widths.

Each example contains independent fair bits repeated in adjacent pairs: `00` or `11`. A 32-bit example has 16 independent choices. Use 512 training examples, 1,024 independent test examples, and seeds 42, 137 and 2026. All circuits contain single-qubit X generators and all adjacent XX generators, including the edges between pairs. Their parameter counts are 15, 31 and 63. The correct pair structure is a subset of each line circuit's supports.

Train with the same empirical-target Monte Carlo estimator and Adam schedule as `iqp-mc-v1`, increasing only the masks per step to 128. Use 1,500 updates and two independent sets of 256 uniform strings per update. Start from the existing seeded uniform angles. No validation or test result selects a checkpoint.

Export the final parameters to Qiskit and draw 2,048 samples with Aer's noiseless matrix product state simulator. Set the truncation threshold to zero and use direct measurement sampling. Draw the same number from the initial circuit. This sparse circuit is classically tractable; the larger qubit count makes no claim about hard sampling or quantum advantage. Record simulator time, circuit depth, gates and QASM.

Compare with independently fitted Bernoulli bits, resampling from the training set, and a classical model that knows the adjacent-pair structure and fits one Bernoulli probability per pair. The last model is an explicitly informed reference. Give every model 2,048 samples. Score the fraction of entirely valid examples, average pair agreement, marginal bias, and an unbiased sample MMD against the independent test set. Negative estimated MMD values are valid. Do not calculate a full probability distribution or claim exact TVD at these widths.

Save raw train, test and output bitstrings in Qiskit's printed order, highest bit first. Save the fitted angles, initial angles and source snapshots. The audit regenerates data, recomputes sample metrics and replays Qiskit sampling with the recorded seed. Three seeds show sensitivity, not a confidence interval. Widths change the target problem and kernel mixture, so MMD numbers across widths are not directly comparable.
