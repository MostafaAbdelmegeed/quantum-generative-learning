# Experiment protocol v1

Frozen before model runs on 23 October 2026. These are exploratory small-system experiments, not an exact numerical reproduction of every setting in the source papers.

## Data and access

Use six binary variables. Generate 2-by-3 Bars-and-Stripes (ten equally likely valid patterns), a discretized two-component Gaussian mixture on integers 0 through 63, and an even-parity distribution as a higher-order diagnostic. Each dataset has 512 training samples and 1,024 independent held-out samples. Seeds 42, 137 and 2026 define paired datasets and initializations. All models in a comparison receive the same training samples.

Bit position zero is qubit zero and the least significant bit of an integer. In a pattern image, bit zero is the first row-major pixel. Qiskit's printed strings list the highest bit first. Tests must check this distinction.

Only empirical training probabilities enter the fitting objective. The true distribution is available solely for evaluation. Three seeds describe optimization and finite-data sensitivity; they do not justify broad statistical claims. Synthetic targets and all results can be published.

## Exact training

The QCBM uses three layers of RY/RZ rotations and line CNOTs, followed by a final RY layer (42 parameters). An exact NumPy state simulator and adjoint gradient make repeated small fits inexpensive. Qiskit statevectors independently validate circuit outputs before experiments. Use L-BFGS-B with a cap of 250 iterations. Report actual iterations and objective/gradient calls; convergence is not guaranteed by the cap.

The loss is squared maximum mean discrepancy (MMD) on bitstrings with kernel `exp(-Hamming(x,y)/(2*b))`. Compare `b=1` with the mean over `b=n*[0.01,0.1,0.25,0.5,1,10]`. These are existing loss choices, not a new algorithm. The exact training objective compares the model with the empirical training distribution. [Liu and Wang](https://arxiv.org/abs/1804.04168) [Rudolph et al.](https://www.nature.com/articles/s41534-024-00902-0)

Classical references are an empirical multinomial, independent Bernoulli variables, an exact pairwise maximum-entropy model, and an autoregressive network with small conditional MLPs. Report parameter counts and compute costs; these are useful classical comparators, not parameter-matched controls.

## Training with finite measurements

Use Qiskit's StatevectorSampler, 256 or 1,024 measurements per objective call, and 100 SPSA updates with two objective calls each. The full quantum training budget is therefore 51,200 or 204,800 shots per fit. Both losses use the same schedule, initial parameters and paired data. Each call uses fresh samples. Use an unbiased model-model U statistic and the training-sample U statistic, so an estimate may be negative. Do not clamp the training objective.

The frozen SPSA schedule is `a_k=0.15/(k+1)^0.602`, `c_k=0.15/(k+1)^0.101`. This first comparison measures behavior under that schedule, not the best performance obtainable from each loss. Exact final metrics are simulator diagnostics outside the shot training loop. They must not choose parameters or stop training.

## IQP connectivity experiment

Use `U(theta)=product exp(i*theta_j*X_Sj)` on the zero state, matching the positive-sign generator convention documented by IQPopt. Include singleton generators and pair generators from a complete graph, a line, or a random labeled tree drawn from a seeded PrÃ¼fer sequence. The line and tree have equal parameter counts; the complete graph has more parameters. Record the actual edge sets. This first implementation uses exact small-system training, not IQPopt's scalable Monte Carlo objective estimator. [IQPopt](https://github.com/XanaduAI/iqpopt)

Freeze the multiscale loss and optimizer across topologies. Transpile the trained circuits onto a bidirectional line with the native basis RZ, SX, X, CX, optimization level 3 and seed 17. Allow the transpiler to select a layout, preserve logical measurement order, and verify noiseless compiled output against the uncompiled distribution.

Evaluate CX depolarizing parameters 0, 0.001, 0.005, 0.01 and 0.02; SX/X errors use one tenth of that parameter. RZ is ideal. The noise model excludes readout, relaxation and crosstalk. Exact density-matrix evolution gives a noise-only distribution, and a separate 4,096-shot sample illustrates finite readout statistics. This is simulated noise, not a calibrated device result.

Compare ideal fit error with noisy fit error, two-qubit gate count and depth. A lower noisy error can reflect model bias as well as routing savings. Equal sparse parameter counts and the ideal reference help separate those effects, but do not establish causality across all architectures.

## Evaluation and records

Report total variation distance (TVD), Jensen-Shannon divergence in nats, held-out MMD and parity imbalance. For Bars-and-Stripes, report invalid-pattern probability and expected coverage of valid patterns in 1,024 draws. Values computed from full model probabilities are explicitly exact simulator diagnostics. Saved parameters, probabilities, empirical sample counts, circuit metadata, configurations, dependency versions and source hashes must support independent metric recomputation.

No parameter or hyperparameter is selected using the known target distribution. If results suggest a different setup, keep the original artifacts and use a new configuration version with an explanation. Checkpoints selected by the training objective are allowed; final test metrics are computed only afterward. The first report will show all completed runs and disclose failures.
