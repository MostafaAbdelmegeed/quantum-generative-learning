# Equal-size IQP circuits across simulated device layouts

Frozen on 11 November 2026 before running `routing-controls-v1`. This local simulation study implements the routing experiment in the work plan. It does not use a hardware account, device calibration or physical shots.

## Expressibility and matched capacity

Reuse all five six-bit targets, training/test datasets and three seeds from `observable-generalization-v1`. Compare two task-informed sparse circuit families. Both have all six single-X supports and exactly five XX supports, hence 11 angles. Both contain the same star generators required by the known target construction. Their additional generators connect the target blocks into a tree; they are zero in the expressibility witness.

Treat every parity block and every unconstrained bit as a component. Within a non-singleton component, connect its first qubit to every other qubit. For `path_bridges`, connect the last qubit of each component to the first qubit of the next. For `hub_bridges`, connect the first component's first qubit to the first qubit of every other component. Components follow the target block order, followed by free qubits in increasing order. Sort supports lexicographically after construction. Verify that each graph is a connected tree and that both target witnesses reproduce the known probability vector in NumPy and Qiskit.

This is an informed architecture comparison: the construction receives target block labels. It does not establish learning an architecture from data. Full target expressibility is held fixed, but the full distribution families differ. Parameter-slot initial angles are identical between the two families; an edge changing slots can change which angle belongs to a shared support. This is a remaining optimization sensitivity, not a connectivity-only causal comparison.

## Training and deployment

Train ordinary multiscale MMD with the fully enumerated gradient from `observable-diagnostics-v1`, using 1,600 Adam updates, the same initial-parameter seed convention, learning rate 0.03 with cosine decay to 0.003, beta1=0.9, beta2=0.999 and epsilon=1e-8. Each fit uses 6,553,600 phase evaluations and no training shots. Do not select a checkpoint or topology by test performance. Thirty fits cover two families, five conditions and three seeds. Reuse the published classical references on identical data, with provenance hashes.

Compile each fitted circuit separately to three six-node bidirectional coupling graphs: line, ring and star centered at physical qubit zero. Fix initial logical-to-physical mapping to identity, optimization level three and transpiler seed 17. Do not search layouts. Save routed CX counts, depth, final measurement mapping and OpenQASM. Changing device connectivity is separate from the logical circuit topology comparison.

For every compiled circuit, evaluate exact density-matrix probabilities and 2,048 sampled outcomes at CX depolarizing parameters 0, 0.005 and 0.02. Single-qubit SX/X depolarizing parameters are one tenth of the CX parameter; RZ is ideal. Reuse the existing noise definition and logical-output decoding. Simulator seed is `seed + 3109`. There are 90 compiled circuits and 270 noise evaluations. No readout error, relaxation, crosstalk or calibration data enter this model.

At zero noise, every layout must match the uncompiled model in logical bit order. Report ideal target TVD separately from the noise-induced change in target TVD, and also report TVD between noisy and ideal circuit outputs. Noise can accidentally improve target TVD for a poorly fitted circuit; that is not evidence that noise improved its implementation. Include Jensen-Shannon divergence, held-out MMD, true block moments and the correct corrupted-pair target. Keep classical controls beside ideal fits.

## Records and limits

Use one worker and one simulator/BLAS thread after a runtime preflight. Keep source, configuration, protocol, lockfile, environment, all seeds, full optimizer diagnostics, compilation records, exact probabilities, sample counts and failures. Audit receipts, target witnesses, data, circuit sizes, fitted probabilities, metrics and budgets; replay every compiled noise evaluation before archiving a new series.

The comparison tests a narrow resource/fidelity tradeoff under fixed simulation choices. Equal parameter counts do not equalize expressiveness, optimizer difficulty or classical cost. The targets remain classically simple, and no quantum advantage or hardware noise prediction follows from this study.

## Numerical-check amendment before the retry

The first attempt retained 28 completed circuit fits and two execution failures for shuffled-pair seed 137. Both failures came from applying the uncompiled-oracle tolerance of 1e-11 to optimized circuits. A saved-parameter regression reproduces the discrepancy: level-three transpilation changes one probability by about 1.07e-11, while level-zero transpilation agrees within 7e-16. Changing Aer's zero threshold does not change the discrepancy. Output-bit decoding is consistent between the compiled statevector and density-matrix simulation.

The retry fixes `compiled_probability_atol=1e-10` and records the actual maximum probability error at zero noise. Training, targets, selectors, transpilation settings, noise and seeds are unchanged. The first attempt's original source, configuration, protocol and records are retained separately. This amendment concerns numerical verification, not model selection or a revised performance budget.

The first retry exposed the corresponding hub circuit's larger discrepancy, 8.86e-10. Checking all 90 compiled circuits found this to be the largest error. Level-zero compilation of the hub case again agrees within 7e-16. The final retry fixes the compiled-probability tolerance at 1e-8, retaining the measured discrepancies rather than rounding them away. Both earlier attempts remain in the archive. The full compiler-check receipt is saved with the report. No fitted parameters, optimizer schedules, noise levels or compilation settings were changed in response to these checks.
