# ILP Decoder (Gurobi)

Integer linear programming decoder for quantum error correction using Gurobi.

## Notes
- Gurobi requires a valid license for practical workloads (e.g., surface code).
- CPLEX is not supported yet (future work).
- The `BeliefMatching/` directory is for development support only and is excluded from packaging.

## Quick Start (Steane code)

1) Create the environment.

```bash
conda env create -f environments.yml
conda activate ilp-decoder
```

2) Run a minimal decode example.

```bash
PYTHONPATH=./src python - <<'PY'
import numpy as np
from scipy.sparse import csc_matrix
import gurobipy as gp

from ilp_decoder import DecoderConfig, ILPDecoder
from ilp_decoder.core import DecoderDependencies

H = csc_matrix(
	[
		[1, 0, 0, 1, 0, 1, 1],
		[0, 1, 0, 1, 1, 0, 1],
		[0, 0, 1, 0, 1, 1, 1],
	]
)
Lx = csc_matrix([[1, 1, 1, 1, 1, 1, 1]])

prior = np.full(7, 0.1, dtype=float)
true_error = np.array([1, 0, 0, 0, 0, 0, 0], dtype=int)
syndrome = (H @ true_error) % 2

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

decoder = ILPDecoder(
	parity_check_matrix=H,
	observables=Lx,
	prior=prior,
	config=DecoderConfig(log_to_console=False),
	deps=DecoderDependencies(env=env),
)

result = decoder.decode_result(syndrome)
print("error_vector:", result.error_vector)
print("predicted_observables:", result.predicted_observables)
print("objective_value:", result.objective_value)

env.dispose()
PY
```


## Future work
- optimize performance for cluster machine