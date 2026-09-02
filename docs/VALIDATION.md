# Instance Validation and Dataset Quality

Every generated instance is validated before it is assigned an instance ID or written to
disk. Validation uses no solver results. Its purpose is to separate generator or encoding
errors from solver performance and to expose structural properties that could make a
benchmark collection trivial or unbalanced.

Validation cannot prove that an instance is universally hard. Difficulty depends on the
solver and representation. It can prove structural consistency, test the QUBO encoding,
flag common sources of accidental triviality, and provide measurements for constructing a
balanced dataset.

## What happens during generation

```text
Generate domain instance and QUBO
              |
              v
Check QUBO structure and coefficients
              |
              v
Run problem-specific structural checks
              |
              v
Compare QUBO energy with an independent objective calculation
              |
              v
Measure instance characteristics and create warnings
              |
              v
Apply the user's predeclared warning-rejection rules
              |
              v
Save JSON and update manifest.csv
```

An instance with a validation error is never saved. Warnings do not prevent saving unless
their codes were explicitly placed in `reject_warnings` before generation.

## Validation metadata

Every problem JSON contains a `validation` object:

```json
{
  "validation": {
    "status": "passed",
    "has_warnings": false,
    "validator_version": 1,
    "errors": [],
    "warnings": [],
    "characteristics": {
      "qubo_variables": 20,
      "qubo_quadratic_density": 0.25,
      "encoding_check": {
        "method": "deterministic_random",
        "samples_checked": 131,
        "maximum_absolute_error": 0.0
      }
    }
  }
}
```

Errors and warnings contain stable `code` values suitable for configurations and a
human-readable `message`.

## QUBO and encoding checks

The generic validator checks:

- Positive variable count and one unique name per variable.
- Valid, unique linear and upper-triangular quadratic terms.
- In-range variable indices.
- Finite coefficients and offset.
- Realized QUBO term counts, density, coefficient statistics, and inactive variables.

Every domain problem also calculates energy independently from its original problem data.
For QUBOs with at most 12 variables, every binary assignment is checked. Larger QUBOs use
the all-zero, all-one, alternating, and 128 deterministically generated samples. Random
QUBO is a direct coefficient definition, so only structural validation applies.

The independent checks include cut weight, conflicts and penalties, coloring violations,
partition imbalance, knapsack capacity residual, set overlaps, unsatisfied clauses and
3-SAT ancilla penalties, and native Ising energy.

## Problem-specific measurements

| Family | Recorded characteristics and checks |
|---|---|
| Graph problems | Realized edges and density, components, isolated vertices, degree statistics, and requested topology invariants |
| Independent Set | Conflict count, vertex-value statistics, penalty safe threshold and margin |
| Maximum Clique | Non-edge count, constraint density, vertex values, penalty safe threshold and margin |
| Graph Coloring | QUBO size, color count, penalty ratio, and exact colorability when the graph has at most 16 vertices |
| Number Partitioning | Sum, numeric bit length, GCD, duplicate fraction, dominant-number fraction, and small-instance perfect-partition check |
| Quadratic Knapsack | Weight/value statistics, realized bonus density, capacity pressure, individually fitting items, slack variables, and penalty margin |
| Set Packing | Set sizes, duplicates, overlap density, unused universe elements, values, and penalty margin |
| SAT / Max-SAT | Clause ratio, duplicates, unused variables, literal balance, occurrences, weights, planted-solution check, ancillas, and small-instance satisfiability |
| Spin Glass | Coupling/field statistics, realized field fraction, sign balance, connectivity, and cycle-basis frustration indicator |
| Random QUBO | Realized density, sign balance, coefficient statistics, and inactive variables |

## Warning codes

Warnings describe valid but potentially undesirable instances. They are deliberately not
rejected automatically.

| Code | Meaning |
|---|---|
| `empty_graph` | Generated graph has no edges. |
| `disconnected_graph` | Generated graph has multiple connected components. |
| `isolated_vertices` | One or more graph vertices have degree zero. |
| `complete_conflict_graph` | Independent Set permits at most one selected vertex. |
| `edgeless_clique_graph` | Maximum Clique permits at most one selected vertex. |
| `complete_clique_graph` | Every vertex forms one clique, making the optimum immediate. |
| `excess_colors` | Graph Coloring has at least one distinct color per vertex. |
| `not_k_colorable` | Exact validation found no valid coloring with the requested colors. |
| `zero_one_hot_penalty` | Coloring does not enforce exactly one color per vertex. |
| `zero_edge_penalty` | Coloring does not enforce different colors on adjacent vertices. |
| `identical_partition_numbers` | Every partition number is identical. |
| `dominant_partition_number` | One partition number exceeds half the total sum. |
| `no_item_fits` | No knapsack item fits individually. |
| `all_items_fit` | Knapsack capacity is nonbinding. |
| `duplicate_candidate_sets` | Set Packing contains duplicate candidate sets. |
| `no_set_overlaps` | Every candidate set can be selected. |
| `all_sets_overlap` | Every candidate-set pair conflicts. |
| `duplicate_clauses` | SAT formula repeats clauses. |
| `unused_boolean_variables` | SAT formula contains variables absent from every clause. |
| `literal_sign_imbalance` | SAT literal signs are strongly imbalanced. |
| `weak_ancilla_penalty` | A 3-SAT ancilla penalty may not enforce its product relation. |
| `interaction_free_spin_glass` | Spin Glass has no pair interactions. |
| `no_quadratic_interactions` | Random QUBO contains only linear terms. |
| `inactive_random_variables` | Random QUBO contains variables with no nonzero terms. |
| `one_sided_coefficients` | All stored Random QUBO coefficients have the same sign. |
| `weak_penalty` | A problem penalty is below the validator's conservative safe threshold. |

## Declaring rejection rules

Rules should be selected before benchmarking any solver. For example, this configuration
requires connected Max-Cut graphs without isolated vertices and allows up to 20 attempts:

```yaml
output_folder: generated_problems
base_seed: 42

validation:
  max_attempts: 20
  reject_warnings:
    - empty_graph
    - disconnected_graph
    - isolated_vertices

problems:
  - type: max_cut
    count: 10
    parameters:
      num_vertices: 30
      edge_probability: 0.15
```

Rules can be overridden on an individual problem entry. A retry uses the original seed
plus a deterministic one-billion-step offset, so repeated runs make the same decisions.
The accepted instance stores the seed that actually generated it.

Direct command-line generation supports the same behavior:

```bash
python generate.py --problem max_cut --count 10 --seed 42 \
  --reject-warning disconnected_graph \
  --reject-warning isolated_vertices \
  --max-attempts 20
```

Do not reject a warning merely because one solver performs poorly on instances carrying
it. That would make the benchmark solver-dependent. Declare structural acceptance rules
first and retain them with the dataset configuration.

## Dataset manifest

Every call to `save_problem` updates `manifest.csv` in the output folder. Saving the same
instance again replaces its row instead of duplicating it. The manifest contains:

- Instance ID, problem type, seed, and JSON filename.
- QUBO variable, linear-term, and quadratic-term counts.
- Validation status and warning/error codes.
- Flattened `parameter.*` columns.
- Flattened realized `stat.*` columns.

This makes it possible to group later results by actual density, connectivity, coefficient
scale, clause ratio, capacity pressure, or other structural properties rather than relying
only on requested parameters.

## Recommended benchmark practice

1. Define benchmark cells such as family × size × density × weight regime.
2. Declare warning-rejection rules before generating or solving anything.
3. Generate at least 5–10 seeds per cell; use more when runtime permits.
4. Keep easy, medium, and hard structural regimes rather than retaining only hard cases.
5. Save and reuse exactly the same JSON instances for every solver.
6. Keep planted and unrestricted SAT collections separate.
7. Report results by benchmark cell and realized characteristics, not only as one average.

