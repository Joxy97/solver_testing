# QUBO Problem Catalog

This document explains every problem supported by the generator, the parameters accepted
by its Python script, and useful parameter sweeps for building benchmark collections.
For structural checks, warning-based acceptance rules, and dataset manifests, see
[`VALIDATION.md`](VALIDATION.md).

All QUBOs use the convention:

```text
E(x) = offset
     + sum_i linear[i] * x_i
     + sum_{i<j} quadratic[i,j] * x_i * x_j
```

The generator minimizes this energy. Generated JSON files also contain the original
problem data, parameter values, random seed, variable names, and a brief explanation of
the encoding.

## How to use the sweep suggestions

The values below are recommendations, not mandatory limits. The current generator does
not interpret YAML lists as sweeps. To generate a sweep, add one problem entry for each
parameter combination and use `count` to generate several random instances of it:

```yaml
problems:
  - type: max_cut
    count: 10
    parameters:
      num_vertices: 20
      edge_probability: 0.1

  - type: max_cut
    count: 10
    parameters:
      num_vertices: 20
      edge_probability: 0.5
```

For meaningful comparisons, generate at least 5–10 seeds for every parameter combination.
Begin with the smaller suggested sizes, then increase them when the intended solver can
comfortably handle the resulting number of QUBO variables and interactions.

## 1. Maximum Cut

**Problem type:** `max_cut`

Maximum Cut divides the vertices into two groups and maximizes the total weight of edges
whose endpoints lie in different groups. There is one binary variable per vertex; its
value selects the side of the cut. QUBO energy is the negative cut weight, so a lower
energy means a larger cut.

| Parameter | Type | Default | Meaning for Maximum Cut |
|---|---:|---:|---|
| `num_vertices` | integer ≥ 2 | `20` | Number of vertices to divide between the two sides of the cut; also the number of QUBO variables. |
| `graph_type` | choice | `erdos_renyi` | Shape of the cut instance: `erdos_renyi`, `random_regular`, `cycle`, or `complete`. |
| `edge_probability` | number in [0, 1] | `0.25` | Probability of each possible cut edge when `graph_type` is `erdos_renyi`; ignored by other graph types. |
| `degree` | integer ≥ 1 | `3` | Number of edges incident to each vertex when `graph_type` is `random_regular`; ignored by other graph types. It must be smaller than `num_vertices`, and their product must be even. |
| `weighted` | boolean | `false` | If true, crossing edges contribute their generated weights to the cut value; otherwise every crossing edge contributes 1. |
| `min_edge_weight` | integer ≥ 1 | `1` | Smallest possible contribution of a crossing edge when `weighted` is true. |
| `max_edge_weight` | integer ≥ 1 | `10` | Largest possible contribution of a crossing edge when `weighted` is true; it must not be smaller than `min_edge_weight`. |

**Suggested sweeps:**

- Size: `num_vertices = [20, 50, 100, 200]`.
- Erdős–Rényi density: `edge_probability = [0.05, 0.1, 0.25, 0.5, 0.75]`.
- Topology: compare `cycle`, `random_regular`, and `erdos_renyi`.
- Regular degree: `degree = [3, 5, 10]`, subject to the regular-graph validity rules.
- Weight structure: compare `weighted = [false, true]`; for weighted graphs try
  `max_edge_weight = [5, 10, 100]`.

Avoid using only complete graphs: they are useful dense controls, but their symmetry makes
them structurally different from typical random instances.

## 2. Maximum Independent Set

**Problem type:** `independent_set`

The goal is to select as much vertex value as possible without selecting both endpoints
of any edge. A binary value of 1 selects a vertex. The QUBO minimizes negative selected
value plus a penalty for every selected edge.

| Parameter | Type | Default | Meaning for Maximum Independent Set |
|---|---:|---:|---|
| `num_vertices` | integer ≥ 2 | `20` | Number of vertices that may be selected; also the number of QUBO variables. |
| `graph_type` | choice | `erdos_renyi` | Structure of the vertex-conflict graph: `erdos_renyi`, `random_regular`, `cycle`, or `complete`. |
| `edge_probability` | number in [0, 1] | `0.25` | Probability that two vertices conflict when `graph_type` is `erdos_renyi`; ignored by other graph types. |
| `degree` | integer ≥ 1 | `3` | Number of conflicting neighbors per vertex for `random_regular`; ignored by other graph types. It must be smaller than `num_vertices`, and their product must be even. |
| `weighted` | boolean | `false` | Controls weights stored on conflict-graph edges. Edge weights do not affect the current Independent Set QUBO. |
| `min_edge_weight` | integer ≥ 1 | `1` | Minimum stored conflict-edge weight when `weighted` is true; it does not affect the QUBO objective. |
| `max_edge_weight` | integer ≥ 1 | `10` | Maximum stored conflict-edge weight when `weighted` is true; it must not be smaller than `min_edge_weight` and does not affect the QUBO objective. |
| `weighted_vertices` | boolean | `false` | If true, maximize total generated vertex value; otherwise maximize the number of selected vertices. |
| `min_vertex_value` | integer ≥ 1 | `1` | Smallest selectable-vertex value when `weighted_vertices` is true. |
| `max_vertex_value` | integer ≥ 1 | `10` | Largest selectable-vertex value when `weighted_vertices` is true; it must not be smaller than `min_vertex_value`. |
| `penalty` | number ≥ 0 or automatic | automatic | Energy added for each edge whose two endpoints are selected. Automatic uses the largest generated vertex value plus 1. |

**Suggested sweeps:**

- Size: `num_vertices = [20, 50, 100, 200]`.
- Graph density: `edge_probability = [0.05, 0.1, 0.2, 0.4, 0.7]`.
- Sparse topology: `random_regular` with `degree = [3, 5, 8]`.
- Objective: compare `weighted_vertices = [false, true]`.
- Value range: `max_vertex_value = [5, 10, 100]`.
- Penalty study: first generate with automatic penalty (P), then compare approximately
  `[0.5P, P, 2P, 5P]`. Values below the automatic penalty may favor infeasible states and
  are useful only when intentionally studying penalty sensitivity.

## 3. Maximum Clique

**Problem type:** `maximum_clique`

Maximum Clique selects a largest or highest-value group of mutually adjacent vertices.
One binary variable selects each vertex. The QUBO minimizes negative selected value plus
a penalty for every selected pair that is not connected by an edge.

| Parameter | Type | Default | Meaning for Maximum Clique |
|---|---:|---:|---|
| `num_vertices` | integer ≥ 2 | `20` | Number of vertices that may belong to the clique; also the number of QUBO variables. |
| `graph_type` | choice | `erdos_renyi` | Structure of the graph in which a clique is sought: `erdos_renyi`, `random_regular`, `cycle`, or `complete`. |
| `edge_probability` | number in [0, 1] | `0.25` | Probability that a vertex pair is compatible for the clique when `graph_type` is `erdos_renyi`; ignored by other graph types. |
| `degree` | integer ≥ 1 | `3` | Number of neighbors per vertex for `random_regular`; ignored by other graph types. It must be smaller than `num_vertices`, and their product must be even. |
| `weighted` | boolean | `false` | Controls weights stored on graph edges. Edge weights do not affect the current Clique QUBO. |
| `min_edge_weight` | integer ≥ 1 | `1` | Minimum stored edge weight when `weighted` is true; it does not affect clique value. |
| `max_edge_weight` | integer ≥ 1 | `10` | Maximum stored edge weight when `weighted` is true; it must not be smaller than `min_edge_weight` and does not affect clique value. |
| `weighted_vertices` | boolean | `false` | If true, maximize total generated vertex value; otherwise maximize clique size. |
| `min_vertex_value` | integer ≥ 1 | `1` | Smallest selected-vertex contribution when `weighted_vertices` is true. |
| `max_vertex_value` | integer ≥ 1 | `10` | Largest selected-vertex contribution when `weighted_vertices` is true; it must not be smaller than `min_vertex_value`. |
| `penalty` | number ≥ 0 or automatic | automatic | Energy added for every selected vertex pair that is not an edge. Automatic uses the largest generated vertex value plus 1. |

**Suggested sweeps:**

- Size: `num_vertices = [20, 50, 100, 200]`.
- Erdős–Rényi density: `edge_probability = [0.1, 0.3, 0.5, 0.7, 0.9]`.
- Include high densities because clique structure changes strongly as edges are added.
- Objective: compare `weighted_vertices = [false, true]` and
  `max_vertex_value = [5, 10, 100]`.
- Penalty study: automatic (P), then `[0.5P, P, 2P, 5P]` if penalty behavior is part of
  the benchmark.

Clique QUBOs penalize graph non-edges, so a sparse original graph produces a dense QUBO.
This is worth tracking when comparing instances of the same vertex count.

## 4. Graph Coloring

**Problem type:** `graph_coloring`

Graph Coloring assigns exactly one color to every vertex while preventing adjacent
vertices from sharing a color. It uses one-hot variables: a graph with (n) vertices and
(k) colors produces (n \times k) QUBO variables. Energy is the sum of one-hot and
same-color edge penalties; a valid coloring has zero energy.

| Parameter | Type | Default | Meaning for Graph Coloring |
|---|---:|---:|---|
| `num_vertices` | integer ≥ 2 | `20` | Number of vertices to color. Together with `num_colors`, it determines the QUBO size. |
| `graph_type` | choice | `erdos_renyi` | Structure of the graph to color: `erdos_renyi`, `random_regular`, `cycle`, or `complete`. |
| `edge_probability` | number in [0, 1] | `0.25` | Probability that a vertex pair must receive different colors for `erdos_renyi`; ignored by other graph types. |
| `degree` | integer ≥ 1 | `3` | Number of differently constrained neighbors per vertex for `random_regular`; ignored by other graph types. It must be smaller than `num_vertices`, and their product must be even. |
| `weighted` | boolean | `false` | Controls weights stored on graph edges. The current coloring QUBO treats every adjacency constraint equally. |
| `min_edge_weight` | integer ≥ 1 | `1` | Minimum stored edge weight when `weighted` is true; it does not change coloring penalties. |
| `max_edge_weight` | integer ≥ 1 | `10` | Maximum stored edge weight when `weighted` is true; it must not be smaller than `min_edge_weight` and does not change coloring penalties. |
| `num_colors` | integer ≥ 1 | `3` | Number of colors available to every vertex. The QUBO has `num_vertices * num_colors` variables. |
| `one_color_penalty` | number ≥ 0 | `2.0` | Energy weight for assigning a vertex zero colors or more than one color. |
| `edge_penalty` | number ≥ 0 | `1.0` | Energy added when two adjacent vertices are assigned the same color. |

**Suggested sweeps:**

- Graph size: `num_vertices = [10, 20, 50, 100]`.
- Colors: `num_colors = [2, 3, 4, 5]`.
- Erdős–Rényi density: `edge_probability = [0.05, 0.1, 0.25, 0.5]`.
- Regular graphs: `degree = [2, 3, 4, 6]` with valid size/degree combinations.
- Penalty ratio: hold `edge_penalty = 1` and use
  `one_color_penalty = [0.5, 1, 2, 5, 10]`.

Color count should be swept near the graph's likely chromatic number. Very many colors
make feasibility easy while greatly increasing the QUBO variable count.

## 5. Number Partitioning

**Problem type:** `number_partitioning`

Number Partitioning divides positive integers into two groups whose sums are as close as
possible. There is one binary variable per number. QUBO energy equals the square of the
difference between the two partition sums, so zero is a perfect partition.

| Parameter | Type | Default | Meaning for Number Partitioning |
|---|---:|---:|---|
| `num_numbers` | integer ≥ 2 | `20` | Number of integers to partition. |
| `min_value` | integer ≥ 1 | `1` | Smallest generated integer. |
| `max_value` | integer ≥ 1 | `100` | Largest generated integer; it must not be smaller than the minimum. |

**Suggested sweeps:**

- Size: `num_numbers = [10, 20, 50, 100, 200]`.
- Numeric range: `max_value = [10, 100, 1_000, 1_000_000]` with `min_value = 1`.
- Narrow versus broad values: compare `[min_value, max_value]` ranges such as `[90, 100]`
  and `[1, 100]`.

Increasing the numeric range stresses coefficient precision even when the QUBO variable
count stays fixed. This should be treated as a separate benchmark dimension from size.

## 6. Quadratic Knapsack

**Problem type:** `quadratic_knapsack`

Quadratic Knapsack selects items under a weight capacity. Selected items contribute linear
values, and selected pairs may contribute additional bonuses. Binary slack variables
represent unused capacity. Energy is negative total value plus the squared capacity
residual multiplied by a penalty.

| Parameter | Type | Default | Meaning for Quadratic Knapsack |
|---|---:|---:|---|
| `num_items` | integer ≥ 1 | `15` | Number of candidate items. |
| `min_weight` | integer ≥ 1 | `1` | Minimum item weight. |
| `max_weight` | integer ≥ 1 | `20` | Maximum item weight. |
| `min_value` | integer ≥ 1 | `1` | Minimum linear item value. |
| `max_value` | integer ≥ 1 | `30` | Maximum linear item value. |
| `quadratic_density` | number in [0, 1] | `0.25` | Probability that an item pair receives a bonus. |
| `min_quadratic_value` | integer ≥ 1 | `1` | Minimum pairwise bonus. |
| `max_quadratic_value` | integer ≥ 1 | `10` | Maximum pairwise bonus. |
| `capacity_ratio` | number in [0.01, 1] | `0.5` | Capacity divided by total generated item weight. |
| `penalty` | number ≥ 0 or automatic | automatic | Capacity penalty. Automatic uses the sum of all possible positive values plus 1. |

Every minimum value must be no larger than its corresponding maximum.

**Suggested sweeps:**

- Items: `num_items = [10, 20, 50, 100]`.
- Capacity pressure: `capacity_ratio = [0.2, 0.35, 0.5, 0.7, 0.9]`.
- Pair structure: `quadratic_density = [0, 0.1, 0.25, 0.5, 1.0]`.
- Weight range: `max_weight = [10, 20, 100]`.
- Bonus strength: `max_quadratic_value = [1, 10, 30]` relative to
  `max_value = 30`.
- Penalty study: automatic (P), followed by `[0.25P, 0.5P, P, 2P]` when intentionally
  measuring objective/feasibility competition.

The total QUBO size is the item count plus capacity slack variables. Larger capacities
therefore change both coefficient magnitudes and the number of variables.

## 7. Set Packing

**Problem type:** `set_packing`

Set Packing chooses a maximum-value collection of mutually disjoint sets. There is one
binary variable per candidate set. Energy is negative selected value plus one penalty for
each selected pair that overlaps.

| Parameter | Type | Default | Meaning for Set Packing |
|---|---:|---:|---|
| `universe_size` | integer ≥ 1 | `20` | Number of distinct elements that sets can contain. |
| `num_sets` | integer ≥ 1 | `30` | Number of candidate sets and QUBO variables. |
| `inclusion_probability` | number in [0, 1] | `0.15` | Probability that an element is included in a generated set. Empty generated sets are replaced by a random singleton. |
| `weighted_sets` | boolean | `true` | Assign random positive set values; otherwise every set has value 1. |
| `min_set_value` | integer ≥ 1 | `1` | Minimum generated set value. |
| `max_set_value` | integer ≥ 1 | `10` | Maximum generated set value; it must not be smaller than the minimum. |
| `penalty` | number ≥ 0 or automatic | automatic | Overlap penalty. Automatic uses `max_set_value_in_instance + 1`. |

**Suggested sweeps:**

- QUBO size: `num_sets = [20, 50, 100, 200]`.
- Universe size: `universe_size = [10, 20, 50, 100]`.
- Incidence density: `inclusion_probability = [0.02, 0.05, 0.1, 0.2, 0.4]`.
- Set-to-element ratio: compare approximately `num_sets / universe_size = [0.5, 1, 2, 5]`.
- Objective: `weighted_sets = [false, true]` and `max_set_value = [5, 10, 100]`.
- Penalty study: automatic (P), then `[0.5P, P, 2P, 5P]`.

Higher inclusion probability creates more overlaps and therefore a denser QUBO.

## 8. SAT / Max-SAT

**Problem type:** `max_sat`

The generator creates weighted or unweighted 2-SAT and 3-SAT formulas. The objective is
to minimize the total weight of unsatisfied clauses. A 2-SAT instance uses one QUBO
variable per Boolean variable. A 3-SAT instance adds one ancilla per clause to reduce the
cubic unsatisfied-clause expression to a quadratic form.

| Parameter | Type | Default | Meaning for SAT / Max-SAT |
|---|---:|---:|---|
| `num_variables` | integer ≥ 2 | `20` | Number of primary Boolean variables. It must be at least `clause_size`. |
| `num_clauses` | integer ≥ 1 | `80` | Number of generated clauses. |
| `clause_size` | `2` or `3` | `3` | Number of distinct literals in every clause. |
| `weighted_clauses` | boolean | `false` | Assign random positive integer clause weights. |
| `min_clause_weight` | integer ≥ 1 | `1` | Minimum generated clause weight. |
| `max_clause_weight` | integer ≥ 1 | `10` | Maximum clause weight; it must not be smaller than the minimum. |
| `planted_solution` | boolean | `true` | Modify each clause so a hidden generated assignment satisfies it. |
| `ancilla_penalty` | number ≥ 0 or automatic | automatic | Product-consistency penalty for 3-SAT. Automatic uses each clause's weight plus 1. It is unused for 2-SAT. |

**Suggested sweeps:**

- Primary variables: `num_variables = [10, 20, 50, 100]`.
- Clause width: `clause_size = [2, 3]`.
- Clause-to-variable ratio for 2-SAT: approximately `[0.5, 1, 2, 4]`.
- Clause-to-variable ratio for 3-SAT: approximately `[2, 3, 4, 4.3, 5, 8]`.
  Set `num_clauses` to the rounded ratio times `num_variables`.
- Satisfiability construction: `planted_solution = [true, false]`.
- Clause weights: compare unweighted clauses with weighted ranges `[1, 5]`, `[1, 10]`,
  and `[1, 100]`.
- Ancilla penalty study for 3-SAT: automatic (P), then approximately
  `[0.5P, P, 2P, 5P]`.

The planted generator guarantees at least one satisfying assignment and does not have the
same distribution as unrestricted random SAT. Keep planted and unplanted results separate.
Also remember that 3-SAT produces `num_variables + num_clauses` QUBO variables.

## 9. Spin Glass

**Problem type:** `spin_glass`

Spin Glass creates a native Ising objective on a graph,

```text
H(s) = sum_i h_i*s_i + sum_{i<j} J_ij*s_i*s_j,
```

then converts it exactly to QUBO using `s_i = 2*x_i - 1`. There is one binary variable per
spin, and QUBO energy equals the original Ising energy.

| Parameter | Type | Default | Meaning for Spin Glass |
|---|---:|---:|---|
| `num_vertices` | integer ≥ 2 | `20` | Number of spins. |
| `graph_type` | choice | `erdos_renyi` | One of `erdos_renyi`, `random_regular`, `cycle`, or `complete`. |
| `edge_probability` | number in [0, 1] | `0.25` | Interaction probability for `erdos_renyi`. |
| `degree` | integer ≥ 1 | `3` | Number of interactions per spin for `random_regular`; ignored by other graph types. It must be smaller than `num_vertices`, and their product must be even. |
| `coupling_distribution` | choice | `bimodal` | `bimodal` produces ±scale couplings; `gaussian` draws zero-mean normal couplings. |
| `coupling_scale` | number > 0 | `1.0` | Bimodal magnitude or Gaussian standard deviation for (J). |
| `field_probability` | number in [0, 1] | `0.0` | Probability that a spin receives a nonzero local field. |
| `field_scale` | number > 0 | `1.0` | Magnitude of each nonzero local field (h). |

**Suggested sweeps:**

- Spins: `num_vertices = [20, 50, 100, 200]`.
- Topology: `cycle`, random regular with `degree = [3, 4, 6]`, Erdős–Rényi with
  `edge_probability = [0.05, 0.1, 0.25, 0.5]`, and `complete` as a dense control.
- Couplings: `coupling_distribution = [bimodal, gaussian]`.
- Coupling scale: `coupling_scale = [0.1, 1, 10]`.
- Field prevalence: `field_probability = [0, 0.1, 0.5, 1.0]`.
- Relative field strength: `field_scale / coupling_scale = [0.1, 1, 10]`.

Changing all coefficients by the same scale does not change the mathematical minimizer,
but it can expose numerical scaling sensitivity in downstream solvers.

## 10. Random QUBO

**Problem type:** `random_qubo`

Random QUBO directly generates linear and pairwise coefficients without an underlying
combinatorial interpretation. It is useful for testing raw QUBO optimization behavior.

| Parameter | Type | Default | Meaning for Random QUBO |
|---|---:|---:|---|
| `num_variables` | integer ≥ 1 | `50` | Number of binary variables. |
| `quadratic_density` | number in [0, 1] | `0.25` | Probability that a variable pair receives a quadratic coefficient. |
| `distribution` | choice | `integer` | `integer`, `uniform`, or zero-mean `normal`. |
| `coefficient_min` | number | `-10.0` | Lower bound for integer and uniform coefficients. It must be an integer when using the integer distribution. |
| `coefficient_max` | number | `10.0` | Upper bound for integer and uniform coefficients; it must not be smaller than the minimum. |
| `normal_scale` | number > 0 | `5.0` | Standard deviation when `distribution` is `normal`; bounds are then unused. |
| `zero_linear_probability` | number in [0, 1] | `0.0` | Probability that an individual linear coefficient is set to zero. |

**Suggested sweeps:**

- Size: `num_variables = [20, 50, 100, 200, 500]`.
- Quadratic density: `quadratic_density = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]`.
- Distribution: compare `integer`, `uniform`, and `normal`.
- Coefficient range for integer/uniform: `[-1, 1]`, `[-10, 10]`, and `[-100, 100]`.
- Normal scale: `normal_scale = [0.1, 1, 10, 100]`.
- Linear-field sparsity: `zero_linear_probability = [0, 0.5, 1.0]`.

Size and density should be swept together carefully: a dense (n)-variable QUBO can
contain (n(n-1)/2) quadratic interactions.

## Recommended starter collection

For an initial balanced dataset, use 10 seeds for every selected combination:

| Family | Starter dimensions |
|---|---|
| Max-Cut | `num_vertices = [20, 50, 100]`, `edge_probability = [0.1, 0.5]` |
| Independent Set | `num_vertices = [20, 50, 100]`, `edge_probability = [0.1, 0.4]` |
| Maximum Clique | `num_vertices = [20, 50, 100]`, `edge_probability = [0.3, 0.7]` |
| Graph Coloring | `num_vertices = [10, 20, 50]`, `num_colors = [3, 4]` |
| Number Partitioning | `num_numbers = [20, 50, 100]`, `max_value = [100, 10_000]` |
| Quadratic Knapsack | `num_items = [10, 20, 50]`, `capacity_ratio = [0.3, 0.7]` |
| Set Packing | `num_sets = [20, 50, 100]`, `inclusion_probability = [0.05, 0.2]` |
| Max-SAT | `num_variables = [10, 20, 50]`, `clause_size = [2, 3]`, clause ratio `[1, 4]` respectively |
| Spin Glass | `num_vertices = [20, 50, 100]`, topology `random_regular`, `degree = [3, 6]` |
| Random QUBO | `num_variables = [20, 50, 100]`, `quadratic_density = [0.05, 0.5]` |

Keep automatic penalties for this first collection. Penalty-strength experiments are best
stored as a separate collection so they do not become mixed with ordinary problem-size
and density comparisons.
