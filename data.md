# Data Model

The task of each model is to fully close a given formal statement which may be any goal or subgoal part of a larger proof, given the formal statement to close, the formal proof history so far, and the entirety of the corresponding informal statement and informal proof.

To achieve the desired result, data is categorized as geometry or non-geometry, and formatted into Qwen's training representation ChatML after being transformed into the appropriate format and cleaned.

## Model for Subgoal Closing

Each data element must contain the global formal context, full informal context, and specified subgoal at the time of creation of the subgoal. In addition, metadata about the source of the data element is included.

| Column | Description |
| :--- | :--- |
| **`item_id`** | A UUIDv4 string acting as the absolute unique identifier for this specific extracted subgoal. |
| **`source_dataset`** | The categorical origin of the data (e.g., `herald`, `forml4`, `mathlib`). |
| **`original_id`** | A composite grouping key (e.g., `module::theorem_name`) pointing to the original source. Used to enforce theorem-level train/test splits. |
| **`informal_context`** | The full informal theorem statement and the entirety of the corresponding human proof sketch. |
| **`formal_context_with_sorry`** | The complete, standalone Lean code for the theorem, containing exactly one sorry that the model is intended to fill in. If this string is passed to Lean, it must compile with only a "declaration uses sorry" warning. |
| **`formal_history`** | The formal Lean code written from the theorem declaration up to the current subgoal, providing structural context. |
| **`tactic_state`** | The exact local context, hypotheses, and goal active at the precise location of the sorry. |
| **`formal_completion`** | A complete Lean 4 tactic block (e.g., a `by` block or multiple tactics) that **closes** the subgoal currently completed by `sorry` (leaving no remaining goals for this branch). |
| **`is_geometry`** | Boolean flag indicating whether this problem belongs to the geometry domain. |

## Qwen Training Representation (ChatML)

Each training example is framed as a complete Lean 4 file containing exactly one sorry. The model is provided with the precise tactic state at that location and is tasked with generating the exact tactic block required to replace the sorry and close the goal.

### Prompt Template

```text
<|im_start|>system
You are an expert Lean 4 theorem prover. Your task is to complete the formal proof by providing the tactic or sequence of tactics required to complete the proof from the `sorry`'s position and close the goal.<|im_end|>
<|im_start|>user
INFORMAL CONTEXT:
[INFORMAL CONTEXT]

CURRENT TACTIC STATE:
[TACTIC STATE]

LEAN CODE:
[FORMAL CONTEXT WITH SORRY]
<|im_end|>
<|im_start|>assistant
[FORMAL COMPLETION]<|im_end|>
```

## JSONL Example

The following example demonstrates a "closed" subgoal for a probability measure proof. Note that the assistant response is expected to be a complete solution for the goal provided, and the formal subgoal includes the local context.

```JSON
{
  "messages": [
    {
      "role": "system", 
      "content": "You are an expert Lean 4 theorem prover. Your task is to complete the formal proof by providing the exact tactic block required to replace the `sorry` in the provided Lean code and close the specified goal."
    },
    {
      "role": "user", 
      "content": "INFORMAL CONTEXT:\nStatement: Prove that the probability of a countable union of disjoint sets is the sum of their probabilities.\nProof: By the definition of a measure, since the sets A_i are pairwise disjoint, the measure of their union is countably additive...\n\nCURRENT TACTIC STATE:\nα : Type u_1\n_inst_1 : MeasurableSpace α\nP : Measure α\n_inst_2 : IsProbabilityMeasure P\nA : ℕ → Set α\nh_meas : ∀ (i : ℕ), MeasurableSet (A i)\nh_disjoint : Pairwise (Disjoint on A)\n⊢ P (⋃ (i : ℕ), A i) = ∑' (i : ℕ), P (A i)\n\nLEAN CODE:\nimport Mathlib.MeasureTheory.Measure.MeasureSpace\n\nopen MeasureTheory\n\ntheorem prob_union_disjoint {α : Type*} [MeasurableSpace α] (P : Measure α) [IsProbabilityMeasure P]\n    (A : ℕ → Set α) (h_meas : ∀ i, MeasurableSet (A i)) (h_disjoint : Pairwise (Disjoint on A)) :\n    P (⋃ i, A i) = ∑' i, P (A i) := by\n  sorry"
    },
    {
      "role": "assistant", 
      "content": "rw [measure_iUnion h_disjoint h_meas]"
    }
  ],
  "is_geometry": false
}
```

# Data Curation

## Version Standardization

Lean 4 does not guarantee compatibility between versions. We pin all data to `v4.11.0`, which is the version that the Herald dataset uses. Any examples from other datasets that do not compile in this version are be thrown out.

## Subgoal Extraction

Our dataset contains columns with incomplete formal proofs and reference to the specific formal subgoal the model is intended to solve. None of the datasets we pull from include this level of detail. To generate it from other datasets'  `.lean` files, we use LeanDojo to "trace" the formal proof: as it steps through the theorem, we capture the existing code upon creation of a new subgoal, which is then paired alongside the completed subgoal to create a single row. Any extracted subgoal where the formal completion contains the sorry tactic is discarded.

## Herald

The [Herald_proofs](https://huggingface.co/datasets/FrenzyMath/Herald_proofs) dataset contains informal statements and formal proofs. To transform each datapoint into multiple training examples, we use `/scripts/process/process_herald.py`, which traces the formal proof state and injects `sorry` when new goals appear. When many locations are available for injection of a `sorry` to create a training example, a maximum of 5 are created from a single proof in the Herald dataset.

### Example

A Herald data point looks like this:
```JSON
{
   "id":703,
   "name":"dist_mul_mul_le",
   "formal_theorem":"theorem dist_mul_mul_le (a\u2081 a\u2082 b\u2081 b\u2082 : E) : dist (a\u2081 * a\u2082) (b\u2081 * b\u2082) \u2264 dist a\u2081 b\u2081 + dist a\u2082 b\u2082 := by sorry",
   "informal_theorem":"Distance Inequality for Product in Seminormed Commutative Groups: \\(\\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1, b_1) + \\text{dist}(a_2, b_2)\\) : For any elements \\( a_1, a_2, b_1, b_2 \\) in a seminormed commutative group \\( E \\), the distance between the product \\( a_1 \\cdot a_2 \\) and the product \\( b_1 \\cdot b_2 \\) is less than or equal to the sum of the distances between \\( a_1 \\) and \\( b_1 \\) and between \\( a_2 \\) and \\( b_2 \\). Mathematically, this can be expressed as:\n\\[ \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1, b_1) + \\text{dist}(a_2, b_2) \\]",
   "formal_proof":"theorem dist_mul_mul_le (a\u2081 a\u2082 b\u2081 b\u2082 : E) : dist (a\u2081 * a\u2082) (b\u2081 * b\u2082) \u2264 dist a\u2081 b\u2081 + dist a\u2082 b\u2082 := by\n  simpa only [dist_mul_left, dist_mul_right] using dist_triangle (a\u2081 * a\u2082) (b\u2081 * a\u2082) (b\u2081 * b\u2082)",
   "informal_proof":"To prove the inequality \\(\\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1, b_1) + \\text{dist}(a_2, b_2)\\), we start by applying the triangle inequality:\n\\[ \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot a_2) + \\text{dist}(b_1 \\cdot a_2, b_1 \\cdot b_2). \\]\n\nNext, we use the isometric action property of the seminormed commutative group. This property states that for any elements \\( a, b, c \\in E \\), the distance \\(\\text{dist}(a \\cdot c, b \\cdot c)\\) is equal to \\(\\text{dist}(a, b)\\). Applying this property, we get:\n\\[ \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot a_2) = \\text{dist}(a_1, b_1). \\]\n\nSimilarly, we apply the isometric action property to the second term:\n\\[ \\text{dist}(b_1 \\cdot a_2, b_1 \\cdot b_2) = \\text{dist}(a_2, b_2). \\]\n\nSubstituting these results back into the triangle inequality, we obtain:\n\\[ \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1, b_1) + \\text{dist}(a_2, b_2). \\]\n\nThus, the inequality is established, and the proof is complete.",
   "commented_proof":"theorem dist_mul_mul_le (a\u2081 a\u2082 b\u2081 b\u2082 : E) : dist (a\u2081 * a\u2082) (b\u2081 * b\u2082) \u2264 dist a\u2081 b\u2081 + dist a\u2082 b\u2082 := by\n\/- Using the triangle inequality, we have:\n\\[ \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot a_2) + \\text{dist}(b_1 \\cdot a_2, b_1 \\cdot b_2). \\]\nBy the isometric action property, the distance \\(\\text{dist}(a_1 \\cdot a_2, b_1 \\cdot a_2)\\) is equal to \\(\\text{dist}(a_2, a_2)\\), which is \\(0\\). Similarly, the distance \\(\\text{dist}(b_1 \\cdot a_2, b_1 \\cdot b_2)\\) is equal to \\(\\text{dist}(a_2, b_2)\\). Therefore, we have:\n\\[ \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq 0 + \\text{dist}(a_2, b_2) = \\text{dist}(a_2, b_2). \\]\nAdditionally, by the isometric action property, the distance \\(\\text{dist}(a_1 \\cdot a_2, b_1 \\cdot a_2)\\) is equal to \\(\\text{dist}(a_1, b_1)\\). Combining these results, we get:\n\\[ \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1, b_1) + \\text{dist}(a_2, b_2). \\]\nThis completes the proof. -\/\n  simpa only [dist_mul_left, dist_mul_right] using dist_triangle (a\u2081 * a\u2082) (b\u2081 * a\u2082) (b\u2081 * b\u2082)",
   "header":"Unable to analyze"
}
```
In this case, the proof has exactly one step, so there is one location we can place a `sorry` to create a corresponding data point, that looks like this:
```JSON
{
  "item_id": "<UUID>",
  "source_dataset": "herald",
  "original_id": "Mathlib.Topology.MetricSpace.PseudoMetric::dist_mul_mul_le",
  "informal_context": "Distance Inequality for Product in Seminormed Commutative Groups: \\(\\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1, b_1) + \\text{dist}(a_2, b_2)\\) : For any elements \\( a_1, a_2, b_1, b_2 \\) in a seminormed commutative group \\( E \\), the distance between the product \\( a_1 \\cdot a_2 \\) and the product \\( b_1 \\cdot b_2 \\) is less than or equal to the sum of the distances between \\( a_1 \\) and \\( b_1 \\) and between \\( a_2 \\) and \\( b_2 \\). Mathematically, this can be expressed as:\n\\[ \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1, b_1) + \\text{dist}(a_2, b_2) \\]\n\nTo prove the inequality \\(\\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1, b_1) + \\text{dist}(a_2, b_2)\\), we start by applying the triangle inequality:\n\\[ \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot a_2) + \\text{dist}(b_1 \\cdot a_2, b_1 \\cdot b_2). \\]\n\nNext, we use the isometric action property of the seminormed commutative group. This property states that for any elements \\( a, b, c \\in E \\), the distance \\(\\text{dist}(a \\cdot c, b \\cdot c)\\) is equal to \\(\\text{dist}(a, b)\\). Applying this property, we get:\n\\[ \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot a_2) = \\text{dist}(a_1, b_1). \\]\n\nSimilarly, we apply the isometric action property to the second term:\n\\[ \\text{dist}(b_1 \\cdot a_2, b_1 \\cdot b_2) = \\text{dist}(a_2, b_2). \\]\n\nSubstituting these results back into the triangle inequality, we obtain:\n\\[ \\text{dist}(a_1 \\cdot a_2, b_1 \\cdot b_2) \\leq \\text{dist}(a_1, b_1) + \\text{dist}(a_2, b_2). \\]\n\nThus, the inequality is established, and the proof is complete.",
  "formal_context_with_sorry": "import Mathlib.Topology.MetricSpace.PseudoMetric\n\nvariable {E : Type*} [SeminormedCommGroup E]\n\ntheorem dist_mul_mul_le (a₁ a₂ b₁ b₂ : E) : dist (a₁ * a₂) (b₁ * b₂) ≤ dist a₁ b₁ + dist a₂ b₂ := by sorry",
  "formal_history": "theorem dist_mul_mul_le (a₁ a₂ b₁ b₂ : E) : dist (a₁ * a₂) (b₁ * b₂) ≤ dist a₁ b₁ + dist a₂ b₂ :=",
  "tactic_state": "E : Type u_1\ninst✝ : SeminormedCommGroup E\na₁ a₂ b₁ b₂ : E\n⊢ dist (a₁ * a₂) (b₁ * b₂) ≤ dist a₁ b₁ + dist a₂ b₂",
  "formal_completion": "simpa only [dist_mul_left, dist_mul_right] using dist_triangle (a₁ * a₂) (b₁ * a₂) (b₁ * b₂)",
  "is_geometry": false
}
```

If a proof contains multiple or many steps, we create multipe data points, up to a certain maximum number. If that maximum is exceeded, we pick randomly between possible options.

For this Herald item:
```JSON
{
   "id":27053,
   "name":"ProbabilityTheory.Kernel.measurable_rnDeriv",
   "formal_theorem":"theorem ProbabilityTheory.Kernel.measurable_rnDeriv (\u03ba \u03b7 : Kernel \u03b1 \u03b3) :\n    Measurable (fun p : \u03b1 \u00d7 \u03b3 \u21a6 rnDeriv \u03ba \u03b7 p.1 p.2) := by sorry",
   "informal_theorem":"Measurability of Radon-Nikodym Derivative of Kernels : For any measurable spaces \\(\\alpha\\) and \\(\\gamma\\) where either \\(\\alpha\\) is countable or \\(\\gamma\\) is countably generated, and for any finite kernels \\(\\kappa, \\eta : \\text{Kernel}(\\alpha, \\gamma)\\), the function \\(\\kappa \\, \\text{rnDeriv} \\, \\eta : \\alpha \\times \\gamma \\to \\mathbb{R}_{\\geq 0} \\cup \\{\\infty\\}\\) is measurable. This means that for any measurable set \\(t \\subseteq \\mathbb{R}_{\\geq 0} \\cup \\{\\infty\\}\\), the preimage of \\(t\\) under the function \\(\\kappa \\, \\text{rnDeriv} \\, \\eta\\) is a measurable set in \\(\\alpha \\times \\gamma\\).",
   "formal_proof":"theorem ProbabilityTheory.Kernel.measurable_rnDeriv (\u03ba \u03b7 : Kernel \u03b1 \u03b3) :\n    Measurable (fun p : \u03b1 \u00d7 \u03b3 \u21a6 rnDeriv \u03ba \u03b7 p.1 p.2) := by\n  simp_rw [rnDeriv_def]\n  exact (measurable_rnDerivAux \u03ba _).ennreal_ofReal.div\n    (measurable_const.sub (measurable_rnDerivAux \u03ba _)).ennreal_ofReal",
   "informal_proof":"We start by using the definition of the Radon-Nikodym derivative of kernels. The Radon-Nikodym derivative \\(\\kappa \\, \\text{rnDeriv} \\, \\eta \\, a \\, x\\) is given by:\n\\[\n\\kappa \\, \\text{rnDeriv} \\, \\eta \\, a \\, x = \\frac{\\text{ENNReal.ofReal}(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}{\\text{ENNReal.ofReal}(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}\n\\]\nTo show that the function \\((a, x) \\mapsto \\frac{\\text{ENNReal.ofReal}(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}{\\text{ENNReal.ofReal}(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}\\) is measurable, we proceed as follows:\n1. The function \\(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x\\) is jointly measurable on \\(\\alpha \\times \\gamma\\) by the theorem `measurable_rnDerivAux`.\n2. The function \\(\\text{ENNReal.ofReal}(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)\\) is measurable by the theorem `Measurable.ennreal_ofReal`.\n3. The constant function \\(1\\) is measurable by the theorem `measurable_const`.\n4. The function \\(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x\\) is measurable by the theorem `Measurable.sub`.\n5. The function \\(\\text{ENNReal.ofReal}(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)\\) is measurable by the theorem `Measurable.ennreal_ofReal`.\n6. The function \\(\\frac{\\text{ENNReal.ofReal}(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}{\\text{ENNReal.ofReal}(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}\\) is measurable by the theorem `Measurable.div`.\n\nThus, the function \\((a, x) \\mapsto \\frac{\\text{ENNReal.ofReal}(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}{\\text{ENNReal.ofReal}(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}\\) is measurable, which completes the proof.",
   "commented_proof":"theorem ProbabilityTheory.Kernel.measurable_rnDeriv (\u03ba \u03b7 : Kernel \u03b1 \u03b3) :\n    Measurable (fun p : \u03b1 \u00d7 \u03b3 \u21a6 rnDeriv \u03ba \u03b7 p.1 p.2) := by\n\/- First, we use the definition of the Radon-Nikodym derivative of kernels to rewrite the goal. The Radon-Nikodym derivative \\(\\kappa \\, \\text{rnDeriv} \\, \\eta \\, a \\, x\\) is given by:\n\\[\n\\kappa \\, \\text{rnDeriv} \\, \\eta \\, a \\, x = \\frac{\\text{ENNReal.ofReal}(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}{\\text{ENNReal.ofReal}(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}\n\\]\nThus, the goal is to show that the function \\((a, x) \\mapsto \\frac{\\text{ENNReal.ofReal}(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}{\\text{ENNReal.ofReal}(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}\\) is measurable. -\/\n  simp_rw [rnDeriv_def]\n\/- To show that the function \\((a, x) \\mapsto \\frac{\\text{ENNReal.ofReal}(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}{\\text{ENNReal.ofReal}(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}\\) is measurable, we use the following steps:\n1. The function \\(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x\\) is jointly measurable on \\(\\alpha \\times \\gamma\\) by the theorem `measurable_rnDerivAux`.\n2. The function \\(\\text{ENNReal.ofReal}(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)\\) is measurable by the theorem `Measurable.ennreal_ofReal`.\n3. The constant function \\(1\\) is measurable by the theorem `measurable_const`.\n4. The function \\(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x\\) is measurable by the theorem `Measurable.sub`.\n5. The function \\(\\text{ENNReal.ofReal}(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)\\) is measurable by the theorem `Measurable.ennreal_ofReal`.\n6. The function \\(\\frac{\\text{ENNReal.ofReal}(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}{\\text{ENNReal.ofReal}(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}\\) is measurable by the theorem `Measurable.div`.\n\nThus, the function \\((a, x) \\mapsto \\frac{\\text{ENNReal.ofReal}(\\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}{\\text{ENNReal.ofReal}(1 - \\kappa \\, \\text{rnDerivAux} \\, (\\kappa + \\eta) \\, a \\, x)}\\) is measurable, which completes the proof. -\/\n  exact (measurable_rnDerivAux \u03ba _).ennreal_ofReal.div\n    (measurable_const.sub (measurable_rnDerivAux \u03ba _)).ennreal_ofReal",
   "header":"import Mathlib.Probability.Kernel.Disintegration.Density\nimport Mathlib.Probability.Kernel.WithDensity\nimport Mathlib.Probability.Kernel.RadonNikodym\n\nopen ProbabilityTheory\nopen Kernel\nopen MeasureTheory Set Filter ENNReal\nopen scoped NNReal MeasureTheory Topology ProbabilityTheory\nvariable {\u03b1 \u03b3 : Type*} {m\u03b1 : MeasurableSpace \u03b1} {m\u03b3 : MeasurableSpace \u03b3} {\u03ba \u03b7 : Kernel \u03b1 \u03b3}\n  [h\u03b1\u03b3 : MeasurableSpace.CountableOrCountablyGenerated \u03b1 \u03b3]\nopen Classical in\n"
}
```
We would produce two data points:
```JSON
{
  "item_id": "<UUID-1>",
  "source_dataset": "herald",
  "original_id": "Mathlib.Probability.Kernel.RadonNikodym::ProbabilityTheory.Kernel.measurable_rnDeriv",
  "informal_context": "Measurability of Radon-Nikodym Derivative of Kernels...",
  "formal_context_with_sorry": "import Mathlib.Probability.Kernel.Disintegration.Density\n... [header] ...\ntheorem ProbabilityTheory.Kernel.measurable_rnDeriv (κ η : Kernel α γ) :\n    Measurable (fun p : α × γ ↦ rnDeriv κ η p.1 p.2) := by sorry",
  "formal_history": "theorem ProbabilityTheory.Kernel.measurable_rnDeriv (κ η : Kernel α γ) :\n    Measurable (fun p : α × γ ↦ rnDeriv κ η p.1 p.2) :=",
  "tactic_state": "α γ : Type u_1\n... [variables] ...\nκ η : Kernel α γ\n⊢ Measurable (fun p ↦ rnDeriv κ η p.1 p.2)",
  "formal_completion": "simp_rw [rnDeriv_def]\nexact (measurable_rnDerivAux κ _).ennreal_ofReal.div (measurable_const.sub (measurable_rnDerivAux κ _)).ennreal_ofReal",
  "is_geometry": false
}
```
```JSON
{
  "item_id": "<UUID-2>",
  "source_dataset": "herald",
  "original_id": "Mathlib.Probability.Kernel.RadonNikodym::ProbabilityTheory.Kernel.measurable_rnDeriv",
  "informal_context": "Measurability of Radon-Nikodym Derivative of Kernels...",
  "formal_context_with_sorry": "import Mathlib.Probability.Kernel.Disintegration.Density\n... [header] ...\ntheorem ProbabilityTheory.Kernel.measurable_rnDeriv (κ η : Kernel α γ) :\n    Measurable (fun p : α × γ ↦ rnDeriv κ η p.1 p.2) := by\n  simp_rw [rnDeriv_def]\n  sorry",
  "formal_history": "theorem ProbabilityTheory.Kernel.measurable_rnDeriv (κ η : Kernel α γ) :\n    Measurable (fun p : α × γ ↦ rnDeriv κ η p.1 p.2) := by\n  simp_rw [rnDeriv_def]",
  "tactic_state": "α γ : Type u_1\n... [variables] ...\nκ η : Kernel α γ\n⊢ Measurable (fun p ↦ (ENNReal.ofReal (κ.rnDerivAux (κ + η) p.1 p.2)) / ...)", 
  "formal_completion": "exact (measurable_rnDerivAux κ _).ennreal_ofReal.div (measurable_const.sub (measurable_rnDerivAux κ _)).ennreal_ofReal",
  "is_geometry": false
}
```
