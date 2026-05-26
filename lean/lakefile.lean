import Lake
open Lake DSL

package «lean_eval» where
  -- Evaluation-only Lake project for the Python harness.

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.11.0"

@[default_target]
lean_lib «LeanEval» where
  roots := #[`LeanEval]
