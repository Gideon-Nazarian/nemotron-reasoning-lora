"""Per-category programmatic solvers for the Nemotron Reasoning Challenge.

Each module exports a `solve(prompt: str) -> Solution` function that:
  1. Parses the example pairs from the prompt
  2. Infers the puzzle rule programmatically
  3. Applies it to the question
  4. Returns reasoning text + computed answer

The reasoning text is the training signal — it's the verbalized chain-of-thought
we want the fine-tuned model to imitate. Keep it CONCISE (short enough to fit in
~512 tokens of generation) so the LoRA-trained model produces outputs that fit
inside the comp grader's likely token budget.
"""
