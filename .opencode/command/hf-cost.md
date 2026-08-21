---
description: Estimate HF Jobs/Kaggle costs for training, eval, and the full 5-iteration learning loop. Track budget and optimize spend.
agent: general
---

Help the user estimate and track training costs. Read the cost-optimization skill for reference.

1. Determine what they want to run:
   - Single 1.5B training run: HF Jobs A10G-small (~$3-4)
   - Single 7B training run: HF Jobs A10G-large (~$20-30)
   - Eval run: HF Jobs CPU (~$0.15)
   - Full 5-iteration loop: ~$4.30 (using Kaggle free tier for training)

2. Cost breakdown by option:
   | Option | Cost | Time |
   |---|---|---|
   | Kaggle T4 (free) | $0 | ~4-5 hrs/run |
   | HF Jobs A10G-small | ~$1.00/hr | ~3-4 hrs |
   | HF Jobs A10G-large | ~$2.50/hr | ~8-12 hrs |
   | HF Jobs CPU | ~$0.15/hr | ~1 hr |

3. Optimization tips:
   - Cache models locally to avoid re-download
   - Use Kaggle T4 for training (free 30 hrs/week)
   - Use CPU for eval (cheaper)
   - Reduce epochs for hyperparameter tests
   - Enable gradient checkpointing (already on)

4. Track with a budget sheet showing:
   - Iteration number, date, model, hardware, time, cost
   - Running total

Return a cost estimate with specific recommendations for cheapest path.
