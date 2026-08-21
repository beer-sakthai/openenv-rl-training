---
name: cycle-workflow
description: Master operation flow for every task. Follows SakThai 6-stage energy cycle: Dream → Hope → Care → Joy → Trust → Growth. Every action passes through all 6 stages. Never skip a stage.
---

# SakThai Cycle Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    SAKTHAI OPERATION CYCLE                   │
│                                                              │
│   🎯 DREAM → 🏗️ HOPE → ❤️ CARE → 🎉 JOY → 🔐 TRUST → 🌱 GROWTH │
│                                                              │
│   Every task. Every fix. Every launch. No skips.             │
└─────────────────────────────────────────────────────────────┘
```

## The 6 Stages

### Stage 1: 🎯 Dream — "What needs to happen?"
**Before any action.** Check context, memory, history.

```
- What is the user asking?
- What have we done before on this? (check memory)
- What's the goal state?
- What files/apis are involved?
```

**Output**: Clear one-sentence objective. No action yet.
**Time**: ~30 seconds

### Stage 2: 🏗️ Hope — "How will I do it?"
**Plan before touching.** Check branches, files, tools available.

```
- Check current state (git status, HF Hub, file system)
- List all files that need to change
- Determine order of operations
- Identify risks (breaking changes, token access, GPU needed)
- Write down the plan
```

**Output**: Ordered plan with files to change and commands to run.
**Rule**: Don't drive without a map.

### Stage 3: ❤️ Care — "Do the work."
**Execute. Carefully.** Every detail matters.

```
- Make changes one at a time
- Verify each change before moving to next
- Check imports, syntax, paths
- No sloppy work — sloppy work breaks trust
```

**Output**: Completed changes.
**Rule**: This is where I earn my name. Care = trust.

### Stage 4: 🎉 Joy — "Ship it."
**Commit, push, deploy.**

```
- Commit with descriptive message
- Push to GitHub
- Upload to HF Hub
- Schedule cron job if needed
```

**Output**: Live changes.
**Rule**: Don't celebrate until the next stage passes.

### Stage 5: 🔐 Trust — "Did it actually work?"
**Verify. Every time.**

```
- Check push landed (git log, HF Hub status)
- Check CI/cron ran
- If I changed 3 files, check all 3
- If I fixed 2 repos, check both
- Run a quick test if possible
```

**Output**: Verified status — all changes working.
**Rule**: Trust is earned by verification, not intention.

### Stage 6: 🌱 Growth — "What did I learn?"
**Save lessons. Get faster.**

```
- What went well?
- What could go faster next time?
- Update skills/AGENTS.md with new knowledge
- Patch any skill that had gaps
- Save timing estimates for future planning
```

**Output**: Updated knowledge. Faster next iteration.
**Rule**: Next time I face the same problem, I'm faster.

---

## Quick Reference Card

```
DREAM  🎯  "What?"      → Context check, 30s
HOPE   🏗️  "How?"       → Plan, verify state
CARE   ❤️  "Do it."     → Execute carefully
JOY    🎉  "Ship."      → Commit, push, deploy
TRUST  🔐  "Verify."    → Check it worked
GROWTH 🌱  "Learn."     → Save lessons, patch skills
```

## Applying to Common Tasks

### Training a model
```
DREAM  → Train sakthai-plus-1.5b on H100
HOPE   → Script ready? Token valid? Repos exist? GPU available?
CARE   → Run hf jobs uv run with correct params
JOY    → Push adapter and merged model to HF Hub
TRUST  → Verify model card, check downloads, run eval
GROWTH → Save training time, update cost estimates
```

### Fixing a bug
```
DREAM  → What error? When did it start? What changed?
HOPE   → Check logs, find root cause, plan fix
CARE   → Fix the code, test locally
JOY    → Commit and push
TRUST  → Verify fix in production
GROWTH → Add test case, update troubleshooting guide
```

### Creating a dataset
```
DREAM  → What gap? What format? What tools?
HOPE   → Check existing data, design schema, plan generation
CARE   → Generate examples, validate format, fix edge cases
JOY    → Push to HF Hub, update dataset card
TRUST  → Verify load_dataset works, check row counts
GROWTH → Document generation method, save script
```
