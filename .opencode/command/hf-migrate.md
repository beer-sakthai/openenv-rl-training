---
description: Manage model/dataset versioning, deprecate old versions, migrate users from v1 to v2, update collections and cross-links.
agent: general
---

Help the user manage version migration across the SakThai family.

1. Read the migration skill for version map and conventions.
2. Determine the migration type:
   - **v1 → v2 model upgrade**: Add deprecation notice to v1 card, add `new_version` YAML, update collection
   - **Dataset versioning**: Mark v6 as superseded, point all cross-links to v7
   - **Repo rename**: Not possible on HF (create new, deprecate old)
   - **Breaking change**: Add migration guide to README

3. For a v1→v2 migration:
   - Update v1 README with deprecation banner
   - Add `new_version: Nanthasit/sakthai-plus-1.5b-lora` to v1 YAML
   - Update HF Collection: add v2 as current, tag v1 as "Previous"
   - Update all cross-references in other model cards

4. Generate the exact files/changes needed.

Return a step-by-step migration plan with exact file paths and commands.
