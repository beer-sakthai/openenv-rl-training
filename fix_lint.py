with open("openenv-multi-catalog-training/a2a_agent/agent_executor.py", "r") as f:
    content = f.read()

import re
content = re.sub(r'self\.extra = \{\}  # per-env scratch \(e\.g\. echo\'s secret phrase\)', 'self.extra: dict = {}  # per-env scratch (e.g. echo\'s secret phrase)', content)

with open("openenv-multi-catalog-training/a2a_agent/agent_executor.py", "w") as f:
    f.write(content)
