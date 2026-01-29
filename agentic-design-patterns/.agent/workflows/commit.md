---
description: Commit changes to the repository with a clear message and push to remote.
---

1. Check current changes to see what needs to be committed.
// turbo
2. Stage all changed and untracked files.
   ```bash
   git add .
   ```
3. Create a commit with a descriptive message following conventional commits (e.g., `feat:`, `fix:`, `refactor:`, `docs:`).
   ```bash
   git commit -m "[type]: [describe changes]"
   ```
// turbo
4. Push the changes to the remote repository.
   ```bash
   git push
   ```
