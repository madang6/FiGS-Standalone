Use **cherry-pick**: keep the branches intentionally divergent, and copy only the commits you want from `acados-away` onto `main`.

### Workflow
1) Make commits on `acados-away` as usual.
2) When you want to port some of them to `main`:

```bash
git fetch origin
git switch main

# pick specific commits (oldest -> newest)
git cherry-pick <hash1> <hash2> <hash3>

git push origin main
```

### Getting the commit hashes
```bash
git log --oneline origin/acados-away
```

### Cherry-pick a contiguous range
If it’s “everything from A (exclusive) up to B (inclusive)”:

```bash
git cherry-pick A..B
```

### If you hit conflicts
Resolve in VS Code, then:
```bash
git add <files>
git cherry-pick --continue
```
Abort if needed:
```bash
git cherry-pick --abort
```

### Tip to avoid accidentally picking submodule-related commits
If commits in `acados-away` touch the submodule or `.gitmodules`, just don’t cherry-pick those commits (or split your work so “portable” changes are in separate commits).