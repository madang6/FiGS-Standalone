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
If commits in `acados-away` touch the submodule or `.gitmodules`, just don't cherry-pick those commits (or split your work so "portable" changes are in separate commits).

---

## Submodule Workflow (e.g., gemsplat)

### Push changes in submodule, then update parent

```bash
# 1. Commit and push in the SUBMODULE
cd gemsplat
git add -A
git commit -m "Your commit message"
git push origin main   # or whatever branch

# 2. Update the PARENT to point to new submodule commit
cd ..   # back to FiGS-Standalone
git add gemsplat
git commit -m "Update gemsplat submodule"
git push origin main
```

### Pull submodule updates (on another machine)

```bash
# If you already cloned FiGS-Standalone
git pull origin main
git submodule update --init --recursive
```

### Clone with submodules from scratch

```bash
git clone --recurse-submodules <repo-url>
# OR if already cloned without submodules:
git submodule update --init --recursive
```

### Check submodule status

```bash
git submodule status
# Shows commit hash each submodule points to
# A leading '-' means not initialized
# A leading '+' means local is different from what parent expects
```

### Common gotcha
The parent repo tracks a **specific commit** of the submodule, not a branch. If you commit in the submodule but forget to `git add <submodule>` in the parent, the parent still points to the old commit.