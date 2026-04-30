---
name: rover-remote-exec
description: Sync local code changes to the Pi rover via git diff over SSH and run scripts there. Use whenever code needs to execute on the physical rover.
---

# Rover Remote Exec

Diff-sync local changes to `ssh rover` and execute.

- Remote path: `/home/yasen/traktor-paper`
- Remote venv: `/home/yasen/traktor-venv/bin/python`

## Sync after every local edit

```bash
git add -N .
git diff --binary | ssh rover 'set -e; cd /home/yasen/traktor-paper; git reset --hard; git clean -fd; git apply --index'
```

## Run a script

```bash
ssh rover 'set -e; cd /home/yasen/traktor-paper; /home/yasen/traktor-venv/bin/python script.py'
```

## Pull data back

```bash
scp rover:/home/yasen/traktor-paper/<path> ./local/
```

## End-of-session cleanup

```bash
ssh rover 'set -e; cd /home/yasen/traktor-paper; git reset --hard; git clean -fd'
```

> Never edit files directly on the Pi — always sync from local.
