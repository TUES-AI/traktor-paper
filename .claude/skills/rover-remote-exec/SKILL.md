---
name: rover-remote-exec
description: Sync local code changes to the Pi rover via git diff over SSH and run scripts there. Use whenever code needs to execute on the physical rover.
---

# Rover Remote Exec

Diff-sync local changes to `ssh rover` and execute.

- Remote path: `/home/yasen/traktor-paper`

## FIRST bring the raspberry to the latest commit!

The raspberry could be some commits behind and that will result in problems when doing git diff syncing.
To prevent this check if there are files on the raspberry that need to be `scp`-ed over, then reset the state and pull.

## Sync after every local edit

```bash
git add -N .
git diff --binary | ssh rover 'set -e; cd /home/yasen/traktor-paper; git reset --hard; git clean -fd; git apply --index'
```

## Run a script

```bash
ssh rover 'set -e; cd /home/yasen/traktor-paper; python3 script.py'
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

## Don't delete data on the remote

If a script has generated some data in the remote filesystem that is git tracked, make sure you copy it outside of the repo, gitignore it or git commit it - decide based on the setup. Don't run desctructive git reset.
