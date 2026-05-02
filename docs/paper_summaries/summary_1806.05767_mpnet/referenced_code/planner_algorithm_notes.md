# MPNet Planner Algorithm Notes

Core online loop:

```text
Z = Enet(obstacles)
path_a = [start]
path_b = [goal]

repeat up to N steps:
    x_new = Pnet(Z, path_a[-1], path_b[-1])
    append x_new to path_a

    if steerTo(path_a[-1], path_b[-1]) is collision-free:
        return lazy_state_contraction(path_a + reverse(path_b))

    swap(path_a, path_b)

if path failed:
    replan failed segments neurally or with RRT*
```

Important implementation points for this project:

- `steerTo` must become differential-drive aware or be followed by a tracker feasibility check.
- `lazy_state_contraction` is useful for smoothing and reducing unnecessary waypoints.
- Keep a classical fallback planner so the neural model is an accelerator, not the sole safety mechanism.
