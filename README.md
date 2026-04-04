# mjlab playground

A collection of tasks built with [mjlab](https://github.com/mujocolab/mjlab), starting with ports from [MuJoCo Playground](https://playground.mujoco.org/).

## Tasks

| Task ID | Robot | Description |
|---------|-------|-------------|
| `Mjlab-Getup-Flat-Unitree-Go1` | Unitree Go1 | Fall recovery on flat terrain |
| `Mjlab-Getup-Flat-Booster-T1` | Booster T1 | Fall recovery on flat terrain |

## Setup

```bash
uv sync
```

## Training

```bash
uv run train --task Mjlab-Getup-Flat-Unitree-Go1 --num_envs 4096
```
