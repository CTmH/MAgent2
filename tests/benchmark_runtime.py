"""Reproducible runtime benchmark; run from the project root with PYTHONPATH=.

Example: PYTHONPATH=. python tests/benchmark_runtime.py --sizes 45 120 200 400
Use --library for a native control build and --baseline-python for a saved
pre-optimization magent_env.py. Each invocation uses one native library.
"""

import argparse
import ctypes
import gc
import importlib
import importlib.util
import json
import statistics
import time
import types

import numpy as np

from magent2 import gridworld


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library")
    parser.add_argument("--baseline-python")
    parser.add_argument("--label", default="current")
    parser.add_argument("--environment", default="battle_v4")
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--features", choices=("none", "minimap", "extra", "both"), default="none")
    parser.add_argument("--actions", choices=("zero", "random"), default="zero")
    parser.add_argument("--modes", nargs="+", choices=("step", "state", "step_state"), default=["step", "state", "step_state"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1 or args.warmup < 0:
        parser.error("repeats and steps must be positive; warmup must be nonnegative")
    if args.library:
        gridworld._LIB = ctypes.CDLL(args.library)
    old = None
    if args.baseline_python:
        spec = importlib.util.spec_from_file_location("baseline_env", args.baseline_python)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
    maker = importlib.import_module("magent2.environments." + args.environment).parallel_env
    for size in args.sizes or [None]:
        kwargs = {} if size is None else {"map_size": size}
        env = maker(
            **kwargs, num_threads=args.threads, max_cycles=args.steps + args.warmup + 10,
            minimap_mode=args.features in ("minimap", "both"),
            extra_features=args.features in ("extra", "both"),
        )
        if old:
            for name in ("step", "state", "_compute_observations", "_compute_rewards", "_compute_terminates"):
                setattr(env, name, types.MethodType(getattr(old.magent_parallel_env, name), env))
        # Generate the same action sequence for all builds outside timed regions.
        rng = np.random.default_rng(123)
        plans = [
            {a: int(rng.integers(env.action_space(a).n)) if args.actions == "random" else 0
             for a in env.possible_agents}
            for _ in range(args.steps + args.warmup)
        ]
        for mode in args.modes:
            wall_samples, cpu_samples, populations = [], [], []

            def run_step(i):
                if mode != "state":
                    if not env.agents:
                        raise RuntimeError("episode ended during benchmark; reduce --steps")
                    env.step(plans[i])
                if mode != "step":
                    env.state()

            for _ in range(args.repeats):
                env.reset(seed=123)
                for i in range(args.warmup):
                    run_step(i)
                gc.collect()
                cpu_start = time.process_time()
                wall_start = time.perf_counter()
                for i in range(args.warmup, args.warmup + args.steps):
                    run_step(i)
                wall_samples.append((time.perf_counter() - wall_start) * 1000 / args.steps)
                cpu_samples.append((time.process_time() - cpu_start) * 1000 / args.steps)
                populations.append(len(env.agents))
            print(json.dumps({
                "label": args.label, "environment": args.environment,
                "map_size": env.map_size, "initial_agents": len(env.possible_agents),
                "final_agents": populations, "threads": args.threads,
                "features": args.features, "actions": args.actions, "mode": mode,
                "steps": args.steps, "repeats": args.repeats,
                "wall_ms": statistics.median(wall_samples),
                "cpu_ms": statistics.median(cpu_samples), "samples_ms": wall_samples,
                "openmp": bool(getattr(gridworld._LIB, "env_openmp_enabled", lambda: 0)()),
            }), flush=True)
        env.close()
        del env
        gc.collect()


if __name__ == "__main__":
    main()
