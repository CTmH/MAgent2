"""Run directly with `python tests/test_threads.py`."""

import ctypes

from magent2 import gridworld
from magent2.environments import battle_v4


def test_num_threads():
    for invalid in (0, -1, True, 2**31):
        try:
            gridworld.GridWorld(gridworld.Config(), num_threads=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted num_threads={invalid}")

    default_env = battle_v4.parallel_env(map_size=12)
    default_env.reset()
    default_env.close()

    if not getattr(gridworld._LIB, "env_openmp_enabled", lambda: 0)():
        try:
            battle_v4.parallel_env(map_size=12, num_threads=2)
        except RuntimeError:
            return
        raise AssertionError("accepted num_threads without OpenMP")

    envs = [
        battle_v4.parallel_env(map_size=12, **kwargs)
        for kwargs in ({}, {"num_threads": 2})
    ]
    try:
        for env, expected in zip(envs, (1, 2)):
            actual = ctypes.c_int()
            gridworld._LIB.env_get_info(
                env.env.game, ctypes.c_int(0), b"num_threads", ctypes.byref(actual)
            )
            assert actual.value == expected
            env.reset()
            env.step({agent: env.action_space(agent).sample() for agent in env.agents})
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    test_num_threads()
