"""Run with `python tests/test_medium_optimizations.py`."""

import ctypes
import importlib
from types import SimpleNamespace

import numpy as np

from magent2 import gridworld as gw
from magent2.environments import tiger_deer_v4
from magent2.environments.magent_env import magent_parallel_env


def test_sparse_results():
    # IDs have holes; d has already disappeared from the engine but still needs
    # its final result. Return ordering follows the current agent list.
    env = object.__new__(magent_parallel_env)
    env.possible_agents = list("abcd")
    env.agents = list("acd")
    env._agent_indices = {name: i for i, name in enumerate(env.possible_agents)}
    env._zero_obs = {name: np.zeros((1, 1, 1)) for name in env.possible_agents}
    env.handles = [0]
    env.team_sizes = [2]
    env.minimap_mode = env.extra_features = False
    env.env = SimpleNamespace(
        get_observation=lambda _: (np.array([[[[7]]], [[[8]]]]), np.empty((2, 0))),
        get_reward=lambda _: np.array([2.0, 3.0]),
        get_alive=lambda _: np.array([True, False]),
    )
    ids = [np.array([0, 2])]
    observations = env._compute_observations(ids)
    assert list(observations) == list("acd")
    assert [observations[a].item() for a in env.agents] == [7, 8, 0]
    assert env._compute_rewards(ids) == {"a": 2.0, "c": 3.0, "d": 0.0}
    assert env._compute_terminates(False, ids) == {"a": False, "c": True, "d": True}
    assert env.team_sizes == [1]
    assert all(env._compute_terminates(True, ids).values())
    env.agents = []
    assert env._compute_observations(ids) == env._compute_rewards(ids) == {}


def check_center(engine, handle):
    view, features = engine.get_observation(handle)
    expected_center = view[:, view.shape[1] // 2, view.shape[2] // 2, 2].copy()
    expected_features = features.copy()
    center, actual_features = engine.get_state_observation(handle)
    np.testing.assert_array_equal(center, expected_center)
    np.testing.assert_array_equal(actual_features, expected_features)
    assert center.dtype == actual_features.dtype == np.float32
    # A state read must not overwrite the existing full observation buffers.
    np.testing.assert_array_equal(features, expected_features)


def test_center_sampling():
    for name in ("battle_v4", "battlefield_v5", "combined_arms_v6",
                 "adversarial_pursuit_v4", "gather_v5", "tiger_deer_v4"):
        maker = importlib.import_module("magent2.environments." + name).parallel_env
        for mini, extra in ((False, False), (False, True), (True, False), (True, True)):
            env = maker(minimap_mode=mini, extra_features=extra)
            env.reset(seed=42)
            for _ in range(3):
                for handle in env._all_handles:
                    if env.env.get_num(handle):
                        check_center(env.env, handle)
                state = env.state()
                saved = state.copy()
                env.step(dict.fromkeys(env.agents, 0))
                np.testing.assert_array_equal(state, saved)
            env.close()

    # Sector views, even agent sizes, food channels, all four directions and edges.
    for mini in (False, True):
        cfg = gw.Config()
        cfg.set({"map_width": 20, "map_height": 20, "food_mode": True,
                 "minimap_mode": mini, "embedding_size": 2})
        kind = cfg.register_agent_type("test", {
            "width": 2, "length": 2, "hp": 5, "speed": 1,
            "view_range": gw.SectorRange(4, 90), "attack_range": gw.CircleRange(1),
        })
        cfg.add_group(kind)
        engine = gw.GridWorld(cfg)
        engine.reset()
        handle = engine.get_handles()[0]
        engine.add_agents(handle, method="custom", pos=[
            (1, 1, 0), (16, 1, 1), (1, 16, 2), (16, 16, 3), (10, 10, 0),
        ])
        check_center(engine, handle)


def test_empty_group_state():
    env = tiger_deer_v4.parallel_env(tiger_step_recover=-100)
    env.reset(seed=7)
    agents = env.agents[:]
    result = env.step(dict.fromkeys(agents, 0))
    assert list(result[0]) == agents
    assert all(result[2].values())
    assert any(env.env.get_num(h) == 0 for h in env._all_handles)
    state = env.state()
    assert state.shape == env.state_space.shape and state.dtype == np.float32
    env.reset(seed=7)
    assert env.agents == agents
    env.close()


def test_getter_cutoff():
    if not getattr(gw._LIB, "env_openmp_enabled", lambda: 0)():
        return
    from magent2.environments.battle.battle import default_reward_args, get_config

    engine = gw.GridWorld(get_config(1000, False, 42, **default_reward_args))
    handle = engine.get_handles()[0]
    for count in (16, 65535, 65536):
        engine.reset()
        engine.add_agents(handle, method="random", n=count)
        results = []
        for threads in (1, 4):
            gw._LIB.env_config_game(
                engine.game, b"num_threads", ctypes.byref(ctypes.c_int(threads))
            )
            results.append([
                engine.get_agent_id(handle), engine.get_pos(handle),
                engine.get_alive(handle), engine.get_reward(handle),
            ])
        for serial, parallel in zip(*results):
            np.testing.assert_array_equal(serial, parallel)


if __name__ == "__main__":
    test_sparse_results()
    test_center_sampling()
    test_empty_group_state()
    test_getter_cutoff()
