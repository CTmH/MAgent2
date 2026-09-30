"""Run with PYTHONPATH=. python tests/test_complementary_pursuit.py.

Optionally set MAGENT_TEST_LIBRARY to a freshly built native library.
"""

import ctypes
import os
import pickle

import numpy as np

from magent2 import gridworld as gw
from magent2.environments import complementary_pursuit_v1 as cp
from magent2.environments import tiger_deer_v4


OFFSETS = [
    (x, y) for y in range(-2, 3) for x in range(-2, 3) if 0 < abs(x) + abs(y) <= 2
]


class Scene(cp.parallel_env):
    def __init__(self, modes="AB", positions=None, targets=None, **kwargs):
        self.modes = modes
        self.positions = positions or [(7, 5), (5, 7), (9, 7), (7, 9)][: len(modes)]
        self.targets = targets or [(7, 7), (12, 12)]
        super().__init__(map_size=15, max_cycles=2000, **kwargs)
        self.reset(seed=42)

    def generate_map(self):
        evaders, pursuers = self.env.get_handles()
        self.env.add_agents(evaders, "custom", pos=[(*p, 3) for p in self.targets])
        self.env.add_agents(
            pursuers,
            "custom",
            pos=[(*p, 3) for p in self.positions],
        )

    def attacks(self, target=(7, 7)):
        return {
            f"pursuer_{i}": 5
            + 12 * (mode == "B")
            + OFFSETS.index((target[0] - pos[0], target[1] - pos[1]))
            for i, (mode, pos) in enumerate(zip(self.modes, self.positions))
        }


def hp(env, group):
    center, _ = env.env.get_state_observation(env.handles[group])
    return center.copy() * (5 if group == 0 else 10)


def test_capture_and_credit():
    cases = [
        ("A", [0.02]),
        ("AA", [0.01, 0.01]),
        ("AB", [0.7, 0.7]),
        ("AAB", [0.4, 0.4, 0.6]),
        ("AAAB", [0.3, 0.3, 0.3, 0.5]),
        ("AABB", [0.3] * 4),
    ]
    for threads in (1, 4):
        for modes, expected in cases:
            e = Scene(modes, num_threads=threads)
            for _ in range(30):
                e.step({})
            before = hp(e, 1)
            _, rewards, terms, _, _ = e.step(e.attacks())
            np.testing.assert_allclose(
                [rewards[f"pursuer_{i}"] for i in range(len(modes))],
                expected,
                atol=1e-6,
            )
            effective = "A" in modes and "B" in modes
            assert terms["evader_0"] == effective
            np.testing.assert_allclose(rewards["evader_0"], -1.1 if effective else -0.1)
            recovery = 8 / len(modes) if effective else 0
            np.testing.assert_allclose(
                hp(e, 1), np.minimum(10, before + recovery) - 0.1, atol=2e-5
            )
            e.close()

    for modes in ("A", "AA"):
        e = Scene(modes, ineffective_damage=6)
        for _ in range(30):
            e.step({})
        _, rewards, terms, _, _ = e.step(e.attacks())
        assert terms["evader_0"]
        for i in range(len(modes)):
            np.testing.assert_allclose(rewards[f"pursuer_{i}"], 1.02 / len(modes))
        np.testing.assert_allclose(hp(e, 1), 9.9, atol=1e-5)
        e.close()

    e = Scene("A", ineffective_damage=0.3)
    for _ in range(60):
        _, rewards, terms, _, _ = e.step(e.attacks())
        if terms["evader_0"]:
            np.testing.assert_allclose(rewards["pursuer_0"], 1.02)
            break
    else:
        raise AssertionError("configured ineffective damage did not accumulate")
    e.close()


def test_nonlethal_and_misses():
    e = Scene("AAB", effective_damage=1)
    before = hp(e, 1)
    _, rewards, terms, _, _ = e.step(e.attacks())
    assert not terms["evader_0"]
    np.testing.assert_allclose(hp(e, 0)[0], 4.2, atol=1e-6)
    np.testing.assert_allclose(hp(e, 1), before - 0.1, atol=1e-5)
    np.testing.assert_allclose(
        [rewards[f"pursuer_{i}"] for i in range(3)], [0.2 / 3, 0.2 / 3, 0.2 / 3 + 0.2]
    )
    e.close()
    e = Scene("AB", effective_damage=5)
    assert e.step(e.attacks())[2]["evader_0"]  # Exactly zero HP is lethal.
    e.close()
    for positions, target in (
        ([(7, 5)], (8, 4)),
        ([(1, 1)], (1, -1)),
        ([(1, 1)], (1, 0)),
        ([(7, 5), (6, 5)], (6, 5)),
    ):
        e = Scene(
            "A" * len(positions), positions=positions, missed_attack_reward=-0.125
        )
        action = 5 + OFFSETS.index(
            (target[0] - positions[0][0], target[1] - positions[0][1])
        )
        rewards = e.step({"pursuer_0": action})[1]
        assert rewards["pursuer_0"] == -0.125
        e.close()
    # Different modes aimed at different evaders are two ineffective hits.
    e = Scene("AB", positions=[(7, 5), (11, 10)], targets=[(7, 7), (11, 12)])
    rewards = e.step(
        {
            "pursuer_0": 5 + OFFSETS.index((0, 2)),
            "pursuer_1": 17 + OFFSETS.index((0, 2)),
        }
    )[1]
    np.testing.assert_allclose([rewards["pursuer_0"], rewards["pursuer_1"]], 0.02)
    e.close()
    e = Scene("AA", pursuer_step_recover=0)
    for _ in range(1000):
        e.step(e.attacks())
    np.testing.assert_array_equal(hp(e, 0), [5, 5])
    e.close()
    e = Scene("A", ineffective_attack_reward_pool=0)
    assert e.step(e.attacks())[1]["pursuer_0"] == 0
    e.close()


def test_custom_rewards_and_simultaneous_targets():
    e = Scene(
        "AB",
        effective_attack_reward_pool=0.4,
        kill_reward_pool=2,
        critical_contribution_reward_pool=0.6,
        kill_hp_pool=0,
        pursuer_step_reward=0.1,
        pursuer_step_recover=-10,
        pursuer_dead_penalty=-0.2,
        evader_step_reward=0.2,
        evader_dead_penalty=-2,
        evader_attacked=-0.3,
    )
    # Capture credit must survive the subsequent starvation/death penalty.
    result = e.step(e.attacks())
    np.testing.assert_allclose([result[1]["pursuer_0"], result[1]["pursuer_1"]], 1.4)
    np.testing.assert_allclose(result[1]["evader_0"], -2.1)
    assert result[2]["pursuer_0"] and result[2]["pursuer_1"]
    e.close()
    e = Scene(
        "ABAB",
        positions=[(7, 5), (5, 7), (11, 10), (9, 12)],
        targets=[(7, 7), (11, 12)],
    )
    actions = {
        "pursuer_0": 5 + OFFSETS.index((0, 2)),
        "pursuer_1": 17 + OFFSETS.index((2, 0)),
        "pursuer_2": 5 + OFFSETS.index((0, 2)),
        "pursuer_3": 17 + OFFSETS.index((2, 0)),
    }
    rewards = e.step(actions)[1]
    np.testing.assert_allclose([rewards[f"pursuer_{i}"] for i in range(4)], 0.7)
    assert e.env.get_num(e.handles[0]) == 0
    e.close()


def test_shapes_and_encoding():
    for mini in (False, True):
        for extra in (False, True):
            new = cp.parallel_env(map_size=15, minimap_mode=mini, extra_features=extra)
            old = tiger_deer_v4.parallel_env(
                map_size=15, minimap_mode=mini, extra_features=extra
            )
            assert new.state_space.shape == old.state_space.shape
            for a, b in (("pursuer_0", "tiger_0"), ("evader_0", "deer_0")):
                assert new.observation_space(a).shape == old.observation_space(b).shape
            assert new.action_space("pursuer_0").n == 29
            assert new.action_space("evader_0").n == 5
            new.close()
            old.close()
            e = Scene("A", minimap_mode=mini, extra_features=extra)
            _, mask = e.env.get_view2attack(e.handles[1])
            assert np.count_nonzero(mask >= 0) == 12
            for i, (dx, dy) in enumerate(OFFSETS):
                assert mask[4 + dy, 4 + dx] == i
            for action in range(29):
                e.reset(seed=42)
                e.step({"pursuer_0": action})
                _, features = e.env.get_observation(e.handles[1])
                _, center_features = e.env.get_state_observation(e.handles[1])
                expected = np.zeros(9, dtype=np.float32)
                if action < 5:
                    expected[action] = 1
                else:
                    mode, offset = divmod(action - 5, 12)
                    expected[5 + mode] = 1
                    expected[7:] = (np.array(OFFSETS[offset]) + 2) / 4
                np.testing.assert_array_equal(features[0, 10:19], expected)
                np.testing.assert_array_equal(features, center_features)
                if extra:
                    x, y = e.env.get_pos(e.handles[1])[0]
                    np.testing.assert_array_equal(e.state()[x, y, 15:24], expected)
            e.close()


def test_validation_and_interfaces():
    for kwargs in (
        {"ineffective_damage": -1},
        {"kill_reward_pool": -1},
        {"effective_damage": float("nan")},
        {"pursuer_hp": 0},
        {"evader_step_recover": -0.1},
        {"map_size": 9},
        {"max_cycles": 0},
        {"num_threads": 0},
        {"kill_hp_pool": float("inf")},
        {"effective_damage": True},
    ):
        try:
            cp.parallel_env(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid configuration {kwargs}")
    e = cp.parallel_env(
        map_size=15,
        ineffective_attack_reward_pool=0,
        kill_reward_pool=2,
        num_threads=2,
        max_cycles=3,
    )
    clone = pickle.loads(pickle.dumps(e))
    assert clone.metadata["name"] == "complementary_pursuit_v1"
    for obj in (e, clone):
        obj.reset(seed=7)
        for invalid in (-1, 29, 1.5, True):
            try:
                obj.step({"pursuer_0": invalid})
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted invalid action {invalid}")
        for _ in range(3):
            result = obj.step({})
        assert all(result[3].values()) and not obj.agents
        assert obj.step({}) == ({}, {}, {}, {}, {})
        obj.close()
    e = Scene("AB", targets=[(7, 7)], minimap_mode=True, extra_features=True)
    result = e.step(e.attacks())
    assert all(result[2].values()) and not e.agents
    assert e.state().shape == (15, 15, 25)
    assert e.step({}) == ({}, {}, {}, {}, {})
    e.reset(seed=2)
    assert len(e.agents) == 3
    e.close()


def test_thread_equivalence():
    envs = [
        cp.parallel_env(
            map_size=100, num_threads=t, minimap_mode=True, extra_features=True
        )
        for t in (1, 4)
    ]
    for e in envs:
        e.reset(seed=17)
    rng = np.random.default_rng(17)
    for _ in range(30):
        actions = {
            a: int(rng.integers(envs[0].action_space(a).n)) for a in envs[0].agents
        }
        a, b = [e.step(actions) for e in envs]
        for x, y in zip(a, b):
            assert x.keys() == y.keys()
            for key in x:
                np.testing.assert_equal(x[key], y[key])
        np.testing.assert_array_equal(envs[0].state(), envs[1].state())
        if not envs[0].agents:
            break
    for e in envs:
        e.close()


if __name__ == "__main__":
    if os.environ.get("MAGENT_TEST_LIBRARY"):
        gw._LIB = ctypes.CDLL(os.environ["MAGENT_TEST_LIBRARY"])
    for test in (
        test_capture_and_credit,
        test_nonlethal_and_misses,
        test_custom_rewards_and_simultaneous_targets,
        test_shapes_and_encoding,
        test_validation_and_interfaces,
        test_thread_equivalence,
    ):
        test()
        print(test.__name__, "PASS")
