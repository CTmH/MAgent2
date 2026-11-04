"""Run with PYTHONPATH=. python tests/test_entity_features.py."""

import ctypes
import importlib
import os

import numpy as np

from magent2 import gridworld as gw


ENVIRONMENTS = (
    "adversarial_pursuit_v4",
    "battle_v4",
    "battlefield_v5",
    "combined_arms_v6",
    "gather_v5",
    "tiger_deer_v4",
    "complementary_pursuit_v1",
)


def check_entities(env):
    width = max(env.env.get_feature_space(h)[0] for h in env._all_handles)
    tail = 3 if env.minimap_mode else 1
    entities = {}
    anchors = {}
    for handle in env._all_handles:
        _, features = env.env.get_state_observation(handle)
        if env.metadata["name"].startswith("gather"):
            decoded = features[:, :32].astype(np.int64) @ (
                1 << np.arange(32, dtype=np.int64)
            )
            np.testing.assert_array_equal(decoded, env.env.get_agent_id(handle))
        for position, feature in zip(env.env.get_pos(handle), features):
            padded = np.zeros(width, dtype=np.float32)
            padded[: len(feature) - tail] = feature[:-tail]
            padded[-tail:] = feature[-tail:]
            anchors[tuple(position)] = padded
            body = (
                2
                if env.metadata["name"].startswith("adversarial_pursuit")
                and handle.value == 0
                else 1
            )
            for dx in range(body):
                for dy in range(body):
                    entities[tuple(position + (dx, dy))] = padded
    for handle in env.handles:
        views, features = env.env.get_entity_observation(handle)
        positions = env.env.get_pos(handle)
        h, w = views.shape[1:3]
        stride = 3 if env.minimap_mode else 2
        for i in range(min(3, len(positions))):
            x, y = positions[i]
            if (
                env.metadata["name"].startswith("adversarial_pursuit")
                and handle.value == 0
            ):
                x += 1
                y += 1
            for row in range(h):
                for col in range(w):
                    occupied = views[i, row, col, 1::stride].any()
                    expected = (
                        entities.get((x + col - w // 2, y + row - h // 2))
                        if occupied
                        else None
                    )
                    np.testing.assert_array_equal(
                        features[i, row, col],
                        np.zeros(width) if expected is None else expected,
                    )
    state = env.state()
    start = 1 + 2 * len(env._all_handles)
    for handle in env._all_handles:
        for x, y in env.env.get_pos(handle):
            assert state[x, y, 1 + 2 * handle.value] == 1
    for (x, y), feature in anchors.items():
        np.testing.assert_array_equal(
            state[x, y, start:], feature[:-2] if env.minimap_mode else feature
        )


def test_all_environments():
    for name in ENVIRONMENTS:
        for mini in (False, True):
            maker = importlib.import_module("magent2.environments." + name).parallel_env
            env = maker(extra_features=True, minimap_mode=mini)
            observations, _ = env.reset(seed=9)
            for observation in observations.values():
                assert not observation[..., -3 if mini else -1].any()
            check_entities(env)
            for _ in range(2):
                observations, rewards, _, _, _ = env.step(dict.fromkeys(env.agents, 2))
                check_entities(env)
                state = env.state()
                for handle in env.handles:
                    for agent_id, (x, y) in zip(
                        env.env.get_agent_id(handle), env.env.get_pos(handle)
                    ):
                        agent = env.possible_agents[agent_id]
                        obs = observations[agent]
                        reward_channel = -3 if mini else -1
                        np.testing.assert_allclose(
                            obs[obs.shape[0] // 2, obs.shape[1] // 2, reward_channel],
                            rewards[agent],
                            atol=1e-6,
                        )
                        np.testing.assert_allclose(
                            state[x, y, -1], rewards[agent], atol=1e-6
                        )
            env.close()
        print(name, "PASS")


def test_group_reward_snapshot():
    cfg = gw.Config()
    cfg.set(dict(map_width=12, map_height=12, embedding_size=10))
    attacker = cfg.register_agent_type(
        "attacker",
        dict(
            hp=5,
            speed=1,
            damage=1,
            step_reward=0.3,
            view_range=gw.CircleRange(1),
            attack_range=gw.CircleRange(1),
        ),
    )
    target = cfg.register_agent_type(
        "target",
        dict(
            hp=5, speed=1, view_range=gw.CircleRange(1), attack_range=gw.CircleRange(0)
        ),
    )
    a, b = cfg.add_group(attacker), cfg.add_group(target)
    cfg.add_reward_rule(
        gw.Event(gw.AgentSymbol(a, "any"), "attack", gw.AgentSymbol(b, "any")),
        receiver=gw.AgentSymbol(a, "all"),
        value=0.7,
    )
    engine = gw.GridWorld(cfg)
    engine.reset()
    a, b = engine.get_handles()
    engine.add_agents(a, "custom", pos=[(5, 6)])
    engine.add_agents(b, "custom", pos=[(5, 5)])
    engine.set_action(a, np.array([5], dtype=np.int32))
    engine.step()
    reward = engine.get_reward(a)[0]
    np.testing.assert_allclose(reward, 1)
    np.testing.assert_allclose(engine.get_entity_observation(a)[1][0, 1, 1, -1], reward)
    engine.clear_dead()
    np.testing.assert_allclose(engine.get_state_observation(a)[1][0, -1], reward)


if __name__ == "__main__":
    if os.environ.get("MAGENT_TEST_LIBRARY"):
        gw._LIB = ctypes.CDLL(os.environ["MAGENT_TEST_LIBRARY"])
    test_all_environments()
    test_group_reward_snapshot()
