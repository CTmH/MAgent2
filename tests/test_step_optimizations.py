"""Run with `python tests/test_step_optimizations.py`."""

import numpy as np

from magent2.environments import battle_v4


def test_step_optimizations():
    for minimap in (False, True):
        for extra in (False, True):
            env = battle_v4.parallel_env(
                map_size=12, minimap_mode=minimap, extra_features=extra, max_cycles=2
            )
            observations, _ = env.reset(seed=7)
            for handle in env.handles:
                ids = env.env.get_agent_id(handle)
                view, features = env.env.get_observation(handle)
                if minimap and not extra:
                    features = features[:, -2:]
                expected = view
                if minimap or extra:
                    expected = np.concatenate(
                        [
                            view,
                            np.tile(features[:, None, None, :], (1, *view.shape[1:3], 1)),
                        ],
                        axis=-1,
                    )
                for agent_id, obs in zip(ids, expected):
                    actual = observations[env.possible_agents[agent_id]]
                    np.testing.assert_array_equal(actual, obs)
                    assert actual.dtype == obs.dtype

            snapshots = {agent: obs.copy() for agent, obs in observations.items()}
            calls = []
            get_ids = env.env.get_agent_id

            def counted_ids(handle):
                calls.append(handle.value)
                return get_ids(handle)

            env.env.get_agent_id = counted_ids
            for _ in range(2):
                calls.clear()
                result = env.step(dict.fromkeys(env.agents, 0))
                assert calls == [handle.value for handle in env.handles]
                for agent, obs in observations.items():
                    np.testing.assert_array_equal(obs, snapshots[agent])
            assert all(result[3].values())
            assert not env.agents
            env.reset(seed=7)
            assert env.agents
            env.close()


if __name__ == "__main__":
    test_step_optimizations()
