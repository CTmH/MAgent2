"""Complementary Pursuit: capture evaders with simultaneous A/B attacks.

Attacks target one of the 12 cells at Manhattan distance 1 or 2. Pursuers
have Discrete(29) actions: movement 0..4, mode A 5..16, mode B 17..28.
Targets are ordered by local dy, then dx, excluding the origin. Movement 2
is stay. Offsets use the same local coordinate frame as observations.
Walls block movement, but intermediate walls do not block attacks.

Each evader receives one damage event per step. Both modes present means
an effective attack; otherwise the hit is ineffective. Damage is resolved
before recovery and movement. Zero HP is lethal in this environment.
Rewards and capture HP are pools shared by that target's current attackers.
A critical attacker is one whose removal would make the attack ineffective.
Capture rewards and HP apply to both effective and ineffective captures.
A full-HP participant's unused share of HP is not redistributed.

Observations and state retain Tiger-Deer's shapes for every feature mode.
Pursuer action history uses five movement one-hot slots, two attack-mode
one-hot slots, and (dx + 2) / 4, (dy + 2) / 4. Non-applicable slots are zero.
Rewards in extra features retain the existing previous-step convention.
"""

import math
from numbers import Real

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_to_aec_wrapper

from magent2 import gridworld as gw
from magent2.environments.magent_env import magent_parallel_env, make_env


def _get_config(map_size, minimap_mode, seed, settings):
    cfg = gw.Config()
    cfg.set(
        dict(
            map_width=map_size,
            map_height=map_size,
            minimap_mode=minimap_mode,
            embedding_size=10,
        )
    )
    if seed is not None:
        cfg.set({"seed": seed})
    evader = cfg.register_agent_type(
        "evader",
        dict(
            width=1,
            length=1,
            hp=settings["evader_hp"],
            speed=1,
            view_range=gw.CircleRange(1),
            attack_range=gw.CircleRange(0),
            step_recover=settings["evader_step_recover"],
            step_reward=settings["evader_step_reward"],
            dead_penalty=settings["evader_dead_penalty"],
            attacked_penalty=settings["evader_attacked"],
            kill_reward=settings["kill_reward_pool"],
            kill_supply=settings["kill_hp_pool"],
        ),
    )
    pursuer = cfg.register_agent_type(
        "pursuer",
        dict(
            width=1,
            length=1,
            hp=settings["pursuer_hp"],
            speed=1,
            view_range=gw.CircleRange(4),
            complementary_attack=True,
            damage=settings["effective_damage"],
            ineffective_damage=settings["ineffective_damage"],
            effective_attack_reward_pool=settings["effective_attack_reward_pool"],
            ineffective_attack_reward_pool=settings["ineffective_attack_reward_pool"],
            critical_contribution_reward_pool=settings[
                "critical_contribution_reward_pool"
            ],
            attack_penalty=settings["missed_attack_reward"],
            step_recover=settings["pursuer_step_recover"],
            step_reward=settings["pursuer_step_reward"],
            dead_penalty=settings["pursuer_dead_penalty"],
        ),
    )
    cfg.add_group(evader)
    cfg.add_group(pursuer)
    return cfg


class parallel_env(magent_parallel_env, EzPickle):
    """Create the environment; pool values and damage must be nonnegative.

    All numeric settings must be finite. HP must be positive and evader
    recovery nonnegative. Negative pursuer recovery models starvation.
    Unspecified actions default to stay; out-of-range actions raise ValueError.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "complementary_pursuit_v1",
        "render_fps": 5,
    }

    def __init__(
        self,
        map_size=45,
        max_cycles=300,
        minimap_mode=False,
        extra_features=False,
        render_mode=None,
        seed=None,
        num_threads=1,
        *,
        effective_damage=10.0,
        ineffective_damage=0.2,
        effective_attack_reward_pool=0.2,
        ineffective_attack_reward_pool=0.02,
        critical_contribution_reward_pool=0.2,
        kill_reward_pool=1.0,
        kill_hp_pool=8.0,
        missed_attack_reward=0.0,
        pursuer_hp=10.0,
        evader_hp=5.0,
        pursuer_step_recover=-0.1,
        evader_step_recover=0.2,
        pursuer_step_reward=0.0,
        pursuer_dead_penalty=0.0,
        evader_step_reward=0.0,
        evader_dead_penalty=-1.0,
        evader_attacked=-0.1,
    ):
        settings = dict(
            effective_damage=effective_damage,
            ineffective_damage=ineffective_damage,
            effective_attack_reward_pool=effective_attack_reward_pool,
            ineffective_attack_reward_pool=ineffective_attack_reward_pool,
            critical_contribution_reward_pool=critical_contribution_reward_pool,
            kill_reward_pool=kill_reward_pool,
            kill_hp_pool=kill_hp_pool,
            missed_attack_reward=missed_attack_reward,
            pursuer_hp=pursuer_hp,
            evader_hp=evader_hp,
            pursuer_step_recover=pursuer_step_recover,
            evader_step_recover=evader_step_recover,
            pursuer_step_reward=pursuer_step_reward,
            pursuer_dead_penalty=pursuer_dead_penalty,
            evader_step_reward=evader_step_reward,
            evader_dead_penalty=evader_dead_penalty,
            evader_attacked=evader_attacked,
        )
        for name, value in settings.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or abs(value) > np.finfo(np.float32).max
            ):
                raise ValueError(f"{name} must be a finite float32-compatible number")
            if (
                name.endswith("_pool")
                or name.endswith("_damage")
                or name == "evader_step_recover"
            ) and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            if name.endswith("_hp") and value < np.finfo(np.float32).tiny:
                raise ValueError(
                    f"{name} must be positive and representable as float32"
                )
        if isinstance(map_size, bool) or not isinstance(map_size, int) or map_size < 10:
            raise ValueError("map_size must be an integer >= 10")
        if (
            isinstance(max_cycles, bool)
            or not isinstance(max_cycles, int)
            or max_cycles < 1
        ):
            raise ValueError("max_cycles must be a positive integer")
        reward_values = [
            value
            for name, value in settings.items()
            if "reward" in name or "penalty" in name or name == "evader_attacked"
        ]
        reward_range = (
            sum(min(v, 0) for v in reward_values),
            sum(max(v, 0) for v in reward_values),
        )
        if max(abs(v) for v in reward_range) > np.finfo(np.float32).max:
            raise ValueError("combined rewards exceed the float32 range")
        if not getattr(gw._LIB, "env_complementary_pursuit_enabled", lambda: 0)():
            raise RuntimeError(
                "Rebuild MAgent2's native library to use Complementary Pursuit"
            )
        EzPickle.__init__(
            self,
            map_size,
            max_cycles,
            minimap_mode,
            extra_features,
            render_mode,
            seed,
            num_threads,
            **settings,
        )
        engine = gw.GridWorld(
            _get_config(map_size, minimap_mode, seed, settings), num_threads=num_threads
        )
        super().__init__(
            engine,
            engine.get_handles(),
            ["evader", "pursuer"],
            map_size,
            max_cycles,
            reward_range,
            minimap_mode,
            extra_features,
            render_mode,
        )

    def generate_map(self):
        evader, pursuer = self.env.get_handles()
        self.env.add_walls(method="random", n=self.map_size**2 * 0.04)
        self.env.add_agents(evader, method="random", n=self.map_size**2 * 0.05)
        self.env.add_agents(pursuer, method="random", n=self.map_size**2 * 0.01)

    def step(self, actions):
        # Validate before crossing the C boundary; missing actions mean stay.
        complete = {}
        for agent in self.agents:
            action = actions.get(agent, 2)
            if isinstance(action, (bool, np.bool_)) or not self.action_space(
                agent
            ).contains(action):
                raise ValueError(f"Invalid action for {agent}: {action!r}")
            complete[agent] = action
        return super().step(complete)


def raw_env(**kwargs):
    return parallel_to_aec_wrapper(parallel_env(**kwargs))


env = make_env(raw_env)
