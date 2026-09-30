<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/MAgent2/main/MAgent2-text.png" width="500px"/>
</p>

MAgent2 is a library for the creation of environments where large numbers of pixel agents in a gridworld interact in battles or other competitive scenarios.

<p align="center">
  <img src="https://raw.githubusercontent.com/Farama-Foundation/MAgent2/main/docs/environments/adversarial_pursuit.gif" width="200">
</p>

MAgent2 is a maintained fork of the original [MAgent](https://github.com/geek-ai/MAgent) codebase. It contains some [reference environments](https://github.com/Farama-Foundation/MAgent2/tree/main/magent2/environments) implemented using the [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) API. These environments used to be included in PettingZoo itself, but have been moved here to exist independently. They are being regularly maintained and will receive bug fixes, support new versions of Python, etc. Development used to take place at [github.com/Farama-Foundation/MAgent](https://github.com/Farama-Foundation/MAgent) but was moved to [github.com/Farama-Foundation/MAgent2](https://github.com/Farama-Foundation/MAgent2) so that the distinction from the original MAgent library is clear to users.

## Installation
Install using pip: `pip install magent2`. See [docs](https://magent2.farama.org/) for usage information.

Each environment uses one OpenMP thread by default. To use more, pass `num_threads` when creating it:

```python
from magent2.environments import battle_v4

env = battle_v4.parallel_env(num_threads=2)
```

This limits each OpenMP parallel region in that environment, not the process's CPU affinity or threads used by other libraries. The native library must be built with OpenMP support to use more than one thread.


## Complementary Pursuit

```python
from magent2.environments import complementary_pursuit_v1

env = complementary_pursuit_v1.parallel_env(
    num_threads=1,
    effective_damage=10.0,
    ineffective_damage=0.2,
    effective_attack_reward_pool=0.2,
    ineffective_attack_reward_pool=0.02,  # Set to zero to disable hit shaping.
    critical_contribution_reward_pool=0.2,
    kill_reward_pool=1.0,
    kill_hp_pool=8.0,
)
observations, infos = env.reset(seed=42)
```

Agents are named `evader_*` and `pursuer_*`. Evaders have five movement
choices; pursuers have 29 choices: movement `0..4` (stay is `2`), A attacks
`5..16`, and B attacks `17..28`. Each attack selects one target offset within
fixed Manhattan distance two. Offsets are ordered by local `dy`, then `dx`,
excluding `(0, 0)`. The first A/B attack targets `(0, -2)`. Intermediate walls
do not obstruct attacks, matching the existing engine's targeting rules.

Attacks are grouped by target before damage is applied. A target receiving
both A and B takes effective damage; any other nonempty set of hits inflicts
ineffective damage. Damage is applied **once per target per step**, regardless
of the number of attackers. Invalid targets receive no damage. At zero HP,
an evader is captured and removed before recovery or movement.

The corresponding hit reward pool is split among the current attackers.
An effective attack also splits the critical-contribution pool among attackers
whose removal would make it ineffective: both attackers in AB, only B in AAB,
and nobody in AABB. Any capture, including an ineffective one, additionally
splits `kill_reward_pool` and `kill_hp_pool` among the current attackers. HP is
capped individually; unused shares are not redistributed. Nonlethal hits
provide no HP. Under default settings, AB earns 0.7 per pursuer, AAB earns
0.4/0.4/0.6, and a lone ineffective hit earns 0.02. Repeated ineffective hits
on a healthy evader are offset by its default recovery. Raising ineffective
damage or lowering recovery allows ineffective captures, with the same
capture reward and HP rules.

Other configurable defaults are `pursuer_hp=10`, `evader_hp=5`,
`pursuer_step_recover=-0.1`, `evader_step_recover=0.2`,
`missed_attack_reward=0`, `pursuer_step_reward=0`, `pursuer_dead_penalty=0`,
`evader_step_reward=0`, `evader_dead_penalty=-1`, and `evader_attacked=-0.1`.
The evader hit penalty applies once per target per step, including capture.
Death penalties are additive to rewards already earned during the step.
The usual `map_size`, `max_cycles`, `minimap_mode`, `extra_features`, `seed`,
`render_mode`, and `num_threads` options are also supported. Reward pools,
damage, and evader recovery must be nonnegative; HP must be positive.

Observation and state shapes match Tiger-Deer under the same map and feature
settings. With `extra_features=True`, the pursuer's nine action-history slots
encode five movement one-hot values, two attack-mode one-hot values, and
`(dx + 2) / 4`, `(dy + 2) / 4`. Movement zeros the attack slots; attacks zero
the movement slots. These feature meanings and the action space differ from
Tiger-Deer, so existing policies are not directly compatible. The spatial
views retain their original sizes: 9x9 for pursuers and 3x3 for evaders.

Rebuild the native library after updating the sources. Regression checks:

```sh
PYTHONPATH=. python tests/test_complementary_pursuit.py
```

## Requirements
MAgent2 supports Linux and macOS and Python 3.10+.

### Building from source

Source builds require a C++ compiler with OpenMP support and its OpenMP runtime
(for example, GCC with libgomp on Linux or macOS). These are system dependencies;
`pyproject.toml` installs the Python build tools, including CMake 4.0+, but does not
install the compiler or OpenMP runtime. Configuration fails if OpenMP is missing.

With a suitable compiler installed, build from the project directory:

```sh
python -m pip install .
```

On macOS, the default Apple Clang compiler may need additional OpenMP setup.
To use MacPorts GCC 15 instead, select it explicitly:

```sh
CC=/opt/local/bin/gcc-mp-15 CXX=/opt/local/bin/g++-mp-15 python -m pip install .
```

Use a clean source tree or remove the previous CMake build cache when switching
compilers. OpenMP support is required at build time; the default runtime setting
remains `num_threads=1`.

## References
```
@inproceedings{zheng2018magent,
  title={MAgent: A many-agent reinforcement learning platform for artificial collective intelligence},
  author={Zheng, Lianmin and Yang, Jiacheng and Cai, Han and Zhou, Ming and Zhang, Weinan and Wang, Jun and Yu, Yong},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}
```

If you wish to cite this repo with it's modifications specifically, please cite:

```
@misc{magent2020,
  author = {Terry, Jordan K and Black, Benjamin and Jayakumar, Mario},
  title = {MAgent},
  year = {2020},
  publisher = {GitHub},
  note = {GitHub repository},
  howpublished = {\url{https://github.com/Farama-Foundation/MAgent}}
}
```
