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
