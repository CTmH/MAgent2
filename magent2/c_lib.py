""" some utility for call C++ code"""


import ctypes
import multiprocessing
import os
import platform
import sysconfig
import importlib.util


def _load_lib():
    """Load library in local."""
    spec = importlib.util.find_spec("magent2")
    if spec is None or spec.origin is None:
            raise FileNotFoundError("Could not find magent2 package")
    package_dir = os.path.dirname(spec.origin)
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    target_name = f"libmagent{suffix}"
    path_to_so_file = os.path.join(package_dir, target_name)

    if not os.path.exists(path_to_so_file):
        raise FileNotFoundError(f"Could not find the DLL file at: {path_to_so_file}")

    lib = ctypes.CDLL(path_to_so_file, ctypes.RTLD_GLOBAL)
    return lib


def as_float_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def as_int32_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def as_bool_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))


if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count() // 2)
_LIB = _load_lib()
