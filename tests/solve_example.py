import sys, os

sys.path.append(os.path.abspath(".."))

from raytracer.simulation import *


# os.environ['CPU_MAX_COMPUTE_UNITS'] = '1'
os.environ["PYOPENCL_NO_CACHE"] = "1"
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
os.environ["CUDA_CACHE_DISABLE"] = "1"


if __name__ == "__main__":

    parameters = {
        "np": int(1e7),
        "dim": 3,
        "box_x": 1,
        "box_y": 1,
        "box_z": 1,
        "tmax": 0.7,
        "src_x": 0.5,
        "src_y": 0.5,
        "src_z": 0.5,
        "src_r": 0.5,
        "rcp_x": 0.8,
        "rcp_y": 0.5,
        "rcp_r": 0.5,
        "rcp_z": 0.5,
        "alpha": 1.0,
        "beta": 0.0,
    }

    simu = Simulation(parameters)
    simu.solve()