#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import meshio
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.clrandom import PhiloxGenerator

# Check that all req_param entries are inside input_param.
def check_parameters(req_param, input_param):
    is_missing = False

    for key in req_param.keys():
        if key not in input_param.keys():
            print(
                "Error- required parameter {k} of type {t} is missing".format(
                    k=key, t=req_param[key]
                )
            )
            is_missing = True

    if is_missing:
        exit()


# Check if val is numeric (int or float).
def is_num_type(val):
    return isinstance(val, int) or isinstance(val, float)


# Return val only if its type is type_to_check, else throw and TypeError.
def safe_assign(key, val, type_to_check):

    if type_to_check == int:
        if is_num_type(val):
            return int(val)
        else:
            raise TypeError("{} must be numeric".format(key))

    if type_to_check == float:
        if is_num_type(val):
            return float(val)
        else:
            raise TypeError("{} must be numeric".format(key))

    if type_to_check == str:
        if isinstance(val, str):
            return val
        else:
            raise TypeError("{} must be str".format(key))


class Simulation:
    def __init__(self, parameters):

        self.__set_parameters(parameters)

        self.dtype = np.float32
        self.source = self.__process_ocl_src(print_src=False)
        self.xdmf_filename = "particle.xdmf"

    def __set_parameters(self, param):

        req_param = {
            "np": int,
            "dim": int,
            "box_x": float,
            "box_y": float,
            "tmax": float,
            "src_x": float,
            "src_y": float,
            "src_r": float,
            "rcp_x": float,
            "rcp_y": float,
            "rcp_r": float,
            "alpha": float,
            "beta": float,
        }

        if param["dim"] == 3:
            req_param.update({"box_z": float, "src_z": float, "rcp_z": float})

        self.src_file = "raytracer.cl"

        check_parameters(req_param, param)

        self.param = param
        self.np = safe_assign("np", self.param["np"], req_param["np"])
        self.dim = safe_assign("dim", self.param["dim"], req_param["dim"])
        self.tmax = safe_assign("tmax", self.param["tmax"], req_param["tmax"])

        self.cells = np.expand_dims(np.arange(0, self.np, dtype=np.int), axis=1)

    def __process_ocl_src(self, print_src=False, module_file="."):

        # Relative path from simulation.py file
        kernel_path = os.path.join(os.path.dirname(__file__), "kernels")
        kernel_path = os.path.join(kernel_path, self.src_file)

        with open(kernel_path, "r") as f:
            src = f.read()

        print("Info- simulation's parameters:")

        for k, v in self.param.items():
            if is_num_type(v):
                if isinstance(v, int):
                    if k == "np":
                        print("\t- {:<12}  {:<1.0e}".format(k, v))
                    else:
                        print("\t- {:<12}  {:<12d}".format(k, v))
                    src = src.replace("_{}_".format(k), "({})".format(v))

                if isinstance(v, float):
                    print("\t- {:<12}  {:<12.6f}".format(k, v))
                    src = src.replace("_{}_".format(k), "({}f)".format(v))

        if print_src:
            print(src)

        return src

    def __build_and_setup_ocl(self):
        self.ocl_options = [
            "-I",
            os.path.join(os.path.dirname(__file__), "kernels"),
            "-Werror",
            "-cl-fast-relaxed-math",
        ]

        self.ocl_ctx = cl.create_some_context(interactive=True)
        self.ocl_prg = cl.Program(self.ocl_ctx, self.source).build(
            options=self.ocl_options
        )

        self.ocl_prop = cl.command_queue_properties.PROFILING_ENABLE
        self.ocl_queue = cl.CommandQueue(self.ocl_ctx, properties=self.ocl_prop)

    def __init_particle(self):
        print("Info- init particles")
        gen = PhiloxGenerator(self.ocl_ctx)

        self.x_gpu = cl_array.empty(
            self.ocl_queue, self.dim * self.np, dtype=self.dtype
        )

        # Init position on a sphere of diameter 0.05 and center (mu,mu,mu)
        # self.x_gpu = gen.normal(
        #     self.ocl_queue, (self.np * self.dim), self.dtype, mu=0.5, sigma=0.05
        # )

        # Init velocity
        self.v_gpu = gen.normal(
            self.ocl_queue, (self.np * self.dim), self.dtype, mu=0, sigma=1
        )

        # Init time
        self.t_gpu = cl_array.zeros(self.ocl_queue, self.np, dtype=self.dtype)

        self.ocl_prg.rt_init_particles(
            self.ocl_queue,
            (self.np,),
            None,
            self.x_gpu.data,
            self.v_gpu.data,
        ).wait()

    def __push_particle(self):
        gen = PhiloxGenerator(self.ocl_ctx)

        rand_gpu = gen.uniform(self.ocl_queue, (self.np, 4), dtype=self.dtype)

        self.ocl_prg.rt_push_particles(
            self.ocl_queue,
            (self.np,),
            None,
            rand_gpu.data,
            self.x_gpu.data,
            self.v_gpu.data,
            self.t_gpu.data,
        ).wait()

    def dump_particles(self):
        # Copy position array from GPU to CPU
        points = self.x_gpu.get().reshape((self.np, self.dim))
        cells = {"vertex": self.cells}
        # This field is usefull to setup energy
        value = np.ones(self.np)

        mesh = meshio.Mesh(points, cells, point_data={"f": value})
        mesh.write(self.xdmf_filename)

    def solve(self):
        self.__build_and_setup_ocl()
        self.__init_particle()

        iter_max = 20
        for i in range(0, 20):
            self.__push_particle()
            print("Info- iteration: {}/{} ".format(i, iter_max), end="\r")

        print("Info- ray-tracing done")
        print("Info- dumping particles...")
        self.dump_particles()
        print("Info- particles dumped in file: {}".format(self.xdmf_filename))