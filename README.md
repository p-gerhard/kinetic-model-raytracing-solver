# Kinetic-ray-tracer
Solving the linear kinetic transport equation in cubic geometries with a Monte-Carlo (MC) ray 
tracing method on GPU. The code deals with absorbant (`alpha` coefficient), diffuse, and specular (`beta` and `1-beta` coefficients) boundary conditions.
The Monte-Carlo algorithm is event based. At each iteration we transport the particles from one boundary to the other. 
The particles are exported at final simulation time in `xmdf/h5` file format.

### How to Build and Install
This project is written in Python3 and OpenCL. It mostly relies on the package [PyOpenCL](https://github.com/inducer/pyopencl "PyOpenCL").
To use it on your host system, follow the steps:
1. Ensure that you have a valid OpenCL eco-system (compatible driver, header, compiler) installed on your machine.
2. `git clone https://github.com/p-gerhard/kinetic-ray-tracer.git` -- download the source
3. `pip install -r requirements.txt` -- It will install the dependencies

### Example
To run the provided example, type :

`python ./tests/solve_example.py`
