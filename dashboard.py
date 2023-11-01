import numpy as np

# UNITS
degrees = np.pi / 180
micrometers = 1
meters = 1e6 * micrometers
centimeters = 1e-2 * meters
millimeters = 1e-3 * meters
nanometers = 1e-3 * micrometers
picoseconds = 1
seconds = 1e12 * picoseconds
hertz = 1 / seconds
kilohertz = 1e3 * hertz
megahertz = 1e6 * hertz
gigahertz = 1e9 * hertz
terahertz = 1e12 * hertz

# CONSTANTS
e0 = 8.85418782e-12 * 1 / meters
u0 = 1.25663706e-6 * 1 / meters
N0 = u0 / e0
c0 = 299792458 * meters / seconds

# OCT source
theta = 0 * degrees
l0 = 800 * nanometers
bw_l = 50 * nanometers
num_sample_pts = 100

# DEVICE
NRES = 10
NPML = [20, 20, 20, 20]
Sy = 40 * micrometers

# MATERIAL PROPERTIES
ermin = 1
ermax = 3.5
zmin = 1
zmax = 35

# SIMULATION
default_id = 1
default_num_simulations = 1

# DIRECTORIES
OUT_DIR = "Dataset"

# SLURM
slurm_offset = 200
slurm_dir = "/work/users/k/e/kenyavzz/Dataset"
