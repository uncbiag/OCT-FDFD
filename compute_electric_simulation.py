import pandas as pd
import random
import os
from electric_funcs import *
from dashboard import *


if os.getenv("SLURM_ARRAY_TASK_ID") is None:
    running_slurm = False
else:
    running_slurm = True


def main():
    # Creating the OCT source
    wavenumbers, source_intensity = source_spectrum(l0, bw_l, tail_length=1.7, num_samples=num_sample_pts)
    k_min = wavenumbers[0]
    k_max = wavenumbers[-1]
    l_min = 2 * np.pi / k_max
    l_max = 2 * np.pi / k_min
    bw_k = (2 * np.pi / (l0 - bw_l / 2)) - (2 * np.pi / (l0 + bw_l / 2))

    # Checking critical values - Only necessary to make sure that the OCT simulation is within the theoretical limits
    nyquist_spacing = 1 / (2 * bw_k)
    axial_resolution = 2 * np.log(2) / np.pi * l0 ** 2 / bw_l
    delta_k = wavenumbers[1] - wavenumbers[0]
    max_imaging_depth = np.pi / (2 * delta_k)
    delta_z = np.pi / (2 * delta_k * num_sample_pts)

    # Creating an empty dataset if running the simulation locally
    if not running_slurm:
        columnlist = ["id", "num_layers", "er", "r", "z", "k_0", "k_min", "k_max", "bw", "dx", "dy", "dz", "dk"]
        df = pd.DataFrame(columns=columnlist)

    # Create id based on slurm task_id
    if running_slurm:
        id = slurm_offset + int(os.getenv("SLURM_ARRAY_TASK_ID"))
    else:
        id = default_id

    # Initialize random seed based on id
    random.seed(id)

    # Define FDFD Parameters
    SPACER = l_max
    nmax = np.sqrt(ermax)

    # Optimized grid
    dx = l_min / nmax / NRES
    dy = l_min / nmax / NRES

    # Create length of grid (width is 1 pixel, length depends on the length of device in dashboard
    Nx = int(NPML[0] + 1 + NPML[1])
    Ny = int(NPML[2] + np.ceil(Sy / dy) + NPML[3])

    # 2X Grid
    Nx2 = 2 * Nx
    Ny2 = 2 * Ny
    dx2 = dx / 2
    dy2 = dy / 2

    # Calculate axis vectors
    xa = np.arange(1, Nx + 1, 1) * dx
    ya = np.arange(1, Ny + 1, 1) * dy
    [Y, X] = np.meshgrid(ya, xa)
    grid = (X, Y)

    # Number of simulations and where to store them
    if running_slurm:
        num_sim = 1
        outdir = slurm_dir
    else:
        num_sim = default_num_simulations
        outdir = OUT_DIR

    t_0 = time.time()
    for j in range(num_sim):
        # Randomize the number of layers between 1 and 3
        num_layers = random.randint(1, 3)

        # Compute er1 and initialize a list of the electric permitivities and distances
        er1 = random.uniform(ermin, ermax)
        er = [er1]
        z = []
        r = []

        # Computing the rest of the layers
        for i in range(num_layers):
            er_i = random.uniform(ermin, ermax)
            er.append(er_i)
            n1 = np.sqrt(er[i])
            n2 = np.sqrt(er_i)
            theta2 = np.arcsin((n1 * np.sin(theta)) / n2)
            r_i = (n1 * np.cos(theta) - n2 * np.cos(theta2)) / (n1 * np.cos(theta) + n2 * np.cos(theta2))
            r.append(r_i)
            if i == 0:
                z_i = random.uniform(zmin, zmax)
                z.append(z_i)
            else:
                z_i = random.uniform(z[i-1], zmax)
                z.append(z_i)

        # Adding values to dataframe
        if not running_slurm:
            new_row = pd.Series({"id": id, "num_layers": num_layers, "er": er, "r": r, "z": z,
                             "k_0": wavenumbers[int(num_sample_pts / 2) - 1], "k_min": k_min,
                             "k_max": k_max, "bw": bw_k, "dx": dx, "dy": dy, "dz": delta_z, "dk": delta_k})
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

        print("###### Computing Simulation {0:03d} of {1} ######".format(j + 1, num_sim))

        # Computing the Layered Medium
        if num_layers == 1:
            ER1 = one_layered_medium(er[0], er[1], int(NPML[2] + np.round(z[0] / dy)), Nx, Ny)
            ER2 = one_layered_medium(er[0], er[1], int(NPML[2] + np.round(z[0] / dy2)), Nx2, Ny2)
        elif num_layers == 2:
            ER1 = two_layered_medium(er[0], er[1], er[2], int(NPML[2] + np.round(z[0] / dy)),
                                     int(NPML[2] + np.round(z[1] / dy)), Nx, Ny)
            ER2 = two_layered_medium(er[0], er[1], er[2], int(NPML[2] + np.round(z[0] / dy2)),
                                     int(NPML[2] + np.round(z[1] / dy2)), Nx2, Ny2)
        elif num_layers == 3:
            ER1 = three_layered_medium(er[0], er[1], er[2], er[3], int(NPML[2] + np.round(z[0] / dy)),
                                       int(NPML[2] + np.round(z[1] / dy)), int(NPML[2] + np.round(z[2] / dy)), Nx, Ny)
            ER2 = three_layered_medium(er[0], er[1], er[2], er[3], int(NPML[2] + np.round(z[0] / dy2)),
                                       int(NPML[2] + np.round(z[1] / dy2)), int(NPML[2] + np.round(z[2] / dy2)), Nx2, Ny2)

        # Magnetic permeability
        UR2 = np.ones((Nx2, Ny2))

        # Save the Layered medium
        np.save("{1}/ER/ER_{0}.npy".format(id, outdir), ER1)

        # Incorporate PML
        ER_UR_PML = addupml2d_sparse(ER2, UR2, NPML)

        # Parameters for FDFD
        nsrc = np.sqrt(UR2[0, 0] * ER2[0, 0])
        NS = [Nx, Ny]
        BC = [0, 0]
        Q = np.zeros((Nx, Ny))
        Q[:, 0:int(NPML[0] + 2)] = 1

        # Diagonalize masking matrix and source field
        Q = Q.flatten('F')
        Q = diags(Q, 0)

        # Prepare array to store reflected field
        reflected_fields = np.array([]).reshape(0, num_sample_pts)
        t_0_s = time.time()

        # Compute the simulated electric field
        for i in range(num_sample_pts):
            k0 = wavenumbers[i]
            f, fsrc = solve_electric_field(k0, theta, nsrc, dx, dy, NS, grid, Q, ER_UR_PML)

            # Analyze reflection
            fref = f[:, NPML[0]] / fsrc[:, 0]
            reflected_fields = np.vstack([reflected_fields, fref]) if reflected_fields.size else fref

        # Save reflected electric fields array
        np.save("{1}/ReflectedFields/RF_{0}.npy".format(id, outdir),
                reflected_fields)

        # Printing the simulation time
        t_f_s = time.time()
        print("Time to complete simulation", t_f_s - t_0_s, "s")

        # Computing the A-Line (Complex) and the A-Line Intensity (Real)
        a_line_c = compute_a_line(wavenumbers, source_intensity, reflected_fields, delta_z, intensity=False)
        a_line_r = compute_a_line(wavenumbers, source_intensity, reflected_fields, delta_z, intensity=True)
        np.save("{1}/AL/AL_{0}.npy".format(id, outdir), a_line_c)
        np.save("{1}/ALI/AL_{0}.npy".format(id, outdir), a_line_r)

        # Continuously save dataframe
        if not running_slurm:
            df.to_csv("reflected_fields_dataset_v2.csv", index=False)

        # Increase identification number
        if not running_slurm:
            id += 1

    # Computing time of building dataset
    t_f = time.time()
    print("Total time to build dataset: ", (t_f - t_0) / 3600, " hrs")

if __name__ == '__main__':
    main()
