import os
from electric_funcs import *
from dashboard import *
import matplotlib.pyplot as plt


if os.getenv("SLURM_ARRAY_TASK_ID") is None:
    running_slurm = False
else:
    running_slurm = True


def main():
    # Creating the OCT source
    l0 = 0.800
    bw_l = 0.100
    num_sample_pts_2D = 300
    wavenumbers, source_intensity = source_spectrum(l0, bw_l, tail_length=4, num_samples=num_sample_pts_2D)
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
    delta_z = np.pi / (2 * delta_k * num_sample_pts_2D)

    # Create id based on slurm task_id
    if running_slurm:
        id = slurm_offset + int(os.getenv("SLURM_ARRAY_TASK_ID"))
    else:
        id = default_id
        id = "test"

    # Define FDFD Parameters
    SPACER = l_max
    nmax = np.sqrt(ermax)

    # Optimized grid
    dx = l_min / nmax / NRES
    dy = l_min / nmax / NRES

    # Create length of grid (width is 1 pixel, length depends on the length of device in dashboard
    Sx = 10
    Sy = 10
    Nx = int(NPML[0] + np.ceil(Sx / dx) + NPML[1])
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
        outdir = slurm_dir_2D
    else:
        num_sim = default_num_simulations
        outdir = OUT_DIR_2D

    t_0 = time.time()
    for j in range(num_sim):
        print("###### Computing Simulation {0:03d} of {1} ######".format(j + 1, num_sim))
        # ER1 = circles_2D(Sx, Sy, dx, dy, NPML, s_avg_radius, s_std_radius, ermax, id, noise=True)
        # ER2 = circles_2D(Sx, Sy, dx2, dy2, 2*np.array(NPML), s_avg_radius, s_std_radius, ermax, id, noise=False)

        # Testing the simplest ER2
        ER2 = 1.77 * np.ones((Nx2, Ny2))
        radius = int(np.ceil(3.5/dx2))
        center = (int(np.ceil(5/dx2)), int(np.ceil(4/dx2)))
        for i in range(Nx2):
            for j in range(Ny2):
                if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2:
                    ER2[i, j] = 2.8

        # # Visualizing ER and checking its reproducibility in ER2
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 16))
        # im = ax1.imshow(ER1.T)
        # fig.colorbar(im, ax=ax1)
        # ax1.axis('scaled')
        # ax1.set_title('ER')
        #
        # im = ax2.imshow(ER2.T)
        # fig.colorbar(im, ax=ax2)
        # ax2.axis('scaled')
        # ax2.set_title('ER2')
        # plt.show()

        # Magnetic permeability
        UR2 = np.ones((Nx2, Ny2))

        # Save the Layered medium
        np.save("{1}/ER/ER_{0}.npy".format(id, outdir), ER2)

        # Visualize ER
        visualize_er("{1}/ER/ER_{0}.npy".format(id, outdir), Sx, Sy)

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
        reflected_fields = np.array([]).reshape(0, num_sample_pts_2D)
        t_0_s = time.time()

        # Compute the simulated electric field
        for i in range(num_sample_pts_2D):
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

        # Increase identification number
        if not running_slurm:
            id += 1

        # Computing time of building dataset
    t_f = time.time()
    print("Total time to build dataset: ", (t_f - t_0) / 3600, " hrs")

    # Visualizing the b-scan
    visualize_a_line("{1}/ALI/AL_{0}.npy".format(id, outdir), delta_z, Sx, Sy, avg_ri=np.mean(np.sqrt(ER2)))


if __name__ == '__main__':
    main()
