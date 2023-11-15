import numpy as np
from scipy.sparse import diags, linalg, csc_matrix
import time
import random


# ------------------ Electric Simulation Functions ------------------
def yeeder2d(grid_size, resolution, k=None):
    """
    Function that computes the scipy sparse derivative matrices on the 2D Yee Grid
    Arguments:
        grid_size = two-tuple of the size of the grid
        resolution = two-tuple of the grid resolution (i.e. incremental step)
        k = incident waves

    Returns:
    Derivative matrices with Dirichlet conditions in the x and y direction (DEx, DEy, DHx, DHy)
    """
    # Extracting grid parameters
    Nx = grid_size[0]
    Ny = grid_size[1]
    dx = resolution[0]
    dy = resolution[1]

    # Default indicent wave
    if k is None:
        k_incident = np.zeros(2)

    # Matrix size and zero matrix
    M = int(Nx * Ny)

    ### Building DEx
    if Nx == 1:
        DEx = -1j * k_incident[0] * np.eye(M)
    else:
        d0 = - np.ones(M)
        d1 = np.ones(M - 1)
        d1[range(Nx - 1, M - 1, Nx)] = 0
        DEx = (1 / dx) * diags([d0, d1], [0, 1])


    ### Building DEy
    if Ny == 1:
        DEy = -1j * k_incident[1] * np.eye(M)
    else:
        d0 = - np.ones(M)
        d1 = np.ones(int(M - Nx))
        DEy = (1 / dy) * diags([d0, d1], [0, Nx])


    ### Build DHx and DHy
    DHx = - DEx.transpose()
    DHy = - DEy.transpose()

    # Transforming to sparse matrices
    DEx = csc_matrix(DEx)
    DEy = csc_matrix(DEy)
    DHx = csc_matrix(DHx)
    DHy = csc_matrix(DHy)

    return DEx, DEy, DHx, DHy


def addupml2d_sparse(ER2, UR2, NPML):
    """
    Computes the Electric Permittivity and Magnetic Permeability on a 2xGrid after adding a Perfectly Matched Layer.
    The electromagnetic properties are flattened and turned into diagonal scipy sparse matrices
    Arguments:
        ER2 Relative permittivity on 2x grid (make sure it is a numpy array)
        UR2 Relative permeability on 2x grid
        NPML [NXLO NXHI NYLO NYHI] Size of UPML on a 1x grid
    Returns:
        ERxx xx tensor element for relative electric permittivity
        ERyy yy tensor element for relative electric permittivity
        ERzz zz tensor element for relative electric permittivity
        URxx xx tensor element for relative magnetic permeability
        URyy yy tensor element for relative magnetic permeability
        URzz zz tensor element for relative magnetic permeability
    """
    # Define PML parameters
    amax = 4
    cmax = 1
    p = 3

    # Extract grid parameters
    [Nx2, Ny2] = np.shape(ER2)

    # Extract PML parameters
    NXLO = 2 * NPML[0]
    NXHI = 2 * NPML[1]
    NYLO = 2 * NPML[2]
    NYHI = 2 * NPML[3]

    # Calculate PML Parameters to problem space
    sx = np.ones((Nx2, Ny2), dtype=np.csingle)
    sy = np.ones((Nx2, Ny2), dtype=np.csingle)

    # Add XLO PML
    for i in range(NXLO):
        ax = 1 + (amax - 1) * ((i + 1) / NXLO) ** p
        cx = cmax * np.sin(0.5 * np.pi * (i + 1) / NXLO) ** 2
        sx[int(NXLO - (i + 1)), :] = ax * (1 - 1j * 60 * cx)

    # Add XHI PML
    for i in range(NXHI):
        ax = 1 + (amax - 1) * ((i + 1) / NXHI) ** p
        cx = cmax * np.sin(0.5 * np.pi * (i + 1) / NXHI) ** 2
        sx[int(Nx2 - NXHI + i), :] = ax * (1 - 1j * 60 * cx)

    # Add YLO PML
    for i in range(NYLO):
        ay = 1 + (amax - 1) * ((i + 1) / NYLO) ** p
        cy = cmax * np.sin(0.5 * np.pi * (i + 1) / NYLO) ** 2
        sy[:, int(NYLO - (i + 1))] = ay * (1 - 1j * 60 * cy)

    # Add YHI PML
    for i in range(NYHI):
        ay = 1 + (amax - 1) * ((i + 1) / NYHI) ** p
        cy = cmax * np.sin(0.5 * np.pi * (i + 1) / NYHI) ** 2
        sy[:, int(Ny2 - NYHI + i)] = ay * (1 - 1j * 60 * cy)

    # Calculate tensor elements with UPML
    ERxx = ER2 / sx * sy
    ERyy = ER2 * sx / sy
    ERzz = ER2 * sx * sy

    URxx = UR2 / sx * sy
    URyy = UR2 * sx / sy
    URzz = UR2 * sx * sy

    # Extract tensor elements on yee grid
    ERxx = ERxx[1:Nx2:2, 0:Ny2:2]
    ERyy = ERyy[0:Nx2:2, 1:Ny2:2]
    ERzz = ERzz[0:Nx2:2, 0:Ny2:2]

    URxx = URxx[0:Nx2:2, 1:Ny2:2]
    URyy = URyy[1:Nx2:2, 0:Ny2:2]
    URzz = URzz[1:Nx2:2, 1:Ny2:2]

    # Flatten arrays column major style
    ERxx = ERxx.flatten('F')
    ERyy = ERyy.flatten('F')
    ERzz = ERzz.flatten('F')
    URxx = URxx.flatten('F')
    URyy = URyy.flatten('F')
    URzz = URzz.flatten('F')

    # Calculating the multiplicative inverse of the permeability and permittivity matrices
    ERxx_i = 1 / ERxx
    ERyy_i = 1 / ERyy
    URxx_i = 1 / URxx
    URyy_i = 1 / URyy

    # Creating a Sparse Diagonal and transforming it to an array
    ERxx = diags(ERxx, 0)
    ERyy = diags(ERyy, 0)
    ERzz = diags(ERzz, 0)
    URxx = diags(URxx, 0)
    URyy = diags(URyy, 0)
    URzz = diags(URzz, 0)

    # Creating sparse diagonals of inverses
    ERxx_i = diags(ERxx_i, 0)
    ERyy_i = diags(ERyy_i, 0)
    URxx_i = diags(URxx_i, 0)
    URyy_i = diags(URyy_i, 0)

    return ERxx, ERyy, ERzz, URxx, URyy, URzz, ERxx_i, ERyy_i, URxx_i, URyy_i


def solve_electric_field(k, theta, nsrc, dx, dy, NS, grid, Q, ER_UR_PML):
    """
    Solves for an electric field using the methodology in 'Electromagnetic and Photonic Simulation for the Beginner:
    Finite-Difference Frequency-Domain in Matlab' by RC Rumpf (2022)
    Arguments:
        k (float): Wavenumber (2 * pi / wavelength) for which the simulation will be solved
        theta (float): Angle of incidence (in radians) of the source field
        nsrc (float): refractive index of the source field
        dx (float): finite difference in space over the width of the device to be simulated
        dy (float): finite difference in the space over the length of the device to be simulated
        NS (int Nx, int Ny): integer tuple of the size of the device in pixels
        grid (Meshgrid): meshgrid of the distances in the 2D space
        Q (sparse diagonal matrix): flattened and diagonalized matrix of the source field mask Q
        ER_UR_PML (ERxx, ERyy, ERzz, URxx, URyy, URzz, ERxx_i, ERyy_i, URxx_i, URyy_i): Tuple of the electromagnetic
                    properties of the device as computed by the function addupml2d_sparse
    Returns:
        simulated electric field and the electric field of the source
    """
    # Start the time for simulation
    t_i = time.time()

    # Extracting the necessary components from the tuple of grid
    X = grid[0]
    Y = grid[1]
    Nx = NS[0]
    Ny = NS[1]

    # Extracting the necessary permittivity and permeability components from addupml2d_sparse
    ERzz = ER_UR_PML[2]
    URxx_i = ER_UR_PML[8]
    URyy_i = ER_UR_PML[9]

    # Compute wavenumbers for the incident (source) field
    kxinc = k * nsrc * np.sin(theta)
    kyinc = k * nsrc * np.cos(theta)
    kinc = [kxinc / k, kyinc / k]

    # Build derivative matrices
    RES = [k * dx, k * dy]
    DEX, DEY, DHX, DHY = yeeder2d(NS, RES, kinc)

    # Build wave matrix
    A = DHX @ URyy_i @ DEX + DHY @ URxx_i @ DEY + ERzz

    # Calculate source field
    fsrc = np.exp(-1j * (kxinc * X + kyinc * Y))
    fsrc_flat = fsrc.flatten('F')

    # Calculate source vector
    b = (Q @ A - A @ Q) @ fsrc_flat

    # Solve for field
    f = linalg.spsolve(A, b)
    f = np.reshape(f, (Nx, Ny), 'F')
    t_t = time.time()
    print(" Electric field computed in", t_t - t_i, " seconds")

    return f, fsrc


# ------------------ A-Line Function ------------------
def compute_a_line(wavenumbers, source_intensity, reflected_fields, delta_z, intensity=False):
    """
    Transforms an array of reflected fields of shape (num of wavenumber samples x width of device) into its
    corresponding A-line by using an inverse fourier transform along with other OCT processing practices.
    For more information, refer to "Theory of optical coherence tomography" by Izatt and Choma
    Arguments:
        wavenumbers: the sampled wavenumbers from the OCT source spectrum
        source_intensity: the intensity of the source spectrum
        reflected_fields: the compilation of reflected fields along the width of the device for each of the sampled
                          wavenumbers
        delta_z: the distance incremental (for the IFFT) computed as delta_z = 2*pi / (2 * N * delta_k) where Nis the
                 the number of samples, delta_k is the sampling interval of the wavenumbers
        intensity (boolean): determines if it returns the intensity of the A-line (real) or not (complex)
    Returns:
        The A-line of the measured intensity from the reflected fields. The A-line contains "negative distances" and
        positive distances as a result of the Inverse Fourier Transform. The midpoint of the array should be centered
        zero
    """
    # Compute Source Amplitude
    source_amplitude = np.sqrt(source_intensity)

    # Creating electric fields from reference and sample arm. The reference arm is placed arbitrarily at 0
    reflector_distance = 0
    reflector_reflectivity = 1
    reflector_propagation = np.exp(1j * 2 * reflector_distance * wavenumbers)
    E_reflector = 1 / np.sqrt(2) * source_amplitude * reflector_reflectivity * reflector_propagation
    E_sample = 1 / np.sqrt(2) * np.expand_dims(source_amplitude, -1) * reflected_fields

    # Intensity of the interferogram recorded in an instantaneous detector
    I_D = 1 / 2 * np.abs(np.expand_dims(E_reflector, -1) + E_sample) ** 2

    # Intensity of the interferogram without the sample arm
    I_R = 1 / 2 * np.abs(E_reflector) ** 2

    # Subtracting the background from the interferogram
    background_subtracted = I_D - np.expand_dims(I_R, -1)

    # Adding the negative frequencies and DC term to the standard signal
    negative_freqs = np.zeros_like(I_D[1:, :])
    standard_signal = np.concatenate((background_subtracted, negative_freqs), axis=0)
    signal_mean = np.mean(standard_signal, axis=0)
    standard_signal[0] = signal_mean

    # FFT of ID
    a_line = np.fft.ifft(standard_signal, axis=0)

    # Correct order of the FFT
    num_sample_pts = wavenumbers.shape[-1]
    d_positive = delta_z * np.arange(0, num_sample_pts, 1)
    d_negative = delta_z * np.arange(1 - num_sample_pts, 0, 1)
    depth = np.concatenate((d_negative, d_positive))
    a_line_ro = np.concatenate((a_line[num_sample_pts:], a_line[:num_sample_pts]))

    return np.abs(a_line_ro) if intensity else a_line_ro


# ------------------ HELPER FUNCTIONS ------------------
def source_spectrum(l0, bw, tail_length=1.7, num_samples=100):
    """
    Computes the source spectrum values in the wavenumber domain
    Arguments:
        l0 (float): central wavelength
        bw (float): bandwidth of the source spectrum
        tail_length (float): number of standard deviations away from the center wavelength to compute
        num_samples (int): number of samples to compute the source spectrum
    Returns:
        wavenumbers (array): wavenumbers that are associated to the source spectrum
        spectral_amplitude (array): amplitude of a gaussian source spectrum
    """
    # Transforming from the wavelength to wavenumber domain
    k_0 = 2 * np.pi / l0
    bw_k = (2 * np.pi / (l0 - bw / 2)) - (2 * np.pi / (l0 + bw / 2))

    # Determining the domain of the source spectrum based on the tail length
    k_min = k_0 - tail_length * bw_k
    k_max = k_0 + tail_length * bw_k

    # Computing the wavenumbers and spectral amplitude
    wavenumbers = np.linspace(k_min, k_max, num_samples)
    spectral_amplitude = 1 / (np.sqrt(np.pi) * bw_k / 2) * np.exp(- ((wavenumbers - k_0) / (bw_k / 2)) ** 2)

    return wavenumbers, spectral_amplitude


def one_layered_medium(er1, er2, nz, Nx, Ny):
    """
    Create the electric permittivity device for a one_layered medium
    Arguments:
        er1 (float): Electric permittivity of the first medium. Must be gte 1
        er2 (float): Electric permittivity of the second medium. Must be gte 1
        nz (int): Distance (in pixels) at which the medium changes. The distance already takes into account the PML
        Nx (int): width of the device in pixels
        Ny (int): height of the device in pixels
    Returns:
        ER (array): Electric permittivity of a one layered medium
    """
    # Compute ER
    ER = er1 * np.ones((Nx, Ny))

    # Build the layered medium
    ER[:, nz:] = er2
    return ER


def two_layered_medium(er1, er2, er3, nz1, nz2, Nx, Ny):
    """
    Create the electric permittivity device for a two-layered medium
    Arguments:
        er1 (float): Electric permittivity of the first medium. Must be gte 1
        er2 (float): Electric permittivity of the second medium. Must be gte 1
        er3 (float): Electric permittivity of the third medium. Must be gte 1
        nz1 (int): Distance (in pixels) at which the medium changes. The distance already takes into account the PML
        nz2 (int): Distance (in pixels) at which the medium changes. The distance already takes into account the PML
        Nx (int): width of the device in pixels
        Ny (int): height of the device in pixels
    Returns:
        ER (array): Electric permittivity of a one layered medium
    """
    # Compute ER
    ER = er1 * np.ones((Nx, Ny))

    # Build the layered medium
    ER[:, nz1:nz2] = er2
    ER[:, nz2:] = er3
    return ER


def three_layered_medium(er1, er2, er3, er4, nz1, nz2, nz3, Nx, Ny):
    """
    Create the electric permittivity device for a three layered medium
    Arguments:
        er1 (float): Electric permittivity of the first medium. Must be gte 1
        er2 (float): Electric permittivity of the second medium. Must be gte 1
        er3 (float): Electric permittivity of the third medium. Must be gte 1
        nz1 (int): Distance (in pixels) at which the medium changes. The distance already takes into account the PML
        nz2 (int): Distance (in pixels) at which the medium changes. The distance already takes into account the PML
        Nx (int): width of the device in pixels
        Ny (int): height of the device in pixels
    Returns:
        ER (array): Electric permittivity of a one layered medium
    """
    # Compute ER
    ER = er1 * np.ones((Nx, Ny))

    # Build the layered medium
    ER[:, nz1:nz2] = er2
    ER[:, nz2:nz3] = er3
    ER[:, nz3:] = er4
    return ER


def circles_2D(Sx, Sy, dx, dy, NPML, S_avg_radius, S_std_radius, er_max, random_seed, noise=True):
    """
    Creates a 2D electric permittivity device with 2-5 circles
    Arguments:
        Sx: Size of the x coordinate in micrometers
        Sy: Size of the y coordinate in micrometers
        dx: Incremental in x in micrometers
        dy: Incremental in y in micrometers
        NPML: 4-element array describing the NPML range in pixels
        S_avg_radius: Size of the average radius of our circles in micrometers
        S_std_radius: Standard deviation of the radius of our circles in micrometers
        er_max: maximum value of our the electric permittivity in our device
        random_seed: random seed for the pseudo-random number generator
        noise (bool): Adds scatterer noise (i.e. particles of radius 0.5 micrometers)
    Returns:
        ER (array): Electric permittivity of randomly placed circles
    """
    # Compute the size in pixels
    Nx = int(NPML[0] + np.ceil(Sx/dx) + NPML[1])
    Ny = int(NPML[2] + np.ceil(Sy/dy) + NPML[3])

    # Compute ER background solution
    random.seed(random_seed)
    er_solution = random.uniform(1, 1.1)
    ER = er_solution * np.ones((Nx, Ny))

    # Determine the number of circles
    random.seed(random_seed)
    num_circles = random.randint(2, 5)

    # Create a list of tuples of the centers of the circles
    N_avg_radius = np.ceil(S_avg_radius / dx)
    N_avg_diam = 2 * N_avg_radius
    random.seed(random_seed)
    pixel_range_x = int(Nx - N_avg_diam)
    pixel_range_y = int(Ny - N_avg_diam)
    centers_range = [divmod(x, pixel_range_x) for x in random.sample(range(pixel_range_x * pixel_range_y), num_circles)]
    centers = [tuple(map(sum, zip(x, (NPML[0] + N_avg_radius, NPML[2] + N_avg_radius)))) for x in centers_range]

    # Create radius
    np.random.seed(random_seed)
    radii_normal_dist = sorted(np.random.normal(S_avg_radius, S_std_radius, 1000))
    random.seed(random_seed)
    radii = random.sample(radii_normal_dist, num_circles)
    radii_n = []
    for i in range(num_circles):
        radius_s = abs(radii[i])
        radius_n = radius_s / dx
        # Resizing radius so that it is inside the left NPML
        if centers[i][0] - radius_n < NPML[0]:
            difference = NPML[0] - (centers[i][0] - radius_n)
            radius_n = radius_n - difference
        # Resizing radius so that it is inside the right NPML
        if (centers[i][0] + radius_n) > (Nx - NPML[1]):
            difference = (centers[i][0] + radius_n) - (Nx - NPML[1])
            radius_n = radius_n - difference
        # Resizing radius so that it is inside the top NPML
        if centers[i][1] - radius_n < NPML[2]:
            difference = NPML[2] - (centers[i][1] - radius_n)
            radius_n = radius_n - difference
        # Resizing radius so that it is inside the bottom NPML
        if (centers[i][1] + radius_n) > (Ny - NPML[3]):
            difference = (centers[i][1] + radius_n) - (Ny - NPML[3])
            radius_n = radius_n - difference

        radii_n.append(radius_n)

    # Removing overlapping between circles
    for i in range(num_circles):
        for j in range(i, num_circles):
            if i != j:
                distances_between_circles = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                while (radii_n[i] + radii_n[j]) > distances_between_circles:
                    radii_n[i] -= 1
                    radii_n[j] -= 1

    # Filling in the circles
    er_vals = sorted(np.linspace(er_solution, er_max, 1000))
    random.seed(random_seed)
    er = random.sample(er_vals, num_circles)

    for k in range(num_circles):
        center = centers[k]
        radius = radii_n[k]
        for i in range(Nx):
            for j in range(Ny):
                if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2:
                    ER[i, j] = er[k]

    # Creating scatterers if noise
    if noise:
        num_scatterers = 100
        random.seed(random_seed)
        scatterers_centers_range = [divmod(x, pixel_range_x) for x in random.sample(range(pixel_range_x * pixel_range_y), num_scatterers)]
        scatterers_centers = [tuple(map(sum, zip(x, (NPML[0] + N_avg_radius, NPML[2] + N_avg_radius)))) for x in scatterers_centers_range]
        radius = 0.5 / dx
        noise_dist = sorted(np.linspace(0.8, 1.2, 1000))
        random.seed(random_seed)
        random_noise = random.sample(noise_dist, num_scatterers)
        for k in range(len(scatterers_centers)):
            noise_prod = random_noise[k]  # Increase or decrease the current ER value by 20%
            center = scatterers_centers[k]
            for i in range(Nx):
                for j in range(Ny):
                    if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius **2:
                        ER[i, j] = noise_prod * ER[i, j]

    return ER
