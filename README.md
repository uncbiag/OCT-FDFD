# OCT-FDFD
Simulates OCT images by solving Maxwell's equations through a Finite-Difference Frequency-Domain method

## Use

### Modifying the dashboard

**OCT Source**

The parameters needed to simulate the OCT source can be found in `dashboard.py`. It is recommended that you do not modify the units and the constants. If you wish to modify the OCT source, keep in mind that the bandwidth affects the Nyquist sampling frequency. For OCT, the broader the bandwidth, the more resolution. However, the broader the bandwidth, the harder it is to get the sampling frequency higher than the Nyquist frequency. Refer to the variables `nyquist_spacing` and `delta_k` in `compute_electric_simulation.py` to see if `delta_k < nyquist_spacing`.  

**Properties of the Layered Medium**

To modify the physical size of the device, change variables `Sy` and `Sx`. The variable `PML` contains the dimensions --in pixels-- of the Perfectly Matched Layer so that $PML=[PML_{XLO}, PML_{XHI}, PML_{YLO}, PML_{YHI}]$. The variable `NRES` defines the number of pixels that are used to define the smallest wavelength in the simulation.

The material properties in the layered medium can be modified to determine the maximum possible electric permittivity and where the layers should appear in the device. Note that increasing the maximum electric permittivity will affect the Optical Path Distance in the A-Line.

### Running the code on SLURM

The code `compute_electric_simulation.py` can be run on SLURM (Longleaf) using the command:
```
sbatch slurm_electric_field_simulations.sl
```
In `dashboard.py` you can change the simulation ID via the variable `slurm_offset`. Using SLURM arrays, the id will start on the offset and be added to the `SLURM_ARRAY_TASK_ID` environment variable. For more information on using SLURM arrays, you can refer to [this article]( https://blog.ronin.cloud/slurm-job-arrays/ )

Note that the purpose of using the SLURM script is to generate thousands of simulations in a quick manner. These thousands of files will be stored in the Longleaf cluster, so it is necessary to move them to your local computer. To move the files from the Longleaf cluster to your computer, you can use 
```
scp -r onyen@longleaf.unc.edu:PATH/TO/DATASET LOCAL/DIRECTORY
```

Once the files are copied, you can generate an annotation file using the code `make_dataset.py`. For more information on using Longleaf, please refer to [this guide](https://help.rc.unc.edu/getting-started-on-longleaf/)

### Running the Simulations on CPU

Once the desired variables have been modified in the dashboard, the layered-medium electric field computation can be started using
```
python compute_electric_simulation.py
```
You can modify the ID of the simulation and the number of simulations computed in the `dashboard`. Running the simulations on CPU will automatically generate an annotations file.

### Training a Neural Network

Once the simulations and annotation files are available, use the function `split_dataset_layered_medium` to prepare your test and training datasets. For the 1D layered medium example, it suffices to run
```
python MLP.py
```
To train the Multi-Layer Perceptron. Look at the comments on the code to modify any hyperparameters such as the number of hidden layers/nodes, the optimizer, the learning rate, etc. 

If you are training the layer remotely with an `ssh` connection, remember to conect with `-X` or `-Y` to enable X11 forwarding to see the graphics.

### 2D Simulation

Note that the variables for running a 2D simulation with `compute_electric_simulation_2D.py` are also included in the dashboard and can be run in a similar way as the 1D simulation. The slurm script will have to reflect the change in the Python script running.

## Description of Algorithm

### OCT Intensity and A-Line Construction

Optical Coherence Tomography (OCT) is a non-invasive imaging technique that utilizes the principle of coherence to generate images along the depth of a tissue. An OCT device consists of a light source that emits a broad spectrum of light in near-infrared wavelengths, an interferometer that splits the light into two beams, a reference arm that reflects one of the beams back to the interferometer, and a sample arm that directs the other beam to the tissue being imaged.

<p align="center">
  <img alt="Schematic of an interferometer from (Drexler, 2008)" src="https://github.com/uncbiag/OCT-FDFD/blob/main/Readme_imgs/MichelsonInter.png" width=80% heights=80%>
</p>

When both fields return from the reference and sample arms respectively, they are halved in power again and interfere at the detector, which generates a photocurrent given by 

$$ I_D(k, \omega) = \frac{1}{2} |E_R + E_S|^2 $$

On the other hand, we can define the power spectral density of the source electric field $E_i$ as   

$$ S(k) = |s(k, \omega)|^2 = \frac{1}{\Delta k \sqrt{\pi}}e^{- \frac{(k - k_0)}{\Delta k}\^2}$$

Where $k_0$ is the central wavenumber of the light source spectrum and $\Delta k$ is the spectral bandwidth, which corresponds to the half-width-half-maximum of the light source power spectrum.

From the power spectral density, we can simulate the electric field from the reference arm in the following way

$$  E_R = \frac{E_i}{\sqrt{2}} r_R e^{i2kz_R} $$

Where $r_R$ is the reflectivity of the reference reflector, $z_R$ is the distance from the beamsplitter to the reference arm, and the factor of $2$ corresponds to the distance that the light has to travel to and from the reference reflector. 

Similarly, the electric field from the sample arm is defined as

$$  E_S = \frac{E_i}{\sqrt{2} }r_{S}e^{i2kz_S} $$

where $r_{S}$ is the reflectivity of the tissue after the electric field has been scattered back to our reflector. Note that there are already very simple ways of modeling $r_S$, but these methods are very simple and do not consider light-matter interactions.

In order to construct the A-line scan, it is necessary to take the inverse Fourier transform of $I_D(k, \omega)$ and subtract the dominant reference spectrum. That is the inverse Fourier transform will be applied to 

$$  I_D(k) - |E_R|^2 $$

Since the amplitude of the incident field is defined in terms of the wavenumber $k$, the inverse fourier transform will be in terms of depth $z$. To obtain the increment in the $z$-domain, we use the formula $\delta_z = 2 \pi / 2 \text{N} \delta_k$, where N is the number of sample points that are used to model the spectrum amplitude $s(k, \omega)$. 

Computing the A-Line is a very straightforward approach from the equations presented in this writeup, however, we still need to figure out how to simulate the electric field that comes from the sample arm.

### Finite Differences for Maxwell's Equations

The basic characteristics of light, such as refraction, diffraction, and scattering, are explained by Maxwell's equations, which also describe how light travels through and interacts with materials. The frequency-domain formulation of Maxwell's equations is 

$$ \nabla \bullet \left(\[\varepsilon_r\] \tilde{E}\right) = 0 $$

$$ \nabla \bullet \left(\[\mu_r\]\hat{H}) \right) = 0 $$

$$ \nabla \times \tilde{E} = k_0 \[\mu_r\]\hat{H}$$

$$ \nabla \times \hat{H} = k_0 \[\varepsilon_r\] \tilde{E} $$

After expanding these equations in Cartesian coordinates and reducing them to two dimensions, we can use finite differences to come up with the following equation to solve for the electric field (based on the curl equations in Maxwell's equations):

$$ \frac{\partial}{\partial x'} \mu_y^{-1} \frac{\partial}{\partial x'} E_z + \frac{\partial}{\partial y'} \mu_x^{-1} \frac{\partial}{\partial y'} E_z + \varepsilon_z E_z = 0 $$

Note that we can come up with a similar equation to solve for the magnetic field $H_z$. While this equation already enforces Maxwell's equations, a source vector has to be incorporated so that the electric and magnetic fields have solutions different from zero. This source vector $b = Q f_{src}$ is comprised of a mask $Q$ that separates the Total Field (TF) and the Scattered Field (SF) and an initial source field $f_{src}$.In turn, the source function is defined as a plane wave so that

$$f_{src}(x, y) = \exp \[-i(k_{x, inc}x + k_{y, inc} y \]$$

$$k_{x, inc} = k_0 \sqrt{\mu \varepsilon} \sin \theta_{inc}$$

$$k_{y, inc} = k_0 \sqrt{\mu \varepsilon} \sin \theta_{inc}$$

Thus, we can simulate an electric field by solving for $E_z$ in the following equation:

$$ \frac{\partial}{\partial x'} \mu_y^{-1} \frac{\partial}{\partial x'} E_z + \frac{\partial}{\partial y'} \mu_x^{-1} \frac{\partial}{\partial y'} E_z + \varepsilon_z E_z = b $$

### Implementation of FDFD

For more information about the implementation of a Finite-Difference Frequency-Domain solver, please refer to [Electromagnetic and Photonic Simluation for the Beginner](https://empossible.net/fdfdbook/) by Raymond C. Rumpf. In general, this section will only discuss three functions found in `electric_funcs.py`.

`yeeder_2d`: In order to solve for the field $E_z$, we need to define how we are calculating the differentials in the equation. In order to do this, we need to combine forward and backward differences based on a staggered grid scheme, called the Yee grid. In order to compute the differentials, we create matrices that reflect these forward and backward differences. If we are working with a MxN surface area of the device we are simulating, then the differential matrices are going to be MN x MN. Because of this, the matrices need to be created sparsely, and any computation done sparsely.

<p align="center">
  <img alt="3D Yee Grid" src="https://github.com/uncbiag/OCT-FDFD/blob/main/Readme_imgs/yee.png" width=50% heights=50%>
</p>

`addupml2d_sparse`: During simulation, waves will propagate until they reach the boundary and will bounce off of it, which makes it difficult to distinguish the waves that are reflected from the device versus what is reflected from the boundary. To overcome this limitation, an absorbing boundary known as the Perfectly Matched Layer (PML) can be added to the borders of the simulation space. In order to add this layer, we need to create an electric permittivity $\epsilon$ and magnetic permeability $\mu$ profiles that are twice the size of the simulation area. In general, we want the PML to be bigger than 10 pixels on every side of the simulation.

<p align="center">
  <img alt="PML of a device" src="https://github.com/uncbiag/OCT-FDFD/blob/main/Readme_imgs/upml.png" width=40% heights=40%>
</p>

`solve_electric_field`: Solving for $E_z$ requires that we efine for which wavelength/wavenumber we are simulating for. Given the nature of the approach, we can only simulate one wavenumber at a time. After creating an (X, Y) grid where the $\epsilon$ and $\mu$ are defined, we can create an initial source field as defined by the equations of $f_src$. Once we have this source field, we create a wave matrix composed of the differentials and $\epsilon$ and $\mu$ as computed after adding the PML. That is 
```
    A = DHX @ URyy_i @ DEX + DHY @ URxx_i @ DEY + ERzz
```
Note that we need a mask matrix $Q$ that separates the Scattered Field and the Total Field. Once we have these matrices, we can solve for the electric field
```
    # Calculate source vector
    b = (Q @ A - A @ Q) @ fsrc_flat

    # Solve for field
    f = linalg.spsolve(A, b)
    f = np.reshape(f, (Nx, Ny), 'F')
```

**Note:** All of the arrays have to be flattened and inserted along the diagonal of a sparse matrix. The returning electric field also needs to be reshaped to be the size of the modeling area (MxN)

### Putting it all together

Now that we know how to simulate an electric field that passes through a medium, how can we use this to create a signal as it would be received by an OCT device?

It is relevant to now define the difference between the Scattered Field and the Total Field in the simulation. The Total Field is where our device is, and where we can see how the light moves and interacts with the materials in it. On the other hand, the scattered field is the electric field that goes out of the simulation after light has traversed through the material. That is, the scattered field is what a device like OCT would capture. In the following figure, the scattered field is denoted as $E_ref$

<p align="center">
  <img alt="Scattered field and Transmitted field (Rumpf, 2022)" src="https://github.com/uncbiag/OCT-FDFD/blob/main/Readme_imgs/RumpfEref.png" width=40% heights=40%>
</p>

However, we are most interested is not the scattered field, but the reflectivity of the scattered field. In order to obtain this measurement, we devide the scattered field by the original source field.

$$ r_S = \frac{E_{ref}}{E_{i}} $$

In which case, the electric field from the sample arm of an OCT device becomes

$$ E_S = \frac{s(k, \omega)}{\sqrt{2}} r_S $$

where $s(k, \omega)$ is the square root of the power spectrum $S(k)$ as previously defined.

Now that we are able to simulate the source power spectrum, the reference electric field, and the sample electric field, we can easily compute the A-line using the Fourier transform.

### Example

Let's say we want to simulate a single A-line from an OCT device with a central wavelength of 800nm and a bandwidth of 50 nm. Its source power spectrum in terms of wavelength would look as follows:

<p align="center">
  <img alt="Power Spectrum of Source Field" src="https://github.com/uncbiag/OCT-FDFD/blob/main/Readme_imgs/ex_source_power_spectrum.png" width=45% heights=45%>
</p>

We want to know what is the A-line of a three layered medium, where the layers are distributed between 0 and 45 micrometers. Since we are only interested in a single A-Line, the lateral dimension of the electric permittivity profile is 1 pixel plus 20 pixels on each side to include the PML. The arrow denotes the direction of propagation of the source field.

<p align="center">
  <img alt="Electric permittivity" src="https://github.com/uncbiag/OCT-FDFD/blob/main/Readme_imgs/ex_layered_medium.png" width=60% heights=60%>
</p>

Now that we have this electric permittivity profile, we can add the perfectly matched layer with `addupml2d_sparse`.

Let's say we have 100 wavenumbers (k) in our power spectrum. For every single wavenumber, we will create derivative matrices using `yeeder_2d`. For every single one of these wavenumbers, we will solve for the electric profile after computing our matrix `A`. This is the result of one of such simulations:

<p align="center">
  <img alt="Total Field of simulation" src="https://github.com/uncbiag/OCT-FDFD/blob/main/Readme_imgs/ex_electric_field.png" width=40% heights=40%>
</p>

This image shows a plane wave going up and down along the layered medium. Note that at the top and bottom of the total field, the values of the wave go to zero, which means that our PML is working. Similarly, note that the waves are not disrupted by the left and right boundaries.

After getting all of these fields, we save the reflectivity from the scattered field of our simulations. The following is the visualization of the reflectivity from the sample arm as a function of wavenumber $r_S(k)$

<p align="center">
  <img alt="Reflectivity from the Sample Arm as a function of Wavenumber" src="https://github.com/uncbiag/OCT-FDFD/blob/main/Readme_imgs/ex_reflected_fields.png" width=60% heights=60%>
</p>

Since we have the reflected field $r_S$ from the sample arm, and we can easily model the electric field from the reference arm and the sample arms via our equations for $E_R$ and $E_S$. Thus the intensity at the interferometer of the OCT will be 

<p align="center">
  <img alt="Intensity at the Interferogram" src="https://github.com/uncbiag/OCT-FDFD/blob/main/Readme_imgs/ex_intensity.png" width=60% heights=60%>
</p>

Once we have the intensity at the inteferometer, we want to subtract the amplitude of the reflected field to minimize the dominant term in our Inverse Fourier Transform.

<p align="center">
  <img alt="Intensity at the Interferogram with background subtracted" src="https://github.com/uncbiag/OCT-FDFD/blob/main/Readme_imgs/ex_intensity_no_background.png" width=60% heights=60%>
</p>

After obtaining this intensity, we need to standardize it to apply the IFFT. In order to do this, we add the "negative" frequencies of our simulation (which are all zero), and set our DC term equal to the average amplitude of our frequencies. After applying the IFFT and correcting the order of our array (for more information, read the documentation of numpy.fft.ifft), we obtain the A-Line of a three layered medium

<p align="center">
  <img alt="Intensity at the Interferogram with background subtracted" src="https://github.com/uncbiag/OCT-FDFD/blob/main/Readme_imgs/ex_aline.png" width=60% heights=60%>
</p>

Note that the peaks roughly correspond to where the medium changed its electric permittivity values. The reason why the peaks do not align perfectly is because the Optical Path Distance does not always correspond to the Geometric Distance in a medium. Similarly, the A-line is symmetric due to the properties of IFFT. 

**Note:** the images for this example were exclusively generated for this writeup. The code `compute_electric_simulation.py` saves the unprocessed A-line (including complex values), the intensity of the A-line (as showed in the last image), the Electric Reflectivity of the medium, and the reflectivity of the medium.
