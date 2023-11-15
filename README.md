# OCT-FDFD
Simulates OCT images by solving Maxwell's equations through a Finite-Difference Frequency-Domain method

## Use

### Modifying the dashboard

**OCT Source**

The parameters needed to simulate the OCT source can be found in `dashboard.py`. It is recommended that you do not modify the units and the constants. If you wish to modify the OCT source, keep in mind that the bandwidth affects the Nyquist sampling frequency. For OCT, the broader the bandwidth, the more resolution. However, the broader the bandwidth, the more samples you have to take so that the sampling frequency is higher than the Nyquist frequency. Refer to the variables `nyquist_spacing` and `delta_k` in `compute_electric_simulation.py` to see if `delta_k > nyquist_spacing`.  

**Properties of the Layered Medium**

To modify the physical size of the device, change variables `Sy` and `Sx`. The variable `PML` contains the dimensions --in pixels-- of the Perfectly Matched Layer so that $PML=[PML_{XLO}, PML_{XHI}, PML_{YLO}, PML_{YHI}]$. The variable `NRES` defines the number of pixels that are used to define the smallest wavelength in the simulation.

The material properties in the layered medium can be modified to determine the maximum possible electric permittivity and where the layers should appear in the device. Note that increasing the maximum electric permittivity will affect the Optical Path Distance in the A-Line.

### Running the code on SLURM

The code `compute_electric_simulation.py` can be run on SLURM (Longleaf) using the command:
```
sbatch slurm_electric_field_simulations.sl
```
In `dashboard.py` you can change the simulation ID via the variable `slurm_offset`. Using SLURM arrays, the id will start on the offset and be added to the `SLURM_ARRAY_TASK_ID` environment variable. For more information on using SLURM arrays, you can refer to [this article]( https://blog.ronin.cloud/slurm-job-arrays/ )

To move the files from the Longleaf cluster to your computer, you can use 
```
scp onyen@longleaf.unc.edu:PATH/TO/DATASET LOCAL/DIRECTORY
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

## Description of Algorithm

### OCT Intensity and A-Line Construction

Optical Coherence Tomography (OCT) is a non-invasive imaging technique that utilizes the principle of coherence to generate images along the depth of a tissue. An OCT device consists of a light source that emits a broad spectrum of light in near-infrared wavelengths, an interferometer that splits the light into two beams, a reference arm that reflects one of the beams back to the interferometer, and a sample arm that directs the other beam to the tissue being imaged.

When both fields return from the reference and sample arms respectively, they are halved in power again and interfere at the detector, which generates a photocurrent given by 

$$ I_D(k, \omega) = \frac{1}{2} |E_R + E_S|^2 $$

On the other hand, we can define the power spectral density of the source electric field $E_i$ as   

$$ S(k) = |s(k, \omega)|^2 = \frac{1}{\Delta k \sqrt{\pi}}e^{- \frac{(k - k_0)}{\Delta k}\^2}$$

Where $k_0$ is the central wavenumber of the light source spectrum and $\Delta k$ is the spectral bandwidth, which corresponds to the half-width-half-maximum of the light source power spectrum.

From the power spectral density, we can simulate the reference electric field in the following way

$$  E_R = \frac{E_i}{\sqrt{2}} r_R e^{i2kz_R} $$

Where $r_R$ is the electric field reflectivity of the reference reflector, $z_R$ is the distance from the beamsplitter to the reference arm, and the factor of $2$ corresponds to the distance that the light has to travel to and from the reference reflector. 

In order to construct the A-line scan, it is necessary to take the inverse Fourier transform of $I_D(k, \omega)$ and subtract the dominant reference spectrum. That is the inverse Fourier transform will be applied to 

$$  I_D(k) - |E_R|^2 $$

Since the amplitude of the incident field is defined in terms of the wavenumber $k$, the inverse fourier transform will be in terms of depth $z$. To obtain the increment in the $z$-domain, we use the formula $\delta_z = 2 \pi / 2 \text{N} \delta_k$, where N is the number of sample points that are used to model the spectrum amplitude $s(k, \omega)$ and the electric fields $E_S(k)$. 

Computing the A-Line is a very straightforward approach from the equations presented in this writeup, however, we still need to figure out how to simulate the electric field that comes from the sample arm.

### FDFD


