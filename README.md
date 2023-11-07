# OCT-FDFD
Simulates OCT images by solving Maxwell's equations through a Finite-Difference Frequency-Domain method

## Use

### Modifying the dashboard

The parameters needed to simulate the OCT source can be found in `dashboard.py`. It is recommended that you do not modify the units and the constants. If you wish to modify the OCT source, keep in mind that the bandwidth affects the Nyquist sampling frequency. For OCT, the broader the bandwidth, the more resolution. However, the broader the bandwidth, the more samples you have to take so that the sampling frequency is higher than the Nyquist frequency. Refer to the variables `nyquist_spacing` and `delta_k` in `compute_electric_simulation.py` to see if `delta_k > nyquist_spacing`.  

To modify the physical size of the device, change variables `Sy` and `Sx`. The variable `PML` contains the dimensions --in pixels-- of the Perfectly Matched Layer so that $PML=[PML_{XLO}, PML_{XHI}, PML_{YLO}, PML_{YHI}]$. The variable `NRES` defines the number of pixels that are used to define the smallest wavelength in the simulation.

The material properties in the layered medium can be modified to determine the maximum possible electric permittivity and where the layers should appear in the device. Note that increasing the maximum electric permittivity will affect the Optical Path Distance in the A-Line.

The code `compute_electric_simulation.py` can be run on SLURM using the command:
<p align="center">
  `sbatch slurm_electric_field_simulations.sl
</p>


