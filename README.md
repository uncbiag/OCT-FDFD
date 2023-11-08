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


