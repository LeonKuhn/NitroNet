Welcome to the Github page of NitroNet, the first NO$_2$ profile retrieval for TROPOMI!



#### What is NitroNet?

NitroNet is an artificial neural network, trained on simulation data produced by the regional chemistry-transport model WRF-Chem. It takes tropospheric NO2 vertical column densities (VCDs) from TROPOMI alongside other ancillary data as input in order to predict a corresponding NO2 profile.

For a short description (~ 20 pages), see:

Kuhn, L., Beirle, S., Osipov, S., Pozzer, A., and Wagner, T.: *NitroNet – a machine learning model for the prediction of tropospheric NO2 profiles from TROPOMI observations*, Atmos. Meas. Tech., 17, 6485–6516, https://doi.org/10.5194/amt-17-6485-2024, 2024.

For a detailed description (~ 200 pages), see:

Kuhn, L.: *NitroNet – A deep-learning NO2 profile retrieval for the TROPOMI satellite instrument*, Dissertation, Universität Heidelberg, https://doi.org/10.11588/heidok.00036022, 2024.



#### How can NitroNet be used?

NitroNet is written in the Python programming language with only few dependencies. Its functionality comprises:

- downloading the required input data
- applying the neural network to TROPOMI data on a per-orbit level (with optional parallelization)
- rudimentary evaluation against TROPOMI and surface observations

> [!IMPORTANT]
>
> NitroNet is not a Python package, but rather a collection of scripts. It can only be obtained via this Github page.



#### License

This project is licensed under a Custom Non-Commercial License.  
You may use, modify, and distribute the code for **non-commercial purposes only**.

Use in academic or scientific research requires **prior written permission** from the author.

For details, see the [LICENSE](./LICENSE) file.  
For permission requests, please contact: l.kuhn@mpic.de



#### Current state of development, technical support

NitroNet has been developed over a span of three years. With the public release of version 1.0 NitroNet has left the stage of active development. The addition of new features is not foreseen but technically possible. Critical bugfixes might still be patched. 

For specific request, please contact: l.kuhn@mpic.de
**Even without regular updates on NitroNet's GitHub page we still aim to reply to inquiries in a timely manner.**



## 1 Installation

### 1.1 Preparatory steps

NitroNet requires the following input data:

- TROPOMI L2 data, specifically 
  - the tropospheric NO2 vertical column density (VCD)
  - the total O3 vertical column density
- The ERA5 variables
  - (single-level): ```boundary_layer_height```, ```2m_temperature```, ```boundary_layer_dissipation```
  - (pressure-level): ```vertical_velocity```, ```u_component_of_wind```, ```v_component_of_wind``` at pressure levels of 1000, 950, 900, 850, 800, 750, and 700 hPa

​	at a resolution of 0.25° $\times$ 0.25°.

In order to ensure users have easy access to these data in the correct file format, NitroNet can fetch them using premade download scripts. If the user wants to use these scripts they must first:

- for ```fetch_TROPOMI.py```:
  create a user account in the Earthdata portal, see the information given here:
  https://tropomi.gesdisc.eosdis.nasa.gov/data/S5P_TROPOMI_Level2
  In particular, the 'NASA GESDISC DATA ARCHIVE' application must be authorized in your Earthdata settings.

- for ```fetch_ERA5.py```:
  setup a cdsapi account, see https://cds.climate.copernicus.eu/how-to-api.
  In particular, the file .cdsapirc must be placed in the user's home folder, and should consiste of two lines:
  
  ```
  url: https://cds.climate.copernicus.eu/api
  key: <PERSONAL-ACCESS-TOKEN>
  ```
  
  where ```<PERSONAL-ACCESS-TOKEN>``` must be replaced with the user's personal access token obtained when registering at the CDS API.

### 1.2 Installation

Fetch the NitroNet code from Github
```
cd ~
git clone https://github.com/LeonKuhn/NitroNet.git
cd NitroNet
```

Create a fresh conda environment by executing

```
conda create -n NitroNet python=3.12 cdsapi numpy matplotlib seaborn datashader netCDF4 xarray dask scipy pandas pytorch cartopy tomli scikit-learn dill

conda activate NitroNet
```

This creates a new environment named NitroNet. Alternatively, the user can use an environment set up of their chosing, as long as it includes the packages listed in ```env.yml```.

Next, download and unpack the NitroNet's weights and the EDGARv5 emission data, which is currently hosted on the Datashare service of the Max Planck Computing & Data Facility

> [!IMPORTANT]
>
> the ``trained_models.zip`` file containing the model weights is currently **password protected**. Access is granted upon personal contact via l.kuhn@mpic.de
>
> For details, see the [LICENSE](./LICENSE) file.

```
wget -O trained_models.zip https://datashare.mpcdf.mpg.de/s/9lSmdRIAUrsKuVx/download
unzip trained_models.zip

mkdir data
cd data
wget -O edgarv5.zip https://datashare.mpcdf.mpg.de/s/yjGoZ9GH90Nk63O/download
unzip edgarv5.zip -d edgarv5
```

This concludes the installation of NitroNet. All steps up to this point must only be performed once.

## 2 Usage (minimal working example)

New users should reproduce the following minimal working example to get accommodated with NitroNet. Here we process two summer days (May 21 2022 and May 22 2022) on a domain covering Germany (latitude = 45° - 55°, longitude = 5° - 15°). The vertical output grid reaches from 0 - 3 km in steps of 100 m.

> [!NOTE]
>
> This example uses no parallelization and no GPUs. An advanced example including these options is given in section 3.

### 2.1 Downloading required input data

Enter the NitroNet directory and activate the NitroNet environment

```
cd ~/NitroNet
conda activate NitroNet
```

then execute the ERA5 downloading script
```
python fetch_ERA5.py 2022-05-21/2022-05-22 45-5-55-15 ./data/ERA5/test
```

where

- the first argument (```2022-05-21/2022-05-22```) specifies the date range to be processed
- the second argument (```45-5-55-15```) specifies the bounding box of the domain, expressed as ```south-west-north-east```
- the third argument (```./data/ERA5/test```) specifies the download location

On success, the folder ```~/NitroNet/data/ERA5/test/```should exist and contain:

- three ```ERA5_pressure-levels_*_to_*.nc``` files for the pressure-level variables
- three ```ERA5_single-levels_*_to_*.zip``` files for the single-level variables
  

> [!NOTE]
>
> The script will always fetch one additional day of data (here: data of 23 May 2022), which is required in some time zones of the world.

> [!NOTE]
>
> For now, all data is downloaded to the NitroNet folder itself, because the minimal working example does not require much data overall.


Next, execute

```
python split_ERA5_files.py ./data/ERA5/test
```

which merges the downloaded ERA5 data and stores them in one file per day. On success, the folder ```~/NitroNet/data/ERA5/test``` should contain three files named ```ERA5_*.nc```. The files ```data_stream-oper_stepType-accum.nc```and ```data_stream-oper_stepType-instant.nc```are intermediate files that can be ignored or deleted.

Lastly, execute the TROPOMI download script

```
python fetch_TROPOMI.py 2022-05-21/2022-05-22 O3_TOT/NO2 RPRO/RPRO ./data/TROPOMI
```

where

- the first argument (```2022-05-21/2022-05-22```) specifies the date range to be processed
- the second argument (```O3_TOT/NO2```) specifies which data products to fetch, separated by ```/```
- the third argument (```RPRO/RPRO```) specifies the timeliness of the preceding data products, separated by ```/```
- the fourth argument (```./data/TROPOMI```) specifies the download location
  

> [!NOTE]
>
> NitroNet can handle all timeliness options, but ```RPRO``` should be preferred over ```OFFL``` if available.


On success, the folder ```~/NitroNet/data/TROPOMI``` should exist and be structured as follows:

```
~/NitroNet/data/TROPOMI
│
└───L2
│   │   
│   └───NO2
│   │    └───RPRO
│   │       └───2022
│	  │							└───05
│		│									└───21
│		│									│		│ 	S5P_RPRO_L2__NO2_*.nc
│		│									└───22
│		│											│ 	S5P_RPRO_L2__NO2_*.nc
│   │
│   └───O3_TOT
│   │    └───RPRO
│   │       └───2022
│	  │							└───05
│		│									└───21
│		│									│		│ 	S5P_RPRO_L2__O3_TOT_*.nc
│		│									└───22
│		│											│ 	S5P_RPRO_L2__O3_TOT_*.nc
```



### 2.2 Running ```preproc.py```

The script ```preproc.py``` interpolates and merges the original input data onto a mutual grid with the same dimensions of the TROPOMI data (scanline, ground_pixel, etc.), with an additional "pressure_level" dimension for the ERA5 data.

From the NitroNet directory, execute

```
python preproc.py configs/test.toml 1
```

where

- the first argument (```configs/test.toml```) specifies NitroNet's configuration file (see sect. 3)
- the second argument (```1```) specifies the "job ID" (see below)

Ensure successful completion of the script by making sure the folder ```~/NitroNet/preproc/test``` exists and contains the files
```
~/NitroNet/preproc/test/2022/05/21/nn_input_23844.nc
~/NitroNet/preproc/test/2022/05/21/nn_input_23845.nc
~/NitroNet/preproc/test/2022/05/21/nn_input_23850.nc
~/NitroNet/preproc/test/2022/05/21/nn_input_23851.nc

~/NitroNet/preproc/test/2022/05/22/nn_input_23858.nc      
~/NitroNet/preproc/test/2022/05/22/nn_input_23859.nc
~/NitroNet/preproc/test/2022/05/22/nn_input_23864.nc
~/NitroNet/preproc/test/2022/05/22/nn_input_23865.nc
```

where the 5-digit number in the filenames indicate the orbit number.

Additionally, some self-explanatory debug plots can be found in ```NitroNet/preproc_debug_plots/```, i.e. to ensure that no errors have occured during the interpolation.

> [!IMPORTANT]
>
> In non-parallel execution the job ID argument must always be set to 1. When parallelized via slurm, the job ID argument instructs NitroNet how to distribute the workload across cores.



### 2.3 Running NitroNet (```main.py```)

In the NitroNet folder, execute the main NitroNet script

```
python main.py configs/test.toml 1
```

The arguments to ```main.py``` have the same meaning as for ```preprocessing.py```. 

Ensure successful completion of the script by making sure the folder ```~/NitroNet/output/test``` exists and contains the files

```
~/NitroNet/output/test/2022/05/21/23844.nc
~/NitroNet/output/test/2022/05/21/23845.nc
~/NitroNet/output/test/2022/05/21/23850.nc
~/NitroNet/output/test/2022/05/21/23851.nc

~/NitroNet/output/test/2022/05/22/23858.nc      
~/NitroNet/output/test/2022/05/22/23859.nc
~/NitroNet/output/test/2022/05/22/23864.nc
~/NitroNet/output/test/2022/05/22/23865.nc
```

where the 5-digit number in the filenames indicate the orbit number.

## 3 Overview of NitroNet's output variables

NitroNet's output files contain the following main variables

- ```no2_conc```  The NO2 concentration profile in units of molec / cm$^3$
- ```no2_conc_NN_err```  The relative prediction error of the NO2 concentration in molec / cm$^3$, based on the evaluation on the test set. E.g. a value of -5 % means that NitroNet underestimated the NO2 concentration by 5 % on average on the test set.
- ```qa```  The same as the qa-value of TROPOMI, but multiplied with the sign of the TROPOMI NO2 VCD. Taking the absolute value of ```qa``` returns the original qa-value, as present in the TROPOMI data
- ```F-value```  A unitless correction factor for the cross-sensitivities of molybdenum-based chemiluminescence measurements. This variable is only available at the surface.

> [!NOTE]
>
> If ```"uncerts" = true```  is selected in the configuration file, the output will contain two further variables named ```no2_conc_err_lower```and ```no2_conc_err_upper```, which resemble the $\pm 1 \sigma$ uncertainty band of the predicted NO2 concentration based on Monte-Carlo error propagation of TROPOMI's NO2 VCD retrieval error.



## 4 Controlling NitroNet through configuration files

### 4.1 Configuration files

The behaviour of NitroNet can be controlled via configuration files written in TOML (see https://toml.io/en/). An example configuration file for the minimal use case in section 2 is provided under ```~/NitroNet/configs/test.toml```

The following table describes the available options in NitroNet's configuration files.

| key                 |                                                              | comment                                                      |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **[general]**       |                                                              |                                                              |
|                     | ```NO2_tropomi_dir```                                        | Location of the TROPOMI NO2 L2 data (trop. vertical column)  |
|                     | ```O3_tropomi_dir```                                         | Location of the TROPOMI O3 L2 data (total vertical column)   |
|                     | ```EDGARv5_dir```                                            | Location of the EDGARv5 emission data                        |
|                     | ```ERA5_dir```                                               | Location of the ERA5 re-analysis data                        |
|                     | ```log_dir```                                                | Location of the slurm log files<br />ℹ️ this option only affects parallelized NitroNet runs using slurm<br />ℹ️ In non-parallelized runs NitroNet creates no log files, because all relevant information is given in the stdout and stderr streams<br />ℹ️ This option is shared across ```preproc.py``` and ```main.py``` |
|                     | ```lat_boundaries```                                         | Tuple in the format [south, north]                           |
|                     | ```lon_boundaries```                                         | Tuple in the format [west, east]                             |
|                     | ```start_date```                                             | Start date in the format 'YYYY-MM-DD'                        |
|                     | ```end_date```                                               | End date in the format 'YYYY-MM-DD' (inclusive)              |
| **[preproc]**       |                                                              | The options in this category control the behavior of ```preproc.py``` |
|                     | ```output_dir```                                             | Output directory                                             |
|                     | ```debug_plot_dir```                                         | Output directory of the debug plots<br />ℹ️ Can also be set to *false* in order to deactivate the creation of debug plots entirely. |
| **[preproc.slurm]** |                                                              | The options in this category control the behavior of ```preproc.py``` when using slurm. See https://slurm.schedmd.com/pdfs/summary.pdf. |
|                     | ```--time```                                                 | Runtime in the format "HH:MM:SS"                             |
|                     | ```--array```                                                | Numer of job array elements (integer ≥ 1)                    |
|                     | ```--cpus_per_task```                                        | Number of cores per task (integer ≥ 1)<br />ℹ️ --array $\times$ --cpus_per_task is the total number of cores used by NitroNet |
|                     | ```--mem```                                                  | Requested amount of memory in MB.                            |
|                     | ```--gres=gpu```                                             | The amount of GPUs to request via slurm.<br />⚠️ ```preproc.py``` is a CPU-only script that **never** uses the GPU. Nonetheless, depending on cluster configuration, a request for memory via the ``--mem`` option might require a certain number of GPUs be requested. |
| **[main]**          |                                                              | The options in this category control the behavior of ```main.py``` |
|                     | ```output_dir```                                             | Output directory                                             |
|                     | ```use_cuda```                                               | *true*: NitroNet will attempt to execute the neural network forward pass on the GPU.<br />*false*: the CPU will be used instead. |
|                     | ```batchsize```                                              | Number of instances to process per neural network forward pass (integer ≥ 1). Larger batchsizes result in shorter runtimes but require more memory. |
|                     | ```random_seed```                                            | Seed of the random number generator involved in the winsorization procedure.<br />ℹ️ Specific choices do not significantly affect results, but ensure reproducability between runs |
|                     | ```z_space```                                                | Vertical output grid in meters above ground                  |
|                     | ```uncerts```                                                | *true*: NitroNet will perform Monte-Carlo based uncertainty propagation of the NO2 VCD through the neural network. Slows down execution by a factor of 3. |
|                     | ```winsor_threshold```                                       | Density threshold below which an instance feature will be replaced by a sample from its marginal probability distribution in the winsorization procedure (float between 0 and 1). Larger values reduce generalization errors (e.g. on foreign domains) but result in partial loss of input information. |
|                     | ```winsor_blacklist```                                       | Input variables to exclude from the winsorization procedure.<br />⚠️ It is recommended to include at least "NO2_tropcol_sat", "day", "SC_urban", "SC_cropland", "SC_forest", "emi_all", and "emi_surface".<br />ℹ️ For a full list of input variable names, see section 4.4 |
|                     | ```sample_method```                                          | Controls the behavior of the winsorization procedure<br />.<br />"replace": replace low-density input data by samples from their marginal density distribution<br /><br />"skip": skips the winsorization procedure entirely<br /><br />"ignore": same as "skip", but still computes computes feature densities in order to obtain winsorization statistics (see "print_winsorization_info" below)<br /><br /><br />"remove": remove low-density input data entirely |
|                     | ```print_winsorization_info```                               | *true*: NitroNet will print winsorization statistics, e.g. number of replaced instances. |
|                     | ```use_bias_correction```                                    | *true*: NitroNet will apply the empirical bias correction obtained from evaluation on the test set. |
|                     | ```reject_partial_nans```                                    | *true*: NO2 profiles are set to np.nan entirely if they include a single np.nan along the vertical axis.<br />ℹ️ partial np.nans along the vertical axis can result in the context of composite models (see section 4.2). |
| **[main.slurm]**    |                                                              |                                                              |
|                     | ```--time```, ```--array```, ```--cpus_per_task```, ```--mem``` | see above                                                    |
|                     | ```--gres=gpu```                                             | The amount of GPUs to request via slurm. The amount of GPUs to request via slurm.<br />⚠️ Even if no GPU is used (```cude=false```) the user might wish to request GPUs in order to gain access to sufficient memory (see above). |

### 4.2 Composite models

NitroNet uses a "composite model" approach. This means that internally, different neural networks make individual predictions for different vertical ranges, and NitroNet then concatenates these partial profiles to form full profiles.

This has two main benefits of technical nature. Firstly, it saves memory during training. Secondly, it was empirically observed to produce much lower training losses in the upper troposphere.

NitroNet has 2 neural networks; one for altitudes of $\sim$ 0-8 km above ground, and one for altitudes of $\sim$ 8-13 km. Additionally, it has an F-network whose sole purpose is to predict the variable ```F-value``` (see sect. 3) at the surface.

The behavior of the composite model can be specified in the configuration files, as explained in the following example:

```
[main.submodels]
    [main.submodels.bottom]
        dir = "./trained_models/2023-08-09_6530_bottom/"
        alt_min = 0
        alt_max = 8000

    [main.submodels.top]
        dir = "./trained_models/2023-11-14_862d_top/"
        alt_min = 8000
        alt_max = 100_000

[main.F-model]
        dir = "./trained_models/F_2023-09-29_1b4a/"
```

This setup uses the saved model ```2023-08-09_6530_bottom``` for predictions in the range of 0-8 km and ```2023-11-14_862d_top``` for predictions in the range of 8 km and above (```alt_max = 100_000``` is an arbitrary altitude beyond the tropopause; NitroNet is only trained on tropospheric data and ```z_space``` should never exceed the troposphere.)

The list of neural networks (submodels) can be arbitrarily enhanced or reduced. The model names (e.g. "bottom" in ```main.submodels.bottom```) can be chosen freely. 

> [!NOTE]
>
> NitroNet currently ships with the two neural networks given above, hence there is no reasonable alternative setup. The same applies to the F-model. However, future updates might include improved neural networks, which can then be easily included.
>
> Still, the user may change the values ```alt_min``` and ```alt_max``` if undesired model behavior is observed (e.g. poor generalization on foreign domains).



### 4.3 Input variables

The ```winsor_blacklist``` option requires a list of NitroNet input variables, which may include any of the following:

| **variable name**             | **meaning**                                                  | source  |
| ----------------------------- | ------------------------------------------------------------ | ------- |
| ```trop_AMF```                | tropospheric air mass factor                                 | TROPOMI |
| ```trop_AK_[i]```             | tropospheric averaging kernels ([i] must be replaced by any number from 0 to 8) | TROPOMI |
| ```crf```                     | cloud radiance fraction                                      | TROPOMI |
| ```surface_albedo```          | surface albedo                                               | TROPOMI |
| ```sza```                     | solar zenith angle                                           | TROPOMI |
| ```saa```                     | solar azimuth angle                                          | TROPOMI |
| ```za_sat```                  | viewing zenith angle                                         | TROPOMI |
| ```aa_sat```                  | viewing azimuth angle                                        | TROPOMI |
| ```cloud_pressure```          | cloud pressure                                               | TROPOMI |
| ```aerosol_index```           | aerosol index                                                | TROPOMI |
| ```surface_pressure```        | surface pressure                                             | TROPOMI |
| ```NO2_tropcol_sat```         | tropospheric NO2 vertial column density                   | TROPOMI |
| ```O3_tropcol_sat```          | total O3 vertial column density                           | TROPOMI |
| ```blh_ERA5```                | boundary layer height                                        | ERA5    |
| ```bld_ERA5```                | boundary layer dissipation                                   | ERA5    |
| ```T2m_ERA5```                | temperature at 2 m                                           | ERA5    |
| ```vertical_speed_ERA5_[i]``` | vertical velocity ([i] must be replaced by any number from 0 to 6) | ERA5    |
| ```ERA5_total_wind_[i]```     | horizontal wind speed ([i] must be replaced by any number from 0 to 6) | ERA5    |
| ```emi_all```                 | NO$_x$ emissions                                             | EDGARv5 |
| ```emi_surface```             | NO$_x$ emissions (surface emissions)                         | EDGARv5 |
| ```emi_energy_1```            | NO$_x$ emissions (SNAP sector 1)                             | EDGARv5 |
| ```emi_industry_3```          | NO$_x$ emissions (SNAP sector 3)                             | EDGARv5 |
| ```emi_industry_4```          | NO$_x$ emissions (SNAP sector 4)                             | EDGARv5 |
| ```SC_urban```                | binary urban mask                                            | TROPOMI |
| ```SC_cropland```             | binary cropland mask                                         | TROPOMI |
| ```SC_forest```               | binary forest mask                                           | TROPOMI |
| ```day```                     | weekday/weekend flag                                         | ---     |
| ```influx```                  | NO2 VCD influx from neighbouring groundpixels             | TROPOMI |

> [!Note]
>
> It is recommended to include at least "NO2_tropcol_sat", "day", "SC_urban", "SC_cropland", "SC_forest", "emi_all", and "emi_surface".



## 5 NitroNet with parallelization and GPU usage

Processing data on larger tempo-spatial domains can result in exceeding computational cost. NitroNet supports parallelization and the usage of GPUs, which accelerates the model considerably.

Here we process 14 summer days (May 21 2021 - June 4 2021) on a domain covering Germany (latitude = 45° - 55°, longitude = 5° - 15°). The vertical output grid reaches from the surface to the tropopause with irregular stepsize, see ```configs/test_parallel.toml```.

> [!IMPORTANT]
>
> The parallelization of NitroNet is currently only available through slurm, see https://slurm.schedmd.com. 

> [!WARNING]
>
> The resource specifications in the exemplary configuration file ```configs/test_parallel.toml``` were tested on the MPCDF machine Viper-GPU and may be invalid on other machines.

> [!IMPORTANT]
>
> The following code requires a larger amount (~ 300 GB) of input data.
>
> ```configs/test_parallel.toml``` defaults to processing input/output from the ```~/NitroNet/data``` folder. The user may want to change the IO directory by typing
>
> ```
> NITRONET_DATADIR=xyz
> ```
>
> to the file ```~/.bashrc```, where ```xyz``` must be replaced with the new location. ```xyz``` should **not** include a trailing ```/```. Then, execute
>
> ```
> source ~.bashrc 
> sed -i -e "s,\./data,$NITRONET_DATADIR,g" configs/test_parallel.toml
> sed -i -e "s,\./data,$NITRONET_DATADIR,g" configs/test_parallel_gpu.toml
> cp -pr data/edgarv5 "${NITRONET_DATADIR}/."
> ```
>
> Ensure that the I/O directories in ```configs/test_parallel.toml``` (e.g. ```NO2_tropomi_dir```) are adequate. Otherwise the configuration files can always be edited by hand.

### 5.1 Download the required input data

Execute

```
python fetch_ERA5.py 2021-05-21/2021-06-04 45-5-55-15 "${NITRONET_DATADIR}/ERA5/test_parallel"
python split_ERA5_files.py "${NITRONET_DATADIR}/ERA5/test_parallel"
python fetch_TROPOMI.py 2021-05-21/2021-06-04 O3_TOT/NO2 RPRO/RPRO "${NITRONET_DATADIR}/TROPOMI"
```

### 5.2 Parallelization of ```preproc.py```

The requested amount of computational resources is specified through the configuration files via the keys ```--array```, ```--cpus_per_task```, ``--mem``, and ```--gres=gpu```.

For ```preproc.py``` the parallelization factor is computed as ```--cpus_per_task * --array```. A typical configuration could be ```--array = 1```, ```--cpus_per_task=72```, resulting in a parallelization factor of 72.

An exemplary parallel run of ```preprocessing.py``` can be launched by executing

```
python preproc_to_slurm.py NitroNet configs/test_parallel.toml
```

> [!NOTE]
>
> The warning ```RuntimeWarning: invalid value encountered in arccos``` is raised if NaN values occur in the computation of the ```influx``` variable and can be ignored here.

> [!NOTE]
>
> The warning ```Critical warning: Found 0 possible O3 files, expected exactly 1. This orbit will be skipped``` occurs if no suitable O3 orbit files were found. This is sometimes the case and can be ignored here.

where

- the first argument (```NitroNet```) specifies the conda environment to be used
- the second argument (```configs/test_parallel.toml```) specifies the configuration file to be used

Upon submission, the user should receive

```Submitting:
Submitting:
 sbatch preproc_[uuid].sh
Submitted batch job [ID]
```

and a new slurm job with the specified ID be added to the queue. The slurm submission script itself will be deleted automatically. The corresponding log and error scripts can be found under ```/logs/test_parallel/```. The user should ensure that the NitroNet input files were generated correctly (see section 2.2).

### 5.3 Parallelization of ```main.py```

> [!IMPORTANT]
>
> ```main.py``` cannot use Python's multiprocessing due to a compatibility issue with the required pickle/dill libraries. Therefore ```main.py``` should be run with ```--cpus_per_task=1```. Parallelization can be achieved via the ```--array``` key.

##### Without using GPUs

Execute

```
python main_to_slurm.py NitroNet configs/test_parallel.toml
```

which executes ```main.py``` with a parallelization factor of 10 (because ```--array=10```). For the meaning of the arguments, and the expected output, see the explanation for ```preproc.py``` above.

##### With GPUs

Running NitroNet on GPUs can reduce runtimes considerably, but requires machine-specific preparations, including:

1. Installation of either **pytorch-cuda** or **pytorch-rocm**, depending on whether the machine uses GPUs from Nvidia (cuda) or AMD (rocm). Either package appears to be no longer available through ```conda```, but through ```pip```.
2. The corresponding modules (**cuda** or **rocm**) must be loaded on the machine, and the version must be identical to that with which the python packages (**pytorch-cuda** or **pytorch-rocm**) were built.

The following example runs NitroNet with GPUs on the Viper-GPU machine.

First, create a corresponding conda environment (here we clone the existing NitroNet evironment named NitroNet-GPU and modify it)

```
conda create -n NitroNet-GPU --clone NitroNet
conda activate NitroNet-GPU
conda uninstall pytorch
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
```

If you wish to install PyTorch built with cuda, the last line must be adjusted accordingly.

Second, load the required GPU modules on the machine, here by executing

```
module load rocm/6.3
```

> [!IMPORTANT]
>
> The version of the GPU module loaded (here: 6.3) must be identical to that with which the pytorch package instaleld with pip has been built.

Afterwards, execute

```
python main_to_slurm.py NitroNet-GPU configs/test_parallel_gpu.toml
```

which executes ```main.py``` with a parallelization factor of 10 (because ```--array=10```), using one GPU for each job array element (i.e. 10 GPUs in parallel).

> [!TIP]
>
> Ensure that the ```*.out``` logfiles in the ```/logs/test_parallel/``` directory include the line ```This NitroNet run uses device 'cuda'```
