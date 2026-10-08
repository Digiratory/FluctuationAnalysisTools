# NORW010 tree-ring preprocessing for the Kalman filtering experiment

This folder contains scripts for the NORW010 tree-ring example used in the Kalman filtering experiment. The data are ring-width measurements of Scots pine from the Forfjorddalen site in northern Norway.

The dataset is available from the NOAA Paleoclimatology archive under the identifier NORW010 and is associated with the coastal northern Norway temperature reconstruction by Kirchhefer. It contains 71 ring-width measurement series from 36 Scots pine trees and spans 877-1994 CE.

All executable scripts are located in the scripts directory. Download_rwl.R downloads the NORW010 source files to data. preprocessingtrw.py performs the Python preprocessing and writes its output to results/python. Preprocess_rwl.R performs the R/dplR preprocessing and writes its output to results/dplr.

The examples directory contains a Jupyter notebook version of the Python preprocessing script.

# What the scripts do

The Python script performs the preprocessing used before the proposed Kalman filtering procedure. Individual raw series are aligned by age, a mean age-growth curve is estimated across the available series and smoothed using a 51-year centered moving window, and each raw series is divided by the corresponding age-specific mean value.

The R script processes the same raw observations using the dplR package. The series are subjected to Cook's power transformation, age-dependent spline detrending with a 50-year parameter, and signal-free standardization. The resulting RWI series are combined into a mean chronology using Tukey's biweight robust mean with prewhitening disabled.

# How to run

### using the RGui

**In RGui, commands are entered without `Rscript`. First set the working directory to the project root:**
setwd("C:/Project")

**Install `dplR` if it has not been installed yet:**
install.packages("dplR", repos = "https://cloud.r-project.org")

**Download the NORW010 source files into the `data` directory:**
source("scripts/Download_rwl.R", encoding = "UTF-8")

**Run the R/dplR preprocessing pipeline:**
source("scripts/Preprocess_rwl.R", encoding = "UTF-8")

### using the Jupyter

**The `examples` directory contains a notebook version of the Python preprocessing workflow:**
cd /d "C:\Project"

**Python packages:**
python -m pip install numpy pandas matplotlib

**If Jupyter is not installed, it can be installed with:**
python -m pip install notebook

**Start JupyterLab**
jupyter lab
