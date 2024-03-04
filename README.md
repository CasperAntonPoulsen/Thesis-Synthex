# Thesis Synthex

## Usage

### Carbontracker

All scripts submitted to a cluster in this repository uses the carbontracker[ https://github.com/lfwa/carbontracker]()

The logs are stored in /carbon_logs/ and to summarize the logs, simply run;

`$ python3 carbon_summary.py`

### NBIA downloader

This repository uses data from the Cancer Imaging Archive. To download data from the archieve you would usually have to use the NBIA Data Retriever. In this repo we have used the TCIA wrapper for the NBIA api to create a data retriever that does not require the desktop app and can be submitted to cluster schedulars like LSF10 and SLURM

The nbia_downloader.py takes two command line arguments --output-dir and --collection-name

Example run:

`$ python3 /home/caap/Thesis-Synthex/nbia_downloader.py --output-dir /home/data_shares/purrlab_students/LIDC-IDRI --collection-name LIDC-IDRA`

## Weekly log

### Weeks 8-9

For these two weeks we have had focus on 3 things, finding related works for DRR and projecting CT scans to 2D for more data, Defining our experimental setup and implementing the creation of the synthetic data using DeepDRR.

#### Experimental setup (Qualitative study)

When working with synthetic images a concern is the images degree of realism. Are the synthetic images easy to
differ from real images? To check the quality of the produced synthetic images we want present a series of real and synthetic images to multiple radiologist with varying years of expertise. The dataset consist of 50 to 100 different chest
x-ray images with a mixture of synthetic and real images. The dataset will consist of synthetic images created from the covid-19 ct scan dataset,images created from (ImagEng) both following the SyntheX method and real images from (QaTa-COV19).

All images were shuffled and no image information was given. When scoring the images we followed the method used in Ali, Murad, and Shah (Spot the fake article). The radiologist were asked to state if the given image was real, fake or they were uncertain. When the radiologist rated an image as fake or uncertain a follow up box appered where they could explain why they rated as such.

After the radiologist have completed looking at all the images, a follow up interview is conducted so they can explain their findings especially what made them rate an image fake.

Synthetic image:

![1709537604085](image/README/synthetic_example.png)

Real image:

![1709537643196](image/README/real_example.png)

## Todo list

### Current

### Backlog
