#!/usr/bin/env python
from __future__ import print_function,division
import os
import sys
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint

# Parser for command line arguments
parser = ArgumentParser()
parser.add_argument("-i", "--inF", default="NNLO_templates.h5", help="input HDF5 file with pandas dataframes")
parser.add_argument("-o", "--outDir", default="NNLO_template_plots", help="output plot directory")
args = parser.parse_args()

inF = args.inF
outDir = args.outDir

# Create output directory if it doesn't already exist
if not os.path.exists(outDir):
    os.makedirs(outDir)

# Open input file
print("Reading dataframes from", inF)
data = pd.HDFStore(inF, mode="r")

# Each observable is stored in a separate dataframe 
observables = data.keys()

# Remove leading '/' from observable names
observables = [obs.strip('/') for obs in observables]
print("Observables found:", observables)

# Determine binning for each observable
# The center of each bin is given by the column with the observable name (e.g. ptll)
# The column 'binWidth2' gives half the width of each bin. 

# To plot a histogram you need a list of the bin edges. These are the same for all 3 masses of a given observable. 
binning = {}
for obs in observables:
    obsdata = data[obs]
    nominal = obsdata[obsdata.mt == 172.5]  # Choose the nominal 172.5 mass

    # List of all the lower bin edges: center - width/2
    binning[obs] = nominal[obs] - nominal["binWidth2"]
    
    # Convert to numpy array
    binning[obs] = binning[obs].values

    # Append the upper edge of the last bin: center + width/2
    # iloc accesses a row of the dataframe by index, with -1 giving the last row 
    binning[obs] = np.append(binning[obs], nominal[obs].iloc[-1] + nominal["binWidth2"].iloc[-1])


# Binning along mass axis. The centers correspond to the values of mt: 171.5, 172,5, 173.5
mtbinning = np.array([171,172,173,174])

# mt values
masses = np.array([171.5,172.5,173.5])


####################
#  Plot templates  #
####################

# Dictionaries to save output from matplotlib histograms
hist = {}
xbins = {}
mesh = {}
hist2D = {}
hist2Dxbins = {}
hist2Dybins = {}
hist2Dmesh = {}

for obs in observables:
    # For clarity, define a variable pointing to the dataframe for this observable
    obsdata = data[obs]
    
    hist[obs] = {}
    xbins[obs] = {}
    mesh[obs] = {}

    # 1D histograms for each mass
    fig, ax = plt.subplots(tight_layout=True)
    for mt in masses:
        # Get arrays of bin centers and contents here for clarity
        binCenters  = obsdata[obsdata.mt == mt][obs]
        binContents = obsdata[obsdata.mt == mt]["binContent"]
        
        # Create 1D histogram
        hist[obs][mt],xbins[obs][mt],mesh[obs][mt] = ax.hist(binCenters,bins=binning[obs],weights=binContents,histtype="step")
   
    # Set legend labels
    ax.legend(["$m_{t}$ = %.1f" % mt for mt in masses])
    
    # Set title and axis labels
    ax.set_title("%s NNLO templates" % obs)
    ax.set_xlabel("%s [GeV]" % obs)
    ax.set_ylabel("Events")

    # Save output
    fig.savefig("%s/%s.png" % (outDir,obs))

    
    # 2D histogram of mt vs observable
    fig, ax = plt.subplots(tight_layout=True)
    hist2D[obs],hist2Dxbins[obs],hist2Dybins[obs],hist2Dmesh[obs] = ax.hist2d(obsdata[obs], obsdata["mt"], bins=(binning[obs],mtbinning), weights=obsdata["binContent"])

    # Set title and axis labels
    ax.set_title("%s NNLO templates" % obs)
    ax.set_xlabel("%s [GeV]" % obs)
    ax.set_ylabel("$m_{t}$ [GeV]")

    # Save output
    fig.savefig("%s/mt_vs_%s.png" % (outDir,obs))

    
print("Plots saved to %s\n" % outDir)

