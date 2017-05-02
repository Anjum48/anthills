import os
import pandas as pd
import numpy as np
import lasio

well = "R-550"
las_file = os.path.join(os.pardir, "Input_files", well+"_CMR.las")
output_folder = os.pardir
output_folder_name = "anthills_output"

if not os.path.exists(os.path.join(output_folder, output_folder_name, "Permeability")):
    os.makedirs(os.path.join(output_folder, output_folder_name, "Permeability"))

means = pd.read_csv(os.path.join(os.pardir, output_folder_name, "CSV", well+"_means.csv"), index_col=0)
weights = pd.read_csv(os.path.join(os.pardir, output_folder_name, "CSV", well+"_weights.csv"), index_col=0)
phi_nmr = lasio.read(las_file).df().dropna()["TCMR"]
phi_nmr = phi_nmr.values.reshape((len(phi_nmr), 1))

if len(means) == len(weights) == len(phi_nmr):
    perm_dict = {}
    for cutoff in [0.3, 50, 100]:
        means_filtered = means[(means > np.log10(cutoff)) & (weights > 0.01)].fillna(0)
        weights_filtered = weights[(means > np.log10(cutoff)) & (weights > 0.01)].fillna(0)
        weights_filtered = weights_filtered.div(np.sum(weights_filtered, axis=1), axis=0).fillna(0)
        perm_dict["Perm_GMM_"+str(cutoff)+"ms"] = (((phi_nmr*weights_filtered)**4) * ((10**means_filtered)**2)).sum(axis=1)
    pd.DataFrame(perm_dict).to_csv(os.path.join(output_folder, output_folder_name, "Permeability", well+"_NMRGMM.csv"))

else:
    print("Mismatch between the number of PHIT samples and processed Gaussians:", len(phi_nmr), "vs", len(weights))
