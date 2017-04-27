import os
import pandas as pd
import lasio

well = "R-538"
las_file = os.path.join(os.pardir, "Input_files", well+"_CMR.las")
output_folder = os.pardir
output_folder_name = "NMRGP_Output_Cored"

if not os.path.exists(os.path.join(output_folder, output_folder_name, "Permeability")):
    os.makedirs(os.path.join(output_folder, output_folder_name, "Permeability"))

means = pd.read_csv(os.path.join(os.pardir, "NMRGP_Output_Cored", "csv", well+"_means.csv"), index_col=0)
weights = pd.read_csv(os.path.join(os.pardir, "NMRGP_Output_Cored", "csv", well+"_weights.csv"), index_col=0)
phi_nmr = lasio.read(las_file).df().dropna()["TCMR"]
phi_nmr = phi_nmr.values.reshape((len(phi_nmr), 1))

if len(means) == len(weights) == len(phi_nmr):
    perm = (((phi_nmr*weights)**4) * ((10**means)**2)).sum(axis=1)
    perm.rename("NMRGP_Perm", inplace=True)
    perm.to_csv(os.path.join(output_folder, output_folder_name, "Permeability", well+"_NMRGP.csv"), header=True)
else:
    print("Mismatch between the number of PHIT samples and processed Gaussians:", len(phi_nmr), "vs", len(weights))
