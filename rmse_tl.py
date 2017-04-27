# import TechlogDatabase as db
# import numpy as np
#
# well = "R-538"
# dataset = "CORE_RCA_MHF"
#
# k_air = db.variableLoad(well, dataset, "KAIR")
# k_syn = db.variableLoad(well, dataset, "KSYN_V4")
# k_nmr = db.variableLoad(well, dataset, "NMRGP_Perm")
# k_sdr = db.variableLoad(well, dataset, "KSDR_HR")
# k_tim = db.variableLoad(well, dataset, "KTIM_HR")
#
# data = np.log10(np.transpose(np.array([k_air, k_syn, k_nmr, k_sdr, k_tim])))
# data = data[~np.isnan(data).any(axis=1)]
#
# k_syn_rmse = np.sqrt(np.mean((data[:,0] - data[:,1])**2))
# k_nmr_rmse = np.sqrt(np.mean((data[:,0] - data[:,2])**2))
# k_sdr_rmse = np.sqrt(np.mean((data[:,0] - data[:,3])**2))
# k_tim_rmse = np.sqrt(np.mean((data[:,0] - data[:,4])**2))
#
# print "Complete rows:", len(data)
# print "K_SYN RMSE:", k_syn_rmse
# print "NMR K RMSE:", k_nmr_rmse
# print "KSDR_HR RMSE:", k_sdr_rmse
# print "KTIM_HR RMSE:", k_tim_rmse
