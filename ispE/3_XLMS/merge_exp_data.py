import sys, os
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind, combine_pvalues
from statsmodels.stats.multitest import multipletests

exp_data_file = './R1_xi_final_0213_v3.xlsx'
out_name = 'PGK_XLMS_R1_merged.xlsx'

XL_data_df = pd.read_excel(exp_data_file)
XL_data = {}
native_abundance_list = XL_data_df[['Abundances: Light,Native', 'Abundances: Light,Native.1', 'Abundances: Light,Native.2']].to_numpy()
refolded_abundance_list = XL_data_df[['Abundances: Heavy,Refolded', 'Abundances: Heavy,Refolded.1', 'Abundances: Heavy,Refolded.2']].to_numpy()

key_list = []
data_list = []
for idx, pair in enumerate(XL_data_df[['RealLinkage1','RealLinkage2']].to_numpy()):
    if not np.any(pd.isnull(pair)):
        resid_list = sorted(pair, key=lambda x: int(x[1:]))
        pair_str = '-'.join(resid_list)
        na_list = native_abundance_list[idx,:]
        ra_list = refolded_abundance_list[idx,:]
        
        # compute ratio and p-value
        nn = len(np.where(na_list == 1000)[0])
        nr = len(np.where(ra_list == 1000)[0])
        if nn == 0 and nr == 0: # all values available
            x1 = ra_list
            x2 = na_list
            alt_hyp = 'two-sided'
        elif (nn == 1 and nr == 0) or (nn == 0 and nr == 1): # one value is missing
            x1 = ra_list[ra_list!=1000]
            x2 = na_list[na_list!=1000]
            alt_hyp = 'two-sided'
        elif nn == len(na_list) and nr == 0: # All-or-nothing case
            x1 = ra_list
            x2 = norm.rvs(1e4, 1e3, 3)
            alt_hyp = 'greater'
        elif nn == 0 and nr == len(ra_list): # All-or-nothing case
            x1 = norm.rvs(1e4, 1e3, 3)
            x2 = na_list
            alt_hyp = 'less'
        else:
            continue
        ratio = np.mean(x1)/np.mean(x2)
        ttest_result = ttest_ind(x1, x2, alternative=alt_hyp, equal_var=False)
            
        # if pair_str in XL_data.keys():
        #     XL_data[pair_str].append([ratio, ttest_result[0], ttest_result[1]])
        # else:
        #     XL_data[pair_str] = [[ratio, ttest_result[0], ttest_result[1]]]

        key_list.append(pair_str)
        data_list.append([ratio, ttest_result[0], ttest_result[1]])

data_list = np.array(data_list)
adj_pvalue_list = multipletests(data_list[:,2], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]
data_list = np.hstack((data_list, adj_pvalue_list.reshape((len(adj_pvalue_list),1))))
for idx, key in enumerate(key_list):
    if key in XL_data.keys():
        XL_data[key].append(data_list[idx,:])
    else:
        XL_data[key] = [data_list[idx,:]]

for k, v in XL_data.items():
    # vv = np.array(v)
    # adj_pvalue = multipletests(vv[:,2], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]
    # XL_data[k] = np.hstack((vv,adj_pvalue.reshape((len(adj_pvalue),1))))
    XL_data[k] = np.array(v)
        
XL_data_merged = {}
for k, v in XL_data.items():
    idx_list = np.where(np.sign(v[:,1]) == np.sign(np.sign(v[:,1]).sum()))[0]
    if len(idx_list) != 0:
        ratio_merged = np.median(v[idx_list,0])
        pvalue_merged = combine_pvalues(v[idx_list,2])[1]
        adj_pvalue_merged = combine_pvalues(v[idx_list,3])[1]
    else:
        ratio_merged = np.median(v[:,0])
        pvalue_merged = 1
        adj_pvalue_merged = 1
    XL_data_merged[k] = [np.log2(ratio_merged), np.abs(-np.log10(pvalue_merged)), np.abs(-np.log10(adj_pvalue_merged))]

df = pd.DataFrame([[k,*v] for k,v in XL_data_merged.items()], columns=['Pairs', 'log2(heavy/light)', '-log10(p-value)', '-log10(Adj. p-value)'])

print(df)

df.to_excel(out_name, index=False, float_format="%.4f")
