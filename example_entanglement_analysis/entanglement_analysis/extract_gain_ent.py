import pandas as pd

df = pd.read_pickle(f'2ww4__s0_e-1_m1.df')
df_gain = df[['frame_num', 'time', 'freqG0AA', 'freqG1AA']]
df_gain.to_csv('gain_entanglement.dat', sep='\t', float_format="%8f")
