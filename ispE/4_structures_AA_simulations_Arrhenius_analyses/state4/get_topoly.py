from topoly import lasso_type, make_surface
from topoly.params import PrecisionSurface, DensitySurface, SurfacePlotFormat, test
from topoly import gln
import topoly

misfolded = 'state_1.pdb'
res = lasso_type(misfolded, [35, 269], min_dist=[10,4,5], pic_files=SurfacePlotFormat.VMD, output_prefix='misfold',more_info=True)
print(res)

