from topoly import lasso_type, make_surface
from topoly.params import PrecisionSurface, DensitySurface, SurfacePlotFormat, test
from topoly import gln
import topoly

native = '2ww4_chain_a_rebuilt_mini_clean.pdb'
res = lasso_type(native, [35, 269], min_dist=[10,4,5], pic_files=SurfacePlotFormat.VMD, output_prefix='native', more_info=True)
print(res)

