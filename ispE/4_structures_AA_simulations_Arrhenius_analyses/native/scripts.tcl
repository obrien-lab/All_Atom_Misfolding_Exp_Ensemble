# Load the native PDB file
set native_pdb_file "2ww4_chain_a_rebuilt_mini_clean.pdb"
mol new $native_pdb_file

# Remove the default line representation
mol delrep 0 top
# Color whole native structure in gray
mol selection top
mol representation NewCartoon
mol color ColorID 8
mol material Edgy
mol addrep top


# Set color scale and projection options
# Set background color to white
color Display Background white
color scale method BGR
axes location off
display projection orthographic
