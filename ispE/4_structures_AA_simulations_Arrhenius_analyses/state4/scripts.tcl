# Load the PDB file from command line argument
if {[llength $argv] > 0} {
    set pdb_file [lindex $argv 0]
    mol new $pdb_file
} else {
    puts "Usage: vmd -e script.tcl -args <pdb_file>"
    exit
}


# Remove the default line representation
mol delrep 0 top

# Color whole native structure in gray
# mol selection top
# mol representation NewCartoon
# mol color ColorID 8
# mol material Edgy
# mol addrep top

# Set variable for state_1 residues
set id1 16
set id2 26
# threading residue
set id3 191

# Display bond between contact residues 44 and 67
set sel [atomselect top "(resid $id1 or resid $id2) and name CA"]
set idx [$sel get index]
topo addbond [lindex $idx 0] [lindex $idx 1]

# Bond representation for contact residues
mol representation Bonds 0.5 12.0
mol color ColorID 4
mol selection "(resid $id1 or resid $id2) and name CA"
mol material Edgy
mol addrep top

# Display VDW representation for contact residues
mol selection "(resid $id1 or resid $id2) and name CA"
mol representation VDW 0.7 80
mol color ColorID 4
mol material Edgy
mol addrep top

# Color and represent loop region in red
mol selection "resid $id1 to $id2"
mol representation NewCartoon
mol color ColorID 1
mol material Edgy
mol addrep top

# Color and represent threading region in blue
# Color and represent threading region as ±6 residues from id3 in blue
set threading_start [expr {$id3 - 6}]
set threading_end [expr {$id3 + 6}]
mol selection "resid $threading_start to $threading_end"
mol representation NewCartoon
mol color ColorID 0
mol material Edgy
mol addrep top

# Color and represent other regions in white
mol selection "not ((resid $id1 to $id2) or (resid $threading_start to $threading_end))"
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
