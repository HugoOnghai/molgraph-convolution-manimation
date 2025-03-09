from manim import *
from manim_chemistry import *
import numpy as np

meth_smiles = "CNC(C)Cc1ccccc1"

ammonia = Molecule(GraphMolecule).molecule_from_pubchem(
        name="ammonia", 
        ignore_hydrogens=False,
        three_d=True,
        label=False,
        numeric_label=True
    )

class GraphMoleculeFromMolecule(Scene):
    def construct(self):
        self.add(molecule)

def init_element_features(mc_molecule, molecule):
    # currently assuming that the indexing of mc_molecule is the same as molecule

    # getting distinct elements in the molecule
    elementlist = [atom.element.symbol for atom in mc_molecule.atoms]
    unique_elements = list(set(elementlist)) # making a list a set first removes redundancies

    # creating a matrix to store the one-hot encoding of each atom (rows = elements, columns = atoms)
    element_features_matrix = np.zeros((len(unique_elements), len(molecule.get_atoms())))

    # creating a one-hot encoding for each atom
    for atom_index, atom in enumerate(mc_molecule.atoms):
        element_features_matrix[unique_elements.index(atom.element.symbol), atom_index] = 1

    draw_element_features(element_features_matrix)

    return element_features_matrix

def draw_element_features(element_features_matrix):
    pass

atoms = ammonia.get_atoms()
print(atoms)

mc_ammonia = MCMolecule.construct_from_file(f"../ammonia.mol")

print(init_element_features(mc_ammonia, ammonia))
