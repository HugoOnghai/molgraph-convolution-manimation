from manim import *
from manim_chemistry import *
from manim_neural_network.neural_network import NeuralNetworkMobject

from manim_ml.neural_network import NeuralNetwork
from manim_ml.neural_network.layers import FeedForwardLayer, Convolutional2DLayer, ImageLayer

import numpy as np
import networkx as nx
import copy
from PIL import Image

from manimolconv import example as mlv

class ThankYou(Scene):
    def construct(self):
        file = f"../TY.mol"

        ty = Molecule(GraphMolecule).molecule_from_file(
            file, 
            ignore_hydrogens=False,
            label=True,
            numeric_label=False
        )

        # makes a manim_chemistry molecule
        mc_ty = MCMolecule.construct_from_file(file)

        self.play(Write(ty.scale_to_fit_width(config.frame_width/1.1)))    

        self.play(Wiggle(ty), ApplyWave(ty))

        ty_elem_matrix, featVec_group = init_element_features(mc_ty, ty, self)
        for i in range(5):
            print("conv step:", i)
            once_convolve(ty_elem_matrix, ty, featVec_group, self, 0.1)

SCALE_FACTOR = 2

class Title(Scene):
    def construct(self):
        t1=Text("Visualizing Molecular ")
        t2=Text("Graph Convolutions with")

        banner = ManimBanner().scale(0.5).next_to(t2,DOWN,buff=0.1)

        name = Text("Hugo Onghai", color=BLUE_C).next_to(banner,DOWN,buff=0.1)

        titleGroup = VGroup(t1, t2, banner, name).arrange(direction=DOWN)
        titleGroup.scale(1.5)

        self.play(Write(t1), Write(t2))
        self.play(banner.create())
        self.play(banner.expand())
        self.wait(1)
        self.play(Write(name))

class NeuralNetwork(Scene):
    def construct(self):
        # Define a neural network
        neural_network = NeuralNetworkMobject([10, 8, 8, 10, 5, 2]).scale(1.5)
        self.play(Create(neural_network))

class ConvolutionalNeuralNetwork(ThreeDScene):
    def construct(self):
        # Make nn
        image = Image.open("charizard.jpeg")
        numpy_image = np.asarray(image)
        # Make nn
        layers = [FeedForwardLayer(num_nodes=3), FeedForwardLayer(num_nodes=5), FeedForwardLayer(num_nodes=3)]
        nn = NeuralNetwork(layers)

        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Play animation
        forward_pass = nn.make_forward_pass_animation()
        self.play(ChangeSpeed(forward_pass, speedinfo={}), run_time=10)
        self.wait(1)

class GraphMoleculeFromMolecule(Scene):
    def construct(self):
        self.add(ammonia.scale(SCALE_FACTOR))

        elementMatrix, featVec_group = init_element_features(mc_ammonia, ammonia, self)

        self.wait(2)

        # neighbors = VGroup(*[ammonia.atoms[i] for i in range(1,4)])
        # self.play(Indicate(neighbors))

        # self.wait(2)

        elementMatrix, featVec_group = once_convolve(elementMatrix, ammonia, featVec_group, self)
        self.wait(2)
        elementMatrix, featVec_group = once_convolve(elementMatrix, ammonia, featVec_group, self)

class ConvolveIsomers(Scene):
    def construct(self):
        one_butanol = Molecule(GraphMolecule).molecule_from_file(
            "../one-butanol.mol", 
            ignore_hydrogens=True,
            label=True,
            numeric_label=False
        )
        
        mc_one_butanol = MCMolecule.construct_from_file(f"../one-butanol.mol")

        diethyl_ether = Molecule(GraphMolecule).molecule_from_file(
            "../diethyl_ether.sdf",
            ignore_hydrogens=True,
            label=True,
            numeric_label=False
        )

        mc_diethyl_ether = MCMolecule.construct_from_file(f"../diethyl_ether.sdf")

        molecules = Group(one_butanol, diethyl_ether).arrange(DOWN, buff=1).scale(1.5)
        self.play(Create(one_butanol), Create(diethyl_ether))

        for isomer, mc_isomer in zip([one_butanol, diethyl_ether], [mc_one_butanol, mc_diethyl_ether]):
            elementMatrix, featVec_group = init_element_features(mc_isomer, isomer, self)
            self.wait(2)
            elementMatrix, featVec_group = once_convolve(elementMatrix, isomer, featVec_group, self, run_time=0.1)
            self.wait(2)
            elementMatrix, featVec_group = once_convolve(elementMatrix, isomer, featVec_group, self, run_time=0.1)
            self.wait(2)
            elementMatrix, featVec_group = once_convolve(elementMatrix, isomer, featVec_group, self, run_time=0.1)

        self.wait(1)

class CodeFromConvolveIsomers(Scene):
    def construct(self):
        code = '''from manim import *
from manim_chemistry import *

class ConvolveIsomers(Scene):
    def construct(self):
        # import molecules
        one_butanol = Molecule(GraphMolecule).molecule_from_file("../one-butanol.mol")
        mc_one_butanol = MCMolecule.construct_from_file(f"../one-butanol.mol")
        diethyl_ether = Molecule(GraphMolecule).molecule_from_file("../diethyl_ether.sdf")
        mc_diethyl_ether = MCMolecule.construct_from_file(f"../diethyl_ether.sdf")

        # add molecules to manim frame
        molecules = Group(one_butanol, diethyl_ether).arrange(DOWN, buff=1).scale(1.5)
        self.play(Create(one_butanol), Create(diethyl_ether))

        # perform and animate convolution
        for isomer, mc_isomer in zip([one_butanol, diethyl_ether], [mc_one_butanol, mc_diethyl_ether]):
            elementMatrix, featVec_group = init_element_features(mc_isomer, isomer, self)
            self.wait(2)
            elementMatrix, featVec_group = once_convolve(elementMatrix, isomer, featVec_group, self, run_time=0.1)
            ...'''
        
        rendered_code = Code(
            code=code,
            background="window",
            tab_width=4,
            language="python",
            style="emacs",
            background_stroke_color=MAROON,
        ).scale(0.5)
        self.play(Write(rendered_code))

def testIndicate(molecule, scene):
    for i in range(len(molecule.get_atoms())):
        scene.play(Indicate(molecule.atoms[i+1]))
        scene.wait(1)

def init_element_features(mc_molecule, molecule, scene):
    # currently assuming that the indexing of mc_molecule is the same as molecule

    # getting distinct elements in the molecule
    elementlist = [atom.element.symbol for atom in mc_molecule.atoms]
    unique_elements = list(set(elementlist)) # making a list a set first removes redundancies

    # creating a matrix to store the one-hot encoding of each atom (rows = elements, columns = atoms)
    #element_features_matrix = np.zeros((len(unique_elements), len(molecule.get_atoms())))
    element_features_matrix = np.zeros((len(unique_elements), len(list(molecule.atoms))))

    # creating a one-hot encoding for each atom
    for atom_index, atom in enumerate(mc_molecule.atoms):
        element_features_matrix[unique_elements.index(atom.element.symbol), atom_index] = 1

    featVec_group = draw_element_features(element_features_matrix, molecule, scene)

    return element_features_matrix, featVec_group

def draw_element_features(element_features_matrix, molecule, scene):
    vg = VGroup()

    for atom_index, atom in enumerate(list(molecule.atoms)):
        pos = molecule.find_atom_position_by_index(atom_index + 1)

        featVec = Matrix( [[round(value, 2)] for value in element_features_matrix[:, atom_index].tolist()] ).scale(0.5)
    
        featVec.next_to(molecule.atoms[atom_index+1], UP)

        vg.add(featVec)

    scene.play(Write(vg))

    return vg

def once_convolve(element_features_matrix, molecule, featVec_group, scene, run_time=1):
    new_element_features_matrix = copy.deepcopy(element_features_matrix)

    new_featVec_VGroup = VGroup()

    for atom_index, atom in enumerate(list(molecule.atoms)):
        curr_featVec = copy.deepcopy(element_features_matrix[:, atom_index])
        curr_featVec_MObject = featVec_group[atom_index]

        # get the neighbors of the atom
        # index is 1-based, for the graph
        neighbors_index = molecule._graph.neighbors(atom_index + 1)
        neighbors_list = list(neighbors_index)
        num_neighbors = len(neighbors_list)

        neighborVec_group = VGroup()
        for index in neighbors_list:
            neighbor_featVec = element_features_matrix[:, index - 1] # grabs values from the origianl element_features_matrix before any convolutions were done
            neighborVec_group.add(featVec_group[index-1])
            curr_featVec += neighbor_featVec

        curr_featVec /= (num_neighbors+1) # adding one to count the atom itself 

        new_element_features_matrix[:, atom_index] = curr_featVec

        # indicate atom and its neighbors
        # molecule.atoms is a dict so it is also 1-based
        animate_conv_at_atom = []
        for i in neighbors_list:
            animate_conv_at_atom.append(Indicate(molecule.atoms[i], run_time=1, color=GREEN))

        new_featVec = Matrix( [[round(value, 2)] for value in new_element_features_matrix[:, atom_index].tolist()] ).scale(0.5)
        new_featVec.move_to(curr_featVec_MObject)

        new_featVec_VGroup.add(new_featVec) # since atoms iterate in order, we can always add the new featVec to the end of the VGroup

        scene.play(Indicate(molecule.atoms[atom_index+1], run_time=1, color=RED), *animate_conv_at_atom, run_time=run_time)
        scene.play(FadeOut(curr_featVec_MObject), TransformFromCopy(neighborVec_group, new_featVec), run_time=run_time)

    return new_element_features_matrix, new_featVec_VGroup

#config.renderer = "opengl"
class DrawIsomers(ThreeDScene):
    def construct(self):
        ethanol = ThreeDMolecule.molecule_from_file("../ethanol.mol", ignore_hydrogens=False, three_d=True)
        ethanol_label = Text("Ethanol").next_to(ethanol, DOWN)
        ethanolGroup = Group(ethanol, ethanol_label)
     
        dimethyl_ether = ThreeDMolecule.molecule_from_file("../dimethyl_ether.mol", ignore_hydrogens=False, three_d=True)
        dimethyl_ether_label = Text("Dimethyl Ether").next_to(dimethyl_ether, DOWN)
        dimethyl_etherGroup = Group(dimethyl_ether, dimethyl_ether_label)

        chemicals1 = Group(ethanolGroup, dimethyl_etherGroup).arrange(RIGHT, buff = 1)

        one_butanol = ThreeDMolecule.molecule_from_file("../one-butanol.mol", ignore_hydrogens=False, three_d=True)
        one_butanol_label = Text("One-Butanol").next_to(one_butanol, DOWN)
        one_butanolGroup = Group(one_butanol, one_butanol_label)
        
        diethyl_ether = ThreeDMolecule.molecule_from_file("../diethyl_ether.sdf", ignore_hydrogens=True, three_d=True)
        diethyl_ether_label = Text("Diethyl Ether").next_to(diethyl_ether, DOWN)
        diethyl_etherGroup = Group(diethyl_ether, diethyl_ether_label)
        
        chemicals2 = Group(one_butanolGroup, diethyl_etherGroup).arrange(RIGHT, buff = 1)

        prezGroup = Group(chemicals1, chemicals2).arrange(DOWN, buff = 1)
        chemGroup = Group(ethanol, dimethyl_ether, one_butanol, diethyl_ether)

        self.play(
                Create(ethanol),
                Create(dimethyl_ether), 
                Write(ethanol_label),
                Write(dimethyl_ether_label)
            )
        self.wait()

        self.play(
            Create(one_butanol),
            Create(diethyl_ether),
            Write(one_butanol_label),
            Write(diethyl_ether_label)
            )
        self.wait()

        self.play(*[Rotating(molecule, axis=np.array([0,1,0]), about_point=molecule.get_center(), rate_func=linear, run_time=10) for molecule in chemGroup])

class MarioAbstraction(Scene):
    def construct(self):
        # Example 16×12 grid of Manim color names.
        # Replace these with your 16-row by 12-column color list.
        color_grid = [
            [BLACK, BLACK, BLACK, RED, RED, RED, RED, RED, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, RED, RED, RED, RED, RED, RED, RED, RED, RED, BLACK],
            [BLACK, BLACK, DARK_BROWN, DARK_BROWN, DARK_BROWN, GOLD_A, GOLD_A, BLACK, GOLD_A, BLACK, BLACK, BLACK],
            [BLACK, DARK_BROWN, GOLD_A, DARK_BROWN, GOLD_A, GOLD_A, GOLD_A, BLACK, GOLD_A, GOLD_A, GOLD_A, BLACK],
            [BLACK, DARK_BROWN, GOLD_A, DARK_BROWN, DARK_BROWN, GOLD_A, GOLD_A, GOLD_A, BLACK, GOLD_A, GOLD_A, GOLD_A],
            [BLACK, DARK_BROWN, DARK_BROWN, GOLD_A, GOLD_A, GOLD_A, GOLD_A, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, GOLD_A, GOLD_A, GOLD_A, GOLD_A, GOLD_A, GOLD_A, GOLD_A, BLACK, BLACK],
            [BLACK, BLACK, RED, RED, DARK_BLUE, RED, RED, DARK_BLUE, RED, BLACK, BLACK, BLACK],
            [BLACK, RED, RED, RED, DARK_BLUE, RED, RED, DARK_BLUE, RED, RED, RED, BLACK],
            [RED, RED, RED, RED, DARK_BLUE, RED, RED, DARK_BLUE, RED, RED, RED, RED],
            [WHITE, WHITE, RED, DARK_BLUE, YELLOW_C, DARK_BLUE, DARK_BLUE, YELLOW_C, DARK_BLUE, RED, WHITE, WHITE],
            [WHITE, WHITE, WHITE, DARK_BLUE, DARK_BLUE, DARK_BLUE, DARK_BLUE, DARK_BLUE, DARK_BLUE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, DARK_BLUE, DARK_BLUE, DARK_BLUE, DARK_BLUE, DARK_BLUE, DARK_BLUE, DARK_BLUE, DARK_BLUE, WHITE, WHITE],
            [BLACK, BLACK, DARK_BLUE, DARK_BLUE, DARK_BLUE, BLACK, BLACK, DARK_BLUE, DARK_BLUE, DARK_BLUE, BLACK, BLACK],
            [BLACK, DARK_BROWN, DARK_BROWN, DARK_BROWN, BLACK, BLACK, BLACK, BLACK, DARK_BROWN, DARK_BROWN, DARK_BROWN, BLACK],
            [DARK_BROWN, DARK_BROWN, DARK_BROWN, DARK_BROWN, BLACK, BLACK, BLACK, BLACK, DARK_BROWN, DARK_BROWN, DARK_BROWN, DARK_BROWN],
        ]
        # First, we collect rows in a VGroup.
        all_rows = VGroup()

        for row_index, row_colors in enumerate(color_grid):
            row_squares = VGroup()
            for col_index, color_name in enumerate(row_colors):
                square = Square(side_length=0.5)
                square.set_fill(color_name, opacity=1)
                square.set_stroke(width=0)

                if col_index == 0:
                    # First square in the row:
                    if row_index == 0:
                        # Very first square in the grid
                        square.to_edge(UP + LEFT)
                    else:
                        # For the first square of each *new* row,
                        # place it below the first square of the previous row
                        square.next_to(all_rows[row_index - 1][0], DOWN, buff=0.1)
                else:
                    # Place to the right of the previous square in this row
                    square.next_to(row_squares[col_index - 1], RIGHT, buff=0.1)

                row_squares.add(square)

            all_rows.add(row_squares)

        # Now flatten all squares into a single VGroup
        pixel_grid = VGroup(*[square for row_squares in all_rows for square in row_squares]).scale(0.75)
        pixel_grid.center()

        self.play(Write(pixel_grid))
        self.wait()
        self.play(pixel_grid.animate.to_edge(LEFT))

        ### Turning Mario into a Graph
        # find number of vertices (total pixels)
        total_pixels = sum(len(row) for row in color_grid)
        vertices = np.reshape(np.arange(1, total_pixels+1, 1), (len(color_grid), -1))
        vertex_config = {}

        # Build edges: each pixel (r,c) connects to its 8 neighbors
        # We only connect "forward" so we don't double-draw edges:
        # neighbor must have a larger row or same row with a larger column.
        edges = []
        num_rows = len(color_grid)
        for r in range(num_rows):
            num_cols = len(color_grid[r])
            for c in range(num_cols):
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        # Skip (0,0) offset => itself
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        # Check bounds
                        if 0 <= nr < num_rows and 0 <= nc < len(color_grid[nr]):
                            # Only add the edge if neighbor is "greater" in scanning order
                            if (nr > r) or (nr == r and nc > c):
                                edges.append((vertices[r][c], vertices[nr][nc]))

        # Build vertex_config
        for r in range(num_rows):
            num_cols = len(color_grid[r])
            for c in range(num_cols):
                color = color_grid[r][c]
                vertex_index = vertices[r][c]
                vertex_config[vertex_index] = {
                    "fill_color": color,
                    "fill_opacity": 1,
                    "radius": 0.5
                }

        graph = Graph(
            vertices=[index for row in vertices for index in row],
            edges=edges,
            vertex_config=vertex_config,
            edge_config={"stroke_width": 2}
        )

        buffer = 1
        for r in range(num_rows):
            num_cols = len(color_grid[r])
            for c in range(num_cols):
                vertex_index = vertices[r][c]
                graph.vertices[vertex_index].move_to((buffer*c, buffer*-r, 1))

        graph.scale_to_fit_height(pixel_grid.height)
        graph.move_to(pixel_grid)
        self.play(Transform(pixel_grid, graph), run_time=3)
        self.wait()

        for r in range(0, num_rows, 3):
            num_cols = len(color_grid[r])
            for c in range(0, num_cols, 3):
                vertex_index = vertices[r][c]
                neighbors_list = list(graph._graph.neighbors(vertex_index))
                # self.play(Indicate(graph.vertices[vertex_index], color=RED), *[Indicate(graph.vertices[neighbor_index], color=GREEN) for neighbor_index in neighbors_list])

        subNodes = [98, 99, 100, 110, 111, 112, 122, 123, 124]
        subVertex_Config = {}

        for i, node_ind in enumerate(subNodes):
            subVertex_Config[str(i+1)] = {
                    "fill_color": graph.vertices[node_ind].get_color(),
                    "fill_opacity": 1,
                    "radius": 1
                }
            
        print(subVertex_Config)

        nx_graphReshape = graph._graph.subgraph(subNodes)
        graphReshape = Graph(
            vertices=list(nx_graphReshape.nodes), 
            edges=list(nx_graphReshape.edges),
            vertex_config=subVertex_Config
        )

        graphReshape.scale_to_fit_height(pixel_grid.height).next_to(graph, RIGHT)

        self.play(ReplacementTransform(VGroup(*[graph.vertices[i] for i in [98, 99, 100, 110, 111, 112, 122, 123, 124]]), graphReshape), FadeOut(graph), run_time=5)
        

class Shapes(ThreeDScene):
    def construct(self):
        # Create axes
        axes = ThreeDAxes(
            x_range=[-20, 20, 1],  # [start, end, step(optional)]
            y_range=[-20, 20, 1],
            z_range=[-20, 20, 1],
            axis_config={
                "include_tip": True,
                "include_ticks": False,
                "stroke_width": 2
            }
        )

        # Parameters
        se = 0.5
        nu = 40
        k = 20
        A = 2.2

        # Colors
        ecolor = rgb_to_color([0 / 255, 170 / 255, 31 / 255])
        mcolor = rgb_to_color([81 / 255, 0 / 255, 122 / 255])

        # Tracker for animation
        tracker = ValueTracker(0)

        # Updaters
        def update_elec(obj):
            t = tracker.get_value()
            xx = obj.get_x()
            pos = xx / se + nu / 2
            new_vec = Vector([0, A * np.sin((2 * np.pi / k) * pos + t), 0], color=ecolor)
            new_vec.shift([xx, 0, 0])
            new_vec.rotate(90 * DEGREES, axis=[1, 0, 0], about_point=[xx, 0, 0])
            obj.become(new_vec)

        def update_mag(obj):
            t = tracker.get_value()
            xx = obj.get_x()
            pos = xx / se + nu / 2
            new_vec = Vector([0, A * np.sin((2 * np.pi / k) * pos + t), 0], color=mcolor)
            new_vec.shift([xx, 0, 0])
            obj.become(new_vec)

        # Groups
        Electric = VGroup()
        Magnetic = VGroup()

        # Create vectors and add updaters
        for i in range(nu):
            x_shift = (i - nu / 2) * se

            elec = Vector([0, A * np.sin((2 * np.pi / k) * i), 0], color=ecolor)
            elec.shift([x_shift, 0, 0])
            elec.rotate(90 * DEGREES, axis=[1, 0, 0], about_point=[x_shift, 0, 0])
            elec.add_updater(update_elec)
            Electric.add(elec)

            mag = Vector([0, A * np.sin((2 * np.pi / k) * i), 0], color=mcolor)
            mag.shift([x_shift, 0, 0])
            mag.add_updater(update_mag)
            Magnetic.add(mag)

        # Set camera orientation
        self.set_camera_orientation(phi=65 * DEGREES,theta=100*DEGREES,gamma = 0*DEGREES)
        self.begin_ambient_camera_rotation(rate=0.02)

        # Uncomment this if you want the axes to appear
        # self.add(axes)

        # Animate creation of the fields
        self.play(
            Create(Electric),
            Create(Magnetic)
        )

        # Animate the tracker’s value to show wave propagation
        self.play(
            tracker.animate.set_value(15),
            run_time=15,
            rate_func=linear
        )


##############################################################################################################

meth_smiles = "CNC(C)Cc1ccccc1"

ammonia = Molecule(GraphMolecule).molecule_from_pubchem(
        name="ammonia", 
        ignore_hydrogens=False,
        three_d=True,
        label=False,
        numeric_label=True
    )

graph_ammonia = GraphMolecule.molecule_from_pubchem(
        name='ammonia',
        ignore_hydrogens=False,
        three_d=True,
        label=False,
        numeric_label=True
    )

mc_ammonia = MCMolecule.construct_from_file(f"../ammonia.mol")
