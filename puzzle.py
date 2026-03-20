"""
Jigsaw Puzzle Toolkit - Core Puzzle Module

This module provides the core functionality for creating and manipulating jigsaw puzzles.
It handles different puzzle types (squared, polyomino, pattern-based, irregular) with
various rotation options, and manages puzzle piece extraction, transformation, and storage.

Main Classes
------------
PuzzleType : Enum
    Enumeration of supported puzzle types with rotation configurations
Puzzle : class
    Main class for creating, processing, and saving jigsaw puzzles
"""

from enum import Enum
import cv2
from shapely import Polygon, transform
import numpy as np 
import os, json
import matplotlib.pyplot as plt
# only needed for rotation!
# is it too much?
import numpy as np 
import random 
import scipy
import shapely
from skimage.transform import resize

from puzzle_generator import PuzzleGenerator



#######################################################
#                                                     #
#  ██████╗ ██╗   ██╗███████╗███████╗██╗     ███████╗  #
#  ██╔══██╗██║   ██║╚══███╔╝╚══███╔╝██║     ██╔════╝  #
#  ██████╔╝██║   ██║  ███╔╝   ███╔╝ ██║     █████╗    #
#  ██╔═══╝ ██║   ██║ ███╔╝   ███╔╝  ██║     ██╔══╝    #
#  ██║     ╚██████╔╝███████╗███████╗███████╗███████╗  #
#  ╚═╝      ╚═════╝ ╚══════╝╚══════╝╚══════╝╚══════╝  #
#                                                     #
#  ████████╗██╗   ██╗██████╗ ███████╗                 #
#  ╚══██╔══╝╚██╗ ██╔╝██╔══██╗██╔════╝                 #
#     ██║    ╚████╔╝ ██████╔╝█████╗                   #
#     ██║     ╚██╔╝  ██╔═══╝ ██╔══╝                   #
#     ██║      ██║   ██║     ███████╗                 #
#     ╚═╝      ╚═╝   ╚═╝     ╚══════╝                 #
#                                                     #
#######################################################
class PuzzleType(Enum):
    """
    Enumeration of puzzle types with rotation configurations.
    
    Each puzzle type is encoded as a two-character string:
    - First character: piece shape (S=squared, P=polyomino, M=pattern map, I=irregular)
    - Second character: rotation type (1=none, 2=90° increments, 3=free rotation)
    
    Attributes
    ----------
    type_S1 : str
        Squared pieces, no rotation
    type_S2 : str
        Squared pieces, 90 degree rotations
    type_P1 : str
        Polyomino pieces, no rotation
    type_P2 : str
        Polyomino pieces, 90 degree rotations
    type_P3 : str
        Polyomino pieces, free rotations
    type_M1 : str
        Pattern map pieces, no rotation
    type_M2 : str
        Pattern map pieces, 90 degree rotations
    type_M3 : str
        Pattern map pieces, free rotations
    type_I1 : str
        Irregular pieces, no rotation
    type_I2 : str
        Irregular pieces, 90 degree rotations
    type_I3 : str
        Irregular pieces, free rotations
    
    Examples
    --------
    >>> puzzle_type = PuzzleType.type_I2
    >>> print(puzzle_type._type_str())
    'irregular pieces'
    >>> print(puzzle_type._rot_str())
    'rotations multiple of 90 degrees'
    """
    # SQUARED
    type_S1 = 'S1'
    type_S2 = 'S2'
    # POLYOMINO
    type_P1 = 'P1'
    type_P2 = 'P2'
    type_P3 = 'P3'
    # PIECES from PATTERN MAP
    type_M1 = 'M1'
    type_M2 = 'M2'
    type_M3 = 'M3'
    # IRREGULAR
    type_I1 = 'I1'
    type_I2 = 'I2'
    type_I3 = 'I3'

    def __str__(self):
        return self.value
    
    def _rot(self):
        """
        Extract rotation type as integer.
        
        Returns
        -------
        int
            1 for no rotation, 2 for 90° rotations, 3 for free rotations
        """
        return int(self.value[-1])
    
    def _rot_str(self):
        """
        Get human-readable rotation type description.
        
        Returns
        -------
        str
            Description of the rotation type
        """
        rot_type_as_int = int(self.value[-1]) 
        if rot_type_as_int == 1:
            rot_str = 'no rotations'
        elif rot_type_as_int == 2:
            rot_str = 'rotations multiple of 90 degrees'
        elif rot_type_as_int == 3:
            rot_str = 'free rotations (float values)'
        else:
            rot_str = 'unknown rotations'
        return rot_str

    def _type(self):
        """
        Extract piece type character.
        
        Returns
        -------
        str
            'S', 'P', 'M', or 'I' representing the piece shape type
        """
        return self.value[0]

    def _type_str(self):
        """
        Get human-readable piece type description.
        
        Returns
        -------
        str
            Description of the piece type
        """
        if self.value[0] == 'S':
            type_str = "squared pieces"
        elif self.value[0] == 'P':
            type_str = "polyominoes pieces"
        elif self.value[0] == 'M':
            type_str = "pattern map pieces"
        elif self.value[0] == 'I':
            type_str = "irregular pieces"
        else:
            type_str = "unknown"
        return type_str
    

def extract_binary_mask(irregular_image: np.ndarray, background: int = 0, close: bool = True):
    """
    Extract binary mask from an image with transparent or uniform background.
    
    Parameters
    ----------
    irregular_image : np.ndarray
        Input image with shape (H, W, C) where C is 3 (RGB) or 4 (RGBA)
    background : int, optional
        Background pixel value to mask out (default: 0)
    close : bool, optional
        Whether to apply morphological closing to fill small holes (default: True)
    
    Returns
    -------
    np.ndarray
        Binary mask with shape (H, W), dtype uint8, values 0 or 1
    
    Notes
    -----
    For RGBA images, uses the alpha channel. For RGB images, uses the red channel.
    """
    # Check if image has alpha channel (RGBA) or is just RGB
    if irregular_image.shape[2] == 4:
        # Use alpha channel: foreground where alpha != background
        binary_mask = 1 - (irregular_image[:,:,3] == background).astype(np.uint8)
    else:
        # Use red channel: foreground where red != background
        binary_mask = 1 - (irregular_image[:,:,0] == background).astype(np.uint8)
    
    # Apply morphological closing to fill small holes and smooth boundaries
    if close == True:
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones_like((5,5)))
    
    return binary_mask

def calculate_center_of_mass(binary_mask: np.ndarray, method: str = 'np'):
    """
    Calculate the center of mass (centroid) of a binary mask.
    
    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask with shape (H, W), values >= 0.5 considered as foreground
    method : str, optional
        Method to use: 'scipy' or 'np' (default: 'np')
    
    Returns
    -------
    list of float
        [center_x, center_y] coordinates of the center of mass
    
    Notes
    -----
    The numpy method ('np') is faster and doesn't require scipy import.
    Both methods should give identical results for binary masks.
    """
    if method == 'scipy':
        import scipy
        # SciPy's center_of_mass returns (y, x) order
        cent_y, cent_x = scipy.ndimage.center_of_mass(bmask)
    else: # method == 'np'
        # Find all foreground pixels (mask value >= 0.5)
        mass_y, mass_x = np.where(binary_mask >= 0.5)
        # Calculate average position
        cent_x = np.average(mass_x)
        cent_y = np.average(mass_y)
        # center = [ np.average(indices) for indices in np.where(th1 >= 255) ]
    
    return [cent_x, cent_y]

def extract_polygon(binary_mask: np.ndarray, return_vals: bool = False):
    """
    Extract polygon contour from binary mask.
    
    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask with shape (H, W)
    return_vals : bool, optional
        If True, also return bounding box half-diagonal and [height, width] (default: False)
    
    Returns
    -------
    polygon : shapely.Polygon
        Polygon representing the piece boundary
    bounding_box_half_diagonal : float, optional
        Half the diagonal of the bounding box (only if return_vals=True)
    wh : list of int, optional
        [height, width] of bounding box (only if return_vals=True)
    
    Raises
    ------
    ValueError
        If fewer than 4 contour points are found (invalid polygon)
    
    Notes
    -----
    The polygon coordinates are offset by -0.5 to align with pixel centers.
    Uses morphological dilation before contour detection for better edge definition.
    """
    bin_img = binary_mask.copy()
    # Dilate the mask slightly to ensure contour is well-defined
    bin_img = cv2.dilate(bin_img.astype(np.uint8), np.ones((2,2)), iterations=1)
    
    # Find external contours using simple approximation
    contours, _ = cv2.findContours(bin_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select the longest contour (should be the piece boundary)
    contour_points = contours[0]
    if len(contours) > 0:
        for cnt in contours:
            if len(cnt) > len(contour_points):
                contour_points = cnt    

    # Convert OpenCV contour points to Shapely format
    # Offset by -0.5 to align with pixel centers (OpenCV uses pixel corners)
    shapely_points = [(point[0][0]-0.5, point[0][1]-0.5) for point in contour_points]
    
    # Verify we have enough points to form a valid polygon
    if len(shapely_points) < 4:
        print('we have a problem, too few points', shapely_points)
        raise ValueError('\nWe have fewer than 4 points on the polygon, so we cannot create a Shapely polygon out of this points! Maybe something went wrong with the mask?')
    
    # Create Shapely polygon from points
    polygon = Polygon(shapely_points)
    
    # Calculate bounding box parameters
    x,y,w,h = cv2.boundingRect(contour_points)
    img_center = np.asarray(bin_img.shape[:2]) / 2

    # Half-diagonal of bounding box (useful for determining canvas size)
    bounding_box_half_diagonal = np.sqrt(np.square(w/2) + np.square(h/2))

    # TO SHOW THE BBOX    
    # plt.figure(figsize=(10, 10))
    # plt.title(f"w:{w}, h:{h}, bbhd:{bounding_box_half_diagonal}")
    # ax = plt.gca()
    # ax.imshow(binary_mask)
    # ax.plot(*(polygon.boundary.xy))
    # rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
    # ax.add_patch(rect)
    # plt.show()

    if return_vals == True:
        return polygon, bounding_box_half_diagonal, [h,w]
    else:
        return polygon



#######################################################
#                                                     #
#  ██████╗ ██╗   ██╗███████╗███████╗██╗     ███████╗  #
#  ██╔══██╗██║   ██║╚══███╔╝╚══███╔╝██║     ██╔════╝  #
#  ██████╔╝██║   ██║  ███╔╝   ███╔╝ ██║     █████╗    #
#  ██╔═══╝ ██║   ██║ ███╔╝   ███╔╝  ██║     ██╔══╝    #
#  ██║     ╚██████╔╝███████╗███████╗███████╗███████╗  #
#  ╚═╝      ╚═════╝ ╚══════╝╚══════╝╚══════╝╚══════╝  #
#                                                     #
#######################################################
class Puzzle:
    """
    Main class for creating and managing jigsaw puzzles.
    
    This class handles the entire pipeline of puzzle creation:
    - Loading input images or existing puzzle pieces
    - Generating pieces using PuzzleGenerator
    - Processing and transforming pieces (rotation, cropping, resizing)
    - Saving puzzle data and ground truth
    
    Parameters
    ----------
    input_path : str
        Path to input image or directory containing puzzle pieces
    puzzle_type : PuzzleType
        Type of puzzle to create (determines piece shape and rotation)
    output_path : str
        Directory where puzzle files will be saved
    input_type : str, optional
        Type of input: 'image', 'repair', or 'json' (default: 'image')
    target_size : int, optional
        Target size for rescaling puzzle pieces (default: 256)
    save_masks : bool, optional
        Whether to save binary masks (default: True)
    save_polygons : bool, optional
        Whether to save polygon data (default: True)
    
    Attributes
    ----------
    pieces : dict
        Dictionary of puzzle pieces with images, masks, polygons, and metadata
    gt : dict
        Ground truth information for puzzle assembly
    puzzle_info : dict
        Metadata about the puzzle configuration
    
    Examples
    --------
    >>> puzzle = Puzzle(input_path='image.jpg', puzzle_type=PuzzleType.type_I2, 
    ...                 output_path='output/', target_size=500)
    >>> puzzle.prepare_puzzle(num_pieces=25, crop_pieces=True)
    >>> puzzle.save()
    """

    def __init__(self, input_path:str, puzzle_type:PuzzleType, output_path:str, input_type:str = 'image', 
                 target_size:int = 256, save_masks:bool = True, save_polygons:bool = True):
        
        self.input_type = input_type
        self.input_path = input_path
        self.puzzle_type = puzzle_type
        self.target_size = target_size
        
        # Create output directory structure
        self.output_dir = self._create_output_dir(output_path, input_path)
        
        # Flags for what data to save
        self.save_masks = save_masks
        self.save_polygons = save_polygons
        
        # Initialize data containers
        self.pieces = {}  # Dictionary to store all piece data
        self.gt = {       # Ground truth for puzzle assembly
            'pieces': {},
            'adjacency': []
        }
        self.puzzle_info = {  # Metadata about puzzle configuration
            'type': str(puzzle_type),
            'type_s': puzzle_type._type_str(),
            'rotation_type': puzzle_type._rot(),
            'rotation_type_s': puzzle_type._rot_str(),
        }

    def _create_output_dir(self, output_path, input_path):
        """
        Create output directory with appropriate naming.
        
        Parameters
        ----------
        output_path : str
            Base output directory
        input_path : str
            Input file/directory path (used for naming)
        
        Returns
        -------
        str
            Full path to created output directory
        
        Notes
        -----
        Directory naming format: {puzzle_type}_{input_name}_{target_size}
        """
        # Extract base name from input path (without extension)
        if self.input_type == 'image':
            name_no_ext = os.path.basename(input_path).split('.')[0]
        else:
            name_no_ext = os.path.basename(input_path)
        
        # Construct output directory name with puzzle configuration
        folder_name = f"{self.puzzle_type}_{name_no_ext}_{self.target_size}"
        out_dir = os.path.join(output_path, folder_name)
        
        # Create directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def prepare_puzzle(self, num_pieces:int = 9, crop_pieces:bool = True, pattern_map_path:str = ''):
        """
        Main method to prepare puzzle by generating and processing pieces.
        
        This method orchestrates the entire puzzle creation process based on input type.
        For images, it uses PuzzleGenerator to create pieces. For existing pieces,
        it loads and processes them.
        
        Parameters
        ----------
        num_pieces : int, optional
            Number of pieces to create (for image input) (default: 9)
        crop_pieces : bool, optional
            Whether to crop pieces to remove excess background (default: True)
        pattern_map_path : str, optional
            Path to pattern map file for pattern-based puzzles (default: '')
        
        Notes
        -----
        The behavior varies based on self.input_type:
        - 'image': Generates pieces from scratch using PuzzleGenerator
        - 'repair'/'json': Loads existing pieces and processes them
        """
        print(f"\nPREPARING PUZZLE OF TYPE {self.puzzle_type} [input: {self.input_type}]")
        
        if self.input_type == 'image':
            # Generate new puzzle from image
            self._prepare_puzzle_from_image(num_pieces, crop_pieces, pattern_map_path)
        
        elif self.input_type == 'repair' or self.input_type == 'json':
            # Load and process existing puzzle pieces
            self._prepare_puzzle_from_pieces()

    def _prepare_puzzle_from_image(self, num_pieces, crop_pieces, pattern_map_path):
        """
        Generate puzzle pieces from an input image.
        
        Parameters
        ----------
        num_pieces : int
            Number of pieces to create
        crop_pieces : bool
            Whether to crop pieces to bounding boxes
        pattern_map_path : str
            Path to pattern map (for M-type puzzles)
        
        Notes
        -----
        Uses PuzzleGenerator to cut the image into pieces, then processes each piece
        to extract masks, polygons, and apply transformations.
        """
        # Load and preprocess input image
        input_image = plt.imread(self.input_path)
        image_name = os.path.basename(self.input_path).split('.')[0]
        
        # Construct parameters for puzzle generation
        puzzle_params = {
            'name': image_name,
            'rotation_type': self.puzzle_type._rot(),
            'rotation_type_s': self.puzzle_type._rot_str(),
            'pieces_type': self.puzzle_type._type(),
            'pieces_type_s': self.puzzle_type._type_str(),
            'padding': 9,
            'rotation_range': 180,
        }
        
        # Initialize puzzle generator
        print(f" > Creating Puzzle Generator")
        puzzle_gen = PuzzleGenerator(input_image, puzzle_params)
        
        # Generate pieces based on puzzle type
        if self.puzzle_type._type() == 'S':
            # Squared pieces
            print(f" > Generating {num_pieces} squared pieces")
            puzzle_gen.squared_pieces(
                num_rows=int(np.sqrt(num_pieces)),
                target_image_shape=(self.target_size, self.target_size),
                padding=puzzle_params['padding']
            )
        elif self.puzzle_type._type() == 'P':
            # Polyomino pieces
            print(f" > Generating polyomino pieces from pattern map: {pattern_map_path}")
            puzzle_gen.polyomino_pieces(
                pattern_map_path=pattern_map_path,
                target_image_shape=(self.target_size, self.target_size),
                padding=puzzle_params['padding']
            )
        elif self.puzzle_type._type() == 'M':
            # Pattern map pieces
            print(f" > Generating pieces from pattern map: {pattern_map_path}")
            puzzle_gen.pattern_map_pieces(
                pattern_map_path=pattern_map_path,
                target_image_shape=(self.target_size, self.target_size),
                padding=puzzle_params['padding']
            )
        elif self.puzzle_type._type() == 'I':
            # Irregular pieces
            print(f" > Generating {num_pieces} irregular pieces")
            puzzle_gen.run(
                piece_n=num_pieces,
                offset_rate_h=0.2,
                offset_rate_w=0.2,
                small_region_area_ratio=0.25,
                rot_range=180,
                smooth_flag=True,
                alpha_channel=True
            )
        
        # Extract raw pieces from generator
        self.raw_images = puzzle_gen.raw_pieces_images
        self.raw_masks = puzzle_gen.raw_pieces_masks
        self.raw_polygons = puzzle_gen.raw_pieces_polygons
        
        # Process each piece
        print(f" > Processing {len(self.raw_images)} pieces")
        for idx, (img, mask, poly) in enumerate(zip(self.raw_images, self.raw_masks, self.raw_polygons)):
            piece_name = f"piece_{idx:04d}"
            
            # Extract polygon and calculate enclosing radius
            _, enclosing_radius, wh = extract_polygon(mask, return_vals=True)
            
            # Store raw piece data
            self.pieces[piece_name] = {
                'raw_image': img,
                'raw_mask': mask,
                'raw_polygon': poly,
                'enclosing_radius': enclosing_radius,
                'size': wh
            }
            
            # Optionally crop and normalize piece
            if crop_pieces:
                cropped_img, cropped_mask, cropped_poly, scale = self.crop_and_rescale_piece(
                    idx, new_size=(self.target_size, self.target_size)
                )
                self.pieces[piece_name]['squared_image'] = cropped_img
                self.pieces[piece_name]['squared_mask'] = cropped_mask
                self.pieces[piece_name]['squared_polygon'] = cropped_poly
                self.pieces[piece_name]['rescale_factor'] = scale
        
        # Extract adjacency relationships between pieces
        self.gt['adjacency'] = puzzle_gen.adjacency_list
        
        # Store puzzle-specific info
        self.puzzle_info.update({
            'num_pieces': len(self.pieces),
            'target_size': self.target_size,
            'source_image': image_name
        })

    def _prepare_puzzle_from_pieces(self):
        """
        Load and process existing puzzle pieces from a directory.
        
        Reads puzzle data from a 'repair' or 'json' format directory containing
        images, masks, polygons, and metadata files.
        
        Notes
        -----
        Expected directory structure:
        - groundtruth_extended.json or info.json
        - Images in PNG/JPG format
        - Optional: binary masks and polygon files
        """
        print(f" > Loading pieces from {self.input_path}")
        
        # Load ground truth or info file
        if os.path.exists(os.path.join(self.input_path, 'groundtruth_extended.json')):
            with open(os.path.join(self.input_path, 'groundtruth_extended.json'), 'r') as f:
                puzzle_data = json.load(f)
        elif os.path.exists(os.path.join(self.input_path, 'info.json')):
            with open(os.path.join(self.input_path, 'info.json'), 'r') as f:
                puzzle_data = json.load(f)
        else:
            raise FileNotFoundError("No groundtruth_extended.json or info.json found!")
        
        # Extract fragments information
        fragments_info = puzzle_data.get('fragments', puzzle_data.get('pieces', {}))
        
        # Process each fragment/piece
        for frag_id, frag_data in fragments_info.items():
            # Load piece image
            img_path = os.path.join(self.input_path, 'images', f"{frag_id}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.input_path, f"{frag_id}.png")
            
            piece_img = plt.imread(img_path)
            
            # Extract mask and polygon from image
            piece_mask = extract_binary_mask(piece_img)
            piece_poly, enclosing_radius, wh = extract_polygon(piece_mask, return_vals=True)
            
            # Store piece data
            self.pieces[frag_id] = {
                'raw_image': piece_img,
                'raw_mask': piece_mask,
                'raw_polygon': piece_poly,
                'enclosing_radius': enclosing_radius,
                'size': wh,
                'metadata': frag_data
            }
            
            # Crop and rescale if needed
            cropped_img, cropped_mask, cropped_poly, scale = self.crop_and_rescale_piece_from_data(
                piece_img, piece_mask, piece_poly, new_size=(self.target_size, self.target_size)
            )
            
            self.pieces[frag_id]['squared_image'] = cropped_img
            self.pieces[frag_id]['squared_mask'] = cropped_mask
            self.pieces[frag_id]['squared_polygon'] = cropped_poly
            self.pieces[frag_id]['rescale_factor'] = scale
        
        # Load adjacency if available
        if 'adjacency' in puzzle_data:
            self.gt['adjacency'] = puzzle_data['adjacency']
        
        print(f" > Loaded {len(self.pieces)} pieces")

    def crop_and_rescale_piece(self, piece_index:int, new_size:tuple = (256, 256)):
        """
        Crop piece to its bounding box and rescale to target size.
        
        This method centers the piece in a square canvas, crops to the minimum
        bounding box that contains the piece, and rescales to the target size
        while maintaining the piece centered.
        
        Parameters
        ----------
        piece_index : int
            Index of the piece to process
        new_size : tuple of int, optional
            Target (height, width) for the output (default: (256, 256))
        
        Returns
        -------
        cropped_image : np.ndarray
            Cropped and rescaled RGB image
        cropped_mask : np.ndarray
            Cropped and rescaled binary mask
        cropped_polygon : shapely.Polygon
            Polygon adjusted to new coordinates
        rescaling_factor : float
            Scale factor applied (original_size / new_size)
        
        Notes
        -----
        The piece is first extended with padding to ensure it fits when cropped,
        then centered, cropped, and finally rescaled to the target size.
        """
        # Get raw piece data
        raw_image = self.raw_images[piece_index]
        raw_mask = self.raw_masks[piece_index]
        raw_polygon = self.raw_polygons[piece_index]
        
        # Calculate center of mass for the piece
        center = np.round(calculate_center_of_mass(raw_mask)).astype(int)
        
        # Determine required canvas size based on piece dimensions
        piece_size = self.sizes[piece_index]
        half_image_side = max(piece_size) // 2 + 10  # Add padding for safety
        
        # Create extended canvas (piece may need more space when rotated)
        extended_canvas_size = (half_image_side * 2 + 1, half_image_side * 2 + 1)
        self.extended_raw_images[piece_index] = np.zeros(
            (extended_canvas_size[0], extended_canvas_size[1], raw_image.shape[2])
        )
        self.extended_raw_masks[piece_index] = np.zeros(extended_canvas_size)
        
        # Place original piece in center of extended canvas
        start_y = half_image_side - raw_image.shape[0] // 2
        start_x = half_image_side - raw_image.shape[1] // 2
        self.extended_raw_images[piece_index][
            start_y:start_y + raw_image.shape[0],
            start_x:start_x + raw_image.shape[1]
        ] = raw_image
        self.extended_raw_masks[piece_index][
            start_y:start_y + raw_mask.shape[0],
            start_x:start_x + raw_mask.shape[1]
        ] = raw_mask
        
        # Translate polygon to extended canvas coordinates
        polygon_translation = np.array([start_x, start_y])
        self.extended_raw_polygons[piece_index] = transform(
            raw_polygon, lambda f: f + polygon_translation
        )
        
        # Recalculate center in extended canvas
        center = np.round(calculate_center_of_mass(
            self.extended_raw_masks[piece_index]
        )).astype(int)

        # Crop to square around center
        cropped_image = self.extended_raw_images[piece_index][
            center[1]-half_image_side: center[1]+half_image_side+1,
            center[0]-half_image_side: center[0]+half_image_side+1, :
        ]
        cropped_mask = self.extended_raw_masks[piece_index][
            center[1]-half_image_side: center[1]+half_image_side+1,
            center[0]-half_image_side: center[0]+half_image_side+1
        ]
        
        # Translate polygon to cropped coordinates
        polygon_translation = center - np.asarray([half_image_side, half_image_side])
        cropped_polygon = transform(
            self.extended_raw_polygons[piece_index],
            lambda f: f - polygon_translation
        )

        # Rescale to target size if specified
        if all(x > 0 for x in new_size):
            new_size = new_size[0]  # Assume square output
                
            cropped_image_size = cropped_image.shape[0]
            rescaling_factor = cropped_image_size / new_size
            
            # Resize image with anti-aliasing
            cropped_image = resize(cropped_image, (new_size, new_size), anti_aliasing=True)
            
            # Resize mask (threshold to maintain binary values)
            cropped_mask = (resize(
                cropped_mask, (new_size, new_size),
                anti_aliasing=True, preserve_range=True
            ) > 0.5).astype(np.uint8)
            
            # Scale polygon coordinates
            cropped_polygon = transform(
                cropped_polygon,
                lambda f: f * new_size / cropped_image_size
            )

        return cropped_image, cropped_mask, cropped_polygon, rescaling_factor

    def crop_and_rescale_piece_from_data(self, raw_image, raw_mask, raw_polygon, new_size=(256, 256)):
        """
        Crop and rescale a piece from raw data (used for loading existing pieces).
        
        Similar to crop_and_rescale_piece but works with direct data instead of indices.
        
        Parameters
        ----------
        raw_image : np.ndarray
            Raw piece image
        raw_mask : np.ndarray
            Raw piece binary mask
        raw_polygon : shapely.Polygon
            Raw piece polygon
        new_size : tuple of int, optional
            Target size (default: (256, 256))
        
        Returns
        -------
        cropped_image : np.ndarray
            Processed image
        cropped_mask : np.ndarray
            Processed mask
        cropped_polygon : shapely.Polygon
            Processed polygon
        rescaling_factor : float
            Applied scale factor
        """
        # Calculate center and bounding box
        center = np.round(calculate_center_of_mass(raw_mask)).astype(int)
        _, _, wh = extract_polygon(raw_mask, return_vals=True)
        
        half_image_side = max(wh) // 2 + 10
        
        # Create extended canvas
        extended_size = (half_image_side * 2 + 1, half_image_side * 2 + 1)
        extended_image = np.zeros((extended_size[0], extended_size[1], raw_image.shape[2]))
        extended_mask = np.zeros(extended_size)
        
        # Center piece in extended canvas
        start_y = half_image_side - raw_image.shape[0] // 2
        start_x = half_image_side - raw_image.shape[1] // 2
        extended_image[
            start_y:start_y + raw_image.shape[0],
            start_x:start_x + raw_image.shape[1]
        ] = raw_image
        extended_mask[
            start_y:start_y + raw_mask.shape[0],
            start_x:start_x + raw_mask.shape[1]
        ] = raw_mask
        
        # Translate polygon
        translation = np.array([start_x, start_y])
        extended_polygon = transform(raw_polygon, lambda f: f + translation)
        
        # Recalculate center
        center = np.round(calculate_center_of_mass(extended_mask)).astype(int)
        
        # Crop around center
        cropped_image = extended_image[
            center[1]-half_image_side: center[1]+half_image_side+1,
            center[0]-half_image_side: center[0]+half_image_side+1, :
        ]
        cropped_mask = extended_mask[
            center[1]-half_image_side: center[1]+half_image_side+1,
            center[0]-half_image_side: center[0]+half_image_side+1
        ]
        polygon_translation = center - np.asarray([half_image_side, half_image_side])
        cropped_polygon = transform(extended_polygon, lambda f: f - polygon_translation)
        
        # Rescale to target size
        if all(x > 0 for x in new_size):
            new_size = new_size[0]
            cropped_image_size = cropped_image.shape[0]
            rescaling_factor = cropped_image_size / new_size
            
            cropped_image = resize(cropped_image, (new_size, new_size), anti_aliasing=True)
            cropped_mask = (resize(
                cropped_mask, (new_size, new_size),
                anti_aliasing=True, preserve_range=True
            ) > 0.5).astype(np.uint8)
            cropped_polygon = transform(
                cropped_polygon,
                lambda f: f * new_size / cropped_image_size
            )
        
        return cropped_image, cropped_mask, cropped_polygon, rescaling_factor

    def preprocess_and_center_irregular_piece(self, raw_image, method='np'):
        """
        Preprocess and center an irregular piece.
        
        Parameters
        ----------
        raw_image : np.ndarray
            Raw piece image with alpha channel or background
        method : str, optional
            Method for center of mass calculation (default: 'np')
        
        Notes
        -----
        This is a helper method for irregular piece processing. Currently incomplete
        in the original implementation.
        """
        bmask = extract_binary_mask(raw_image)
        polygon, enclosing_radius = extract_polygon(bmask, return_vals=True)
        cm = calculate_center_of_mass(bmask, method=method)
        cen_image, cen_bmask, cen_polygon, shift = Puzzle.center_piece(raw_image, bmask, polygon, cm)  
       
        # plt.subplot(131)
        # plt.imshow(raw_image)
        # plt.scatter(raw_image.shape[0] / 2, raw_image.shape[1] / 2, s=15)
        # plt.plot(*(polygon.boundary.xy), linewidth=3)

    def extract_mask_and_polygon_irregular_piece(self, raw_image):
        """
        Extract mask and polygon from an irregular piece image.
        
        Parameters
        ----------
        raw_image : np.ndarray
            Raw piece image
        
        Returns
        -------
        bmask : np.ndarray
            Binary mask
        polygon : shapely.Polygon
            Piece polygon
        enclosing_radius : float
            Bounding box half-diagonal
        wh : list of int
            [height, width] of bounding box
        """
        bmask = extract_binary_mask(raw_image)
        polygon, enclosing_radius, wh = extract_polygon(bmask, return_vals=True)
        return bmask, polygon, enclosing_radius, wh

    @staticmethod
    def center_piece(raw_image, bmask, polygon, cm):
        """
        Center a piece by translating image, mask, and polygon.
        
        This static method shifts a piece so its center of mass is at the
        canvas center. Handles all four quadrant cases for the shift direction.
        
        Parameters
        ----------
        raw_image : np.ndarray
            Original piece image
        bmask : np.ndarray
            Original binary mask
        polygon : shapely.Polygon
            Original polygon
        cm : list of float
            [x, y] center of mass coordinates
        
        Returns
        -------
        cen_image : np.ndarray
            Centered image
        cen_bmask : np.ndarray
            Centered mask
        cen_polygon : shapely.Polygon
            Centered polygon
        shift : np.ndarray
            Applied shift [shift_x, shift_y]
        
        Notes
        -----
        The method creates a new canvas of the same size and shifts the content
        to center the piece. Parts that would fall outside the canvas are cropped.
        """
        # Create empty centered canvases
        cen_image = np.zeros_like(raw_image)
        cen_bmask = np.zeros_like(bmask)
        
        # Calculate required shift to center the piece
        half_image_side = np.round(raw_image.shape[0]/2).astype(int)
        center_pos = [half_image_side, half_image_side]
        shift_x, shift_y = -np.round(np.array(cm) - center_pos).astype(int)
        
        # Handle case of no shift needed
        if shift_x == 0 and shift_y == 0:
            cen_image = raw_image
            cen_bmask = bmask
        
        # Handle four quadrants of shift direction
        if shift_x >= 0 and shift_y >= 0:
            # Shift right and down
            cen_image[shift_y:, shift_x:] = raw_image[
                :raw_image.shape[0]-shift_y, :raw_image.shape[1]-shift_x
            ]
            cen_bmask[shift_y:, shift_x:] = bmask[
                :bmask.shape[0]-shift_y, :bmask.shape[1]-shift_x
            ]
        elif shift_x >= 0 and shift_y < 0:
            # Shift right and up
            cen_image[:shift_y, shift_x:] = raw_image[
                -shift_y:, :raw_image.shape[1]-shift_x
            ]
            cen_bmask[:shift_y, shift_x:] = bmask[
                -shift_y:, :bmask.shape[1]-shift_x
            ]
        elif shift_x < 0 and shift_y >= 0:
            # Shift left and down
            cen_image[shift_y:, :shift_x] = raw_image[
                :raw_image.shape[0]-shift_y, -shift_x:
            ]
            cen_bmask[shift_y:, :shift_x] = bmask[
                :bmask.shape[0]-shift_y, -shift_x:
            ]
        elif shift_x < 0 and shift_y < 0:
            # Shift left and up
            cen_image[:shift_y, :shift_x] = raw_image[-shift_y:, -shift_x:]
            cen_bmask[:shift_y, :shift_x] = bmask[-shift_y:, -shift_x:]

        # Translate polygon coordinates by the shift amount
        cen_polygon = transform(polygon, lambda f: f + [+shift_x,+shift_y])

        return cen_image, cen_bmask, cen_polygon, np.asarray([shift_x, shift_y])

    def save(self):
        """
        Save all puzzle data to disk.
        
        Saves the following files to the output directory:
        - images/ : PNG files of all pieces
        - binary_masks/ : PNG files of piece masks (if save_masks=True)
        - polygons/ : NPY files of piece polygons (if save_polygons=True)
        - ground_truth.json : Ground truth information for puzzle assembly
        - puzzle_info.json : Metadata about puzzle configuration
        
        Notes
        -----
        The output structure depends on input_type. For 'repair'/'json' inputs,
        preserves original naming. For 'image' inputs, uses piece_XXXX naming.
        """
        print("saving..")
        
        # Create output subdirectories
        images_out_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(images_out_dir, exist_ok=True)
        
        if self.save_masks:
            bmasks_out_dir = os.path.join(self.output_dir, 'binary_masks')
            os.makedirs(bmasks_out_dir, exist_ok=True)
        
        if self.save_polygons:
            polygons_out_dir = os.path.join(self.output_dir, 'polygons')
            os.makedirs(polygons_out_dir, exist_ok=True)
        
        # Save pieces based on input type
        if self.input_type == 'repair' or self.input_type == 'json':
            # Preserve original fragment naming
            for frag_key in self.input_data.keys():
                frag_data = self.input_data[frag_key]

                plt.imsave(
                    os.path.join(images_out_dir, f"{frag_data['idx']}_{frag_data['name']}.png"),
                    frag_data['image']
                )
                if self.save_masks:
                    cv2.imwrite(
                        os.path.join(bmasks_out_dir, f"{frag_data['idx']}_{frag_data['name']}.png"),
                        frag_data['mask']
                    )
                if self.save_polygons:
                    np.save(
                        os.path.join(polygons_out_dir, f"{frag_data['idx']}_{frag_data['name'][:-4]}"),
                        frag_data['polygon']
                    )
        else:
            # Save with piece_XXXX naming
            for p_name in self.pieces.keys():
                piece = self.pieces[p_name]
                
                # Clip values to [0,1] range before saving
                plt.imsave(
                    os.path.join(images_out_dir, f"{p_name}.png"),
                    np.clip(piece['squared_image'], 0, 1)
                )
                
                if self.save_masks:
                    cv2.imwrite(
                        os.path.join(bmasks_out_dir, f"{p_name}.png"),
                        piece['squared_mask']
                    )
                
                if self.save_polygons:
                    np.save(
                        os.path.join(polygons_out_dir, f"{p_name}"),
                        piece['squared_polygon']
                    )

        # Save ground truth JSON
        with open(os.path.join(self.output_dir, "ground_truth.json"), 'w') as jf:
            json.dump(self.gt, jf, indent=2)
        
        # Save puzzle info JSON
        with open(os.path.join(self.output_dir, "puzzle_info.json"), 'w') as jf:
            json.dump(self.puzzle_info, jf, indent=2)
        
        print("Done!")

    def show_piece(self, index: int = 0):
        """
        Display a puzzle piece with its image and binary mask.
        
        Parameters
        ----------
        index : int, optional
            Index of piece to display (default: 0)
        
        Notes
        -----
        Creates a matplotlib figure with two subplots showing the RGB image
        and binary mask, both overlaid with the piece polygon boundary.
        """
        plt.suptitle(f"Piece {index}: {self.names[index]}", fontsize=32)
        
        # Show RGB image with polygon
        plt.subplot(121)
        plt.title("Image")
        plt.imshow(self.images[index])
        plt.plot(*(self.polygons[index].boundary.xy), linewidth=3)
        
        # Show binary mask with polygon
        plt.subplot(122)
        plt.title("Binary Mask")
        plt.imshow(self.masks[index])
        plt.plot(*(self.polygons[index].boundary.xy), linewidth=3)
        
        plt.show()

    # def create_pieces(self):
    #     # puzzle_data, pieces, solution = 
    #     return 1,1,1
