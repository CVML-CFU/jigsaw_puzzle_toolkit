"""
Jigsaw Puzzle Generator Module

This module provides the core puzzle piece generation functionality for creating
jigsaw puzzles from images. It supports multiple piece types (squared, irregular,
polyomino, pattern-based) and handles piece extraction, rotation, and adjacency
calculation.

Main Classes
------------
Vector : class
    2D vector class for spatial calculations
PuzzleGenerator : class
    Main class for generating puzzle pieces from images

Utility Functions
-----------------
check_outside : Check if coordinates are outside image bounds
clip_rect : Clip coordinates to image boundaries
new_array : Create nested arrays initialized with a value
get_cm : Calculate center of mass from binary mask
get_polygon : Extract polygon contour from binary image
crop_extrapolated : Crop extrapolated piece image to bounding box
"""

import os
import math
import json
import random
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, ndimage
from glob import glob
import pdb
import cv2
import shapely
from math import hypot

############################
############################
# Utility methods
def check_outside(x, y, width, height):
    """
    Check if pixel coordinates are outside image boundaries.
    
    Parameters
    ----------
    x : int
        X coordinate (column)
    y : int
        Y coordinate (row)
    width : int
        Image width
    height : int
        Image height
    
    Returns
    -------
    bool
        True if coordinates are outside bounds, False otherwise
    """
    if x < 0 or x >= width or y < 0 or y >= height:
        return True
    else:
        return False

def clip_rect(x, y, width, height):
    """
    Clip coordinates to fit within image boundaries.
    
    Parameters
    ----------
    x : int
        X coordinate to clip
    y : int
        Y coordinate to clip
    width : int
        Image width
    height : int
        Image height
    
    Returns
    -------
    x_new : int
        Clipped X coordinate
    y_new : int
        Clipped Y coordinate
    """
    x_new = max(0, min(x, width-1))
    y_new = max(0, min(y, height-1))
    return x_new, y_new

def new_array(dims, val):
    """
    Create a nested array structure initialized with a specific value.
    
    Parameters
    ----------
    dims : int, tuple, or list
        Dimensions of the array to create
    val : any
        Value to initialize all elements with
    
    Returns
    -------
    list
        Nested list structure with specified dimensions
    
    Notes
    -----
    This is a helper function for creating multi-dimensional arrays
    without numpy, used primarily for mask initialization.
    """
    assert(type(dims) is int or type(dims) is tuple or type(dims) is list)
    if type(dims) is int:
        return [val for i in range(dims)]
    elif len(dims) == 1:
        return [val for i in range(dims[0])]
    else:
        return [new_array(dims[1:], val) for i in range(dims[0]) ]

def get_cm(mask):
    """
    Calculate center of mass from a binary mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask where values >= 0.5 indicate foreground
    
    Returns
    -------
    list of float
        [center_x, center_y] coordinates of the center of mass
    
    Notes
    -----
    This is a simplified version used specifically in the puzzle generator.
    Returns coordinates in [x, y] format (column, row).
    """
    # Find all foreground pixels
    mass_y, mass_x = np.where(mask >= 0.5)
    # Calculate average position
    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)
    return [cent_x, cent_y]

def get_polygon(binary_image):
    """
    Extract polygon contour from a binary image.
    
    Parameters
    ----------
    binary_image : np.ndarray
        Binary image with foreground/background
    
    Returns
    -------
    shapely.Polygon
        Simplified polygon representing the object boundary
    
    Raises
    ------
    ValueError
        If fewer than 4 contour points are found
    
    Notes
    -----
    - Uses morphological dilation before contour extraction
    - Applies 0.5 pixel offset to align with pixel centers
    - Simplifies the polygon while preserving topology
    """
    bin_img = binary_image.copy()
    # Dilate to get better edge definition
    bin_img = cv2.dilate(bin_img.astype(np.uint8), np.ones((2,2)), iterations=1)
    
    # Find contours using full chain approximation
    contours, _ = cv2.findContours(bin_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_points = contours[0]
    
    # Convert to Shapely format with 0.5 pixel offset
    shapely_points = [(point[0][0]-0.5, point[0][1]-0.5) for point in contour_points]
    
    # Validate minimum points for polygon
    if len(shapely_points) < 4:
        print('we have a problem, too few points', shapely_points)
        raise ValueError('\nWe have fewer than 4 points on the polygon, so we cannot create a Shapely polygon out of this points! Maybe something went wrong with the mask?')
    
    # Create and simplify polygon
    polygon = shapely.Polygon(shapely_points).simplify(0, preserve_topology=True)
    return polygon

##############################
##############################
class Vector:
    """
    Simple 2D vector class for spatial operations.
    
    Used primarily for direction vectors in region-filling algorithms
    during piece generation.
    
    Parameters
    ----------
    x : float, optional
        X component of the vector (default: 0)
    y : float, optional
        Y component of the vector (default: 0)
    
    Attributes
    ----------
    x : float
        X component
    y : float
        Y component
    
    Examples
    --------
    >>> v1 = Vector(3, 4)
    >>> print(abs(v1))  # Magnitude
    5.0
    >>> v2 = Vector(1, 0)
    >>> v3 = v1 + v2
    >>> print(v3)
    Vector(4, 4)
    """
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return "Vector(%r, %r)" % (self.x, self.y)
    
    def __abs__(self):
        """Calculate vector magnitude (Euclidean norm)."""
        return hypot(self.x, self.y)
    
    def __bool__(self):
        """Vector is truthy if it has non-zero magnitude."""
        return bool(abs(self))
    
    def __add__(self, other):
        """Add two vectors component-wise."""
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)
    
    def __mul__(self, scalar):
        """Multiply vector by a scalar."""
        return Vector(self.x * scalar, self.y * scalar)



###################################################################################
#                                                                                 #
#  ██████╗ ██╗   ██╗███████╗███████╗██╗     ███████╗                              #
#  ██╔══██╗██║   ██║╚══███╔╝╚══███╔╝██║     ██╔════╝                              #
#  ██████╔╝██║   ██║  ███╔╝   ███╔╝ ██║     █████╗                                #
#  ██╔═══╝ ██║   ██║ ███╔╝   ███╔╝  ██║     ██╔══╝                                #
#  ██║     ╚██████╔╝███████╗███████╗███████╗███████╗                              #
#  ╚═╝      ╚═════╝ ╚══════╝╚══════╝╚══════╝╚══════╝                              #
#                                                                                 #
#   ██████╗ ███████╗███╗   ██╗███████╗██████╗  █████╗ ████████╗ ██████╗ ██████╗   #
#  ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗  #
#  ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ██████╔╝███████║   ██║   ██║   ██║██████╔╝  #
#  ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ██╔══██╗██╔══██║   ██║   ██║   ██║██╔══██╗  #
#  ╚██████╔╝███████╗██║ ╚████║███████╗██║  ██║██║  ██║   ██║   ╚██████╔╝██║  ██║  #
#   ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝  #
#                                                                                 #
###################################################################################
class PuzzleGenerator:
    """
    Generate jigsaw puzzle pieces from an image.
    
    This class handles the complete process of cutting an image into puzzle pieces,
    including:
    - Creating cutting masks (smooth or segmented curves)
    - Extracting individual piece regions
    - Calculating piece adjacency relationships
    - Handling different piece types (irregular, polyomino, pattern-based)
    - Applying rotations to pieces
    - Saving puzzle data and ground truth
    
    Parameters
    ----------
    img : np.ndarray
        Input image with floating values between 0 and 1, shape (H, W, C)
    parameters : dict
        Configuration dictionary containing:
        - 'name' : str, puzzle name (default: 'no_name')
        - 'padding' : int, padding around pieces in pixels (default: 9)
        - 'rotation_range' : float, max rotation angle in degrees (default: 180)
        - 'rotation_type' : int, 1=none, 2=90° only, 3=free (default: 1)
        - 'rotation_type_s' : str, rotation description (default: 'no rotation')
        - 'pieces_type' : str, 'S', 'P', 'M', or 'I' (default: 'S')
        - 'pieces_type_s' : str, piece type description (default: 'squared')
        - 'curves_type' : str, 'smooth' or 'segment' (default: 'smooth')
    pieces_centers : dict, optional
        Pre-defined piece centers for polyomino/pattern-based puzzles
    
    Attributes
    ----------
    img : np.ndarray
        Input image
    img_size : tuple
        (Height, Width, Channels) of input image
    aspect_ratio : float
        Height/Width ratio
    pieces : dict
        Dictionary of extracted piece data
    gt : dict
        Ground truth information including piece positions and adjacency
    region_mat : np.ndarray
        Matrix mapping each pixel to its piece index
    region_cnt : int
        Number of pieces generated
    
    Examples
    --------
    >>> img = plt.imread('input.jpg')
    >>> params = {'name': 'my_puzzle', 'rotation_type': 2, 'pieces_type': 'I'}
    >>> gen = PuzzleGenerator(img, params)
    >>> gen.run(piece_n=25, smooth_flag=True)
    >>> pieces, size, gt = gen.extract_pieces()
    
    Notes
    -----
    The generator uses different algorithms based on pieces_type:
    - 'S': Regular grid cutting
    - 'I': Irregular pieces with smooth/segmented curves
    - 'P': Polyomino-based pieces
    - 'M': Pattern map-based pieces
    """

    def __init__(self, img, parameters:dict, pieces_centers=None):
        """
        Initialize the puzzle generator with an image and parameters.
        
        Sets up the generator with image data and configuration. Prepares
        kernels for morphological operations and adjacency detection.
        """
        # Store input image (should have values 0-1)
        self.img = img
        self.img_size = self.img.shape[:2]  # (Height, Width)
        self.aspect_ratio = self.img_size[0] / self.img_size[1]
        
        # Morphological operation kernels
        self.erosion_kernel_size = 7
        self.dilation_kernel_size = 11
        self.dilation_kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size))
        
        # Adjacency detection threshold (in pixels)
        self.minimum_overlap_for_adjacency = 300

        # Extract configuration parameters
        self.name = parameters.get('name', "no_name")
        self.padding = parameters.get('padding', 9)
        self.rotation_range = parameters.get('rotation_range', 180)  
        self.rotation_type = parameters.get('rotation_type', 1)  
        self.rotation_type_description = parameters.get('rotation_type_s', "no rotation")  
        self.pieces_type = parameters.get('pieces_type', "S")  
        self.pieces_type_description = parameters.get('pieces_type_s', "squared")  
        self.curves_type = parameters.get('curves_type', "smooth") 
        
        # Determine if curves should be smooth or segmented
        if self.curves_type == 'segment':
            self.smooth_flag = False  # Use linear segments instead of curves
        
        # For polyomino/pattern-based puzzles, skip first region (background)
        self.start_from = 0
        if self.pieces_type == "M" or self.pieces_type == "P":
            self.start_from = 1
        
        # Store pre-defined centers if provided (for polyomino pieces)
        if pieces_centers is not None:
            self.pieces_centers = pieces_centers


    def get_smooth_curve(self, x_len, x_pt_n, x_offset, y_offset, x_step):
        """
        Generate a smooth or segmented curve for puzzle piece cutting.
        
        Creates a curve by placing random points and interpolating between them.
        Used for creating irregular piece boundaries.
        
        Parameters
        ----------
        x_len : int
            Length of the curve in pixels
        x_pt_n : int
            Number of control points to place along the curve
        x_offset : float
            Maximum random offset for x-coordinates of control points
        y_offset : float
            Maximum random offset for y-coordinates of control points
        x_step : float
            Expected spacing between control points
        
        Returns
        -------
        x_arr : np.ndarray
            X coordinates of the curve (length x_len)
        y_arr : np.ndarray
            Y coordinates of the curve (length x_len)
        
        Notes
        -----
        The interpolation method depends on self.smooth_flag:
        - smooth_flag=True: Uses cubic/quadratic/linear spline based on point count
        - smooth_flag=False: Uses linear segments (for segmented cuts)
        """

        x_arr = []
        y_arr = []

        # Generate random control points along the curve
        for i in range(x_pt_n+1):

            if i == 0:
                # First point at the start
                x = 0
            elif i == x_pt_n:
                # Last point at the end
                x = x_len - 1
            else:
                # Intermediate points with random offset
                x = round(x_step * i + random.uniform(-x_offset, x_offset))
            
            # Y coordinate with random offset
            y = round(random.uniform(-y_offset, y_offset))

            x_arr.append(x)
            y_arr.append(y)

        # Remove duplicate x values and sort
        x_arr = list(set(x_arr))
        y_arr = y_arr[:len(x_arr)]
        x_arr.sort()

        # Choose interpolation method based on smooth_flag
        if self.smooth_flag:
            # Use higher-order interpolation for smooth curves
            if len(x_arr) >= 4:
                smooth_func = interpolate.interp1d(x_arr, y_arr, kind='cubic')
            elif len(x_arr) == 3:
                smooth_func = interpolate.interp1d(x_arr, y_arr, kind='quadratic')
            elif len(x_arr) == 2:
                smooth_func = interpolate.interp1d(x_arr, y_arr, kind='slinear')
            else:
                raise ValueError("The length of cutting points in x_arr must be larger than 0.")

        else:
            # Use linear segments for segmented cuts
            smooth_func = interpolate.interp1d(x_arr, y_arr, kind='linear')

        # Interpolate to get full-resolution curve
        x_arr = np.arange(0, x_len, dtype=np.int32)
        y_arr = smooth_func(x_arr).astype(np.int32)

        return x_arr, y_arr


    def get_mask(self, offset_rate_h, offset_rate_w):

        piece_h = self.img_size[0] / self.h_n
        piece_w = self.img_size[1] / self.w_n

        offset_h = piece_h * offset_rate_h
        offset_w = piece_w * offset_rate_w

        self.mask = new_array(self.img_size, 0)

        # Vertical cuts
        for i in range(1, self.w_n):

            x_arr, y_arr = self.get_smooth_curve(self.img_size[0], self.h_n, offset_h, offset_w, piece_h)
            y_arr = y_arr + round(i * piece_w)
            y_arr = np.clip(y_arr, 0, self.img_size[1] - 1)

            for j in range(self.img_size[0]):
                self.mask[x_arr[j]][y_arr[j]] = 255
                if j > 0:
                    st = min(y_arr[j - 1], y_arr[j])
                    ed = max(y_arr[j - 1], y_arr[j])
                    for k in range(st, ed + 1):
                        self.mask[x_arr[j]][k] = 255

        # Horizontal cuts
        for i in range(1, self.h_n):

            x_arr, y_arr = self.get_smooth_curve(self.img_size[1], self.w_n, offset_w, offset_h, piece_w)
            y_arr = y_arr + round(i * piece_h)
            y_arr = np.clip(y_arr, 0, self.img_size[0] - 1)

            for j in range(self.img_size[1]):
                self.mask[y_arr[j]][x_arr[j]] = 255
                if j > 0:
                    st = min(y_arr[j - 1], y_arr[j])
                    ed = max(y_arr[j - 1], y_arr[j])
                    for k in range(st, ed + 1):
                        self.mask[k][x_arr[j]] = 255

        cv2.imwrite('tmp/mask_init.png', np.array(self.mask, dtype=np.uint8))
        # cv2.imshow('mask', self.mask)
        # cv2.waitKey()

    def get_regions(self):

        dirs = [Vector(0,-1), Vector(0, 1), Vector(-1, 0), Vector(1, 0)] # (x, y)
        small_region_area_limit = self.small_region_area_ratio * \
            self.img_size[0] * self.img_size[1] / (self.w_n * self.h_n)

        mask = np.invert(np.array(self.mask, dtype=np.uint8))

        self.region_cnt, self.region_mat, stats, centroids = \
            cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_32S)
        stats = stats.tolist()

        # Remap region idx
        region_idx_map = -1 * np.ones(self.region_cnt, dtype=np.int32)
        region_new_cnt = 0

        for i in range(1, self.region_cnt):
            if stats[i][4] < small_region_area_limit:
                region_idx_map[i] = -1
            else:
                region_idx_map[i] = region_new_cnt
                region_new_cnt += 1

        self.region_mat = region_idx_map[self.region_mat]
        #print('\tRegion cnt final (raw): %d (%d)' % (region_new_cnt, self.region_cnt - 1))
        self.region_cnt = region_new_cnt

        if self.erosion == 0:
            # Expand valid region to fill out the canvas
            bg_pts = np.transpose(np.nonzero(self.region_mat == -1)).tolist()
            # self.region_mat = self.region_mat.tolist()
            self.region_list = self.region_mat.tolist()
            que = []

            for bg_pt in bg_pts:
                cur_p = Vector(bg_pt[1], bg_pt[0])
                for dir in dirs:
                    next_p = cur_p + dir
                    if check_outside(next_p.x, next_p.y, self.img_size[1], self.img_size[0]) or \
                        self.region_list[next_p.y][next_p.x] == -1:
                        continue
                    que.append(next_p)

            while len(que) > 0:
                cur_p = que.pop(0)
                for dir in dirs:
                    next_p = cur_p + dir
                    if check_outside(next_p.x, next_p.y, self.img_size[1], self.img_size[0]) or \
                        self.region_list[next_p.y][next_p.x] != -1:
                        continue
                    self.region_list[next_p.y][next_p.x] = self.region_list[cur_p.y][cur_p.x]
                    que.append(next_p)

            # Check the region mat
            unlabel_pts = np.transpose(np.nonzero(np.ma.masked_equal(self.region_list, -1).mask))
            assert(unlabel_pts.size == 0)

        else: #if self.erosion > 0:
            # pdb.set_trace()
            eroded_region_mat = np.ones_like(self.region_mat) * -1
            for reg_val in range(self.region_cnt): # in np.unique(self.region_mat):
                cur_reg = self.region_mat == reg_val
                # plt.subplot(121)
                # plt.imshow(cur_reg)
                erosion_kernel = np.random.rand(self.erosion_kernel_size, self.erosion_kernel_size)
                eroded_reg = cv2.erode(cur_reg.astype(np.uint8), erosion_kernel, iterations=1)
                eroded_region_mat += eroded_reg * (reg_val+1) # +1 because we start from -1 (see line 188)
                # plt.subplot(122)
                # plt.imshow(eroded_reg)
                # plt.show()
                # pdb.set_trace()
            self.region_mat = eroded_region_mat
            self.region_list = self.region_mat.tolist()
            # TODO
            # if self.erosion == 1:
            #     #

            # elif self.erosion == 2:
            #     #
            # elif self.erosion == 3:
            #     #
            # else:
            #     print('not done yet')


        # for i in range(self.region_cnt):
        #     mask = np.ma.masked_equal(self.region_mat, i).mask.astype(np.uint8)
        #     mask = mask * 255
        #     cv2.imwrite('tmp/' + str(i) + '.png', mask)
        #     cv2.imshow('tmp', mask)
        #     cv2.waitKey(0)


    def save_raw_regions(self, iter):

        file_path = os.path.join(self.raw_regions, '%d.npy' % iter)
        file_path_mat = os.path.join(self.raw_regions, '%d_mat.npy' % iter)
        np.save(file_path, np.array(self.region_list, dtype=np.int32))
        np.save(file_path_mat, self.region_mat)

        f = open(file_path[:-3] + 'txt', 'w')
        f.write(str(self.region_cnt))
        f.close()
        print('\tSave to %s & %d.txt' % (file_path, iter))

    def extrapolate_regions(self, extr_pixels=5, return_vals=False, start_from=0):
        self.pieces = []
        self.extr_pieces = []
        dilation_kernel = np.ones((extr_pixels * 2 + 1, extr_pixels * 2 + 1))
        for reg_val in range(start_from, self.region_cnt): # in np.unique(self.region_mat):
            cur_reg = self.region_mat == reg_val
            dilated_reg = cv2.dilate(cur_reg.astype(np.uint8), dilation_kernel, iterations=1)
            rgba_ex = cv2.cvtColor(self.img, cv2.COLOR_RGB2RGBA)
            rgba_ex[:, :, 3] = 255*(dilated_reg)
            rgba = cv2.cvtColor(self.img, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = 255*(cur_reg)
            self.pieces.append(rgba)
            self.extr_pieces.append(rgba_ex)
        if return_vals == True:
            return pieces, extr_pieces

    def save_extrapolated_regions(self, extrap_folder=''):
        
        if self.extr_pieces is None:
            extrapolate_regions(self, extr_pixels=5, return_vals=False)
        for j in range(len(self.extr_pieces)):
            rgba_ex_cropped, x0, x1, y0, y1 = crop_extrapolated(self.extr_pieces[j], padding=0, return_vals=True)
            rgba_cropped = self.pieces[j][y0:y1, x0:x1, :]
            path_for_matlab = f"{extrap_folder.split('/')[-3]}_{extrap_folder.split('/')[-2]}"
            cv2.imwrite(os.path.join(extrap_folder, f'{path_for_matlab}_piece_p{j:04d}_v1_ext.png'), rgba_ex_cropped)
            cv2.imwrite(os.path.join(extrap_folder, f'{path_for_matlab}_piece_p{j:04d}_v1.png'), rgba_cropped)
        # for reg_val in range(self.region_cnt): # in np.unique(self.region_mat):
        #     cur_reg = self.region_mat == reg_val
        #     dilation_kernel = np.random.rand(self.dilation_kernel_size, self.dilation_kernel_size)
        #     dilated_reg = cv2.dilate(cur_reg.astype(np.uint8), dilation_kernel, iterations=1)
        #     #dilated_frag = self.img * np.dstack((dilated_reg,dilated_reg,dilated_reg))
        #     rgba_ex = cv2.cvtColor(self.img, cv2.COLOR_RGB2RGBA)
        #     rgba_ex[:, :, 3] = 255*(dilated_reg)
        #     rgba = cv2.cvtColor(self.img, cv2.COLOR_RGB2RGBA)
        #     rgba[:, :, 3] = 255*(cur_reg)

        #     rgba_ex_cropped, x0, x1, y0, y1 = crop_extrapolated(rgba_ex, padding=0, return_vals=True)
        #     rgba_cropped = rgba[y0:y1, x0:x1, :]

        #     # cv2.imwrite(os.path.join(extrap_folder, f'series_p-{reg_val}_v1_ex.png'), rgba_ex)
        #     # cv2.imwrite(os.path.join(extrap_folder, f'series_p-{reg_val}_v1.png'), rgba)
        #     cv2.imwrite(os.path.join(extrap_folder, f'series_p-{reg_val}_v2_ex.png'), rgba_ex_cropped)
        #     cv2.imwrite(os.path.join(extrap_folder, f'series_p-{reg_val}_v2.png'), rgba_cropped)
        #     #print(os.path.join(extrap_folder, f'piece-{reg_val}.png'))

    def save_jpg_regions(self, folder_path, skip_bg=False):
        regions_path = os.path.join(folder_path, 'regions')
        os.makedirs(regions_path, exist_ok=True)
        cv2.imwrite(os.path.join(regions_path, 'regions_uint8.png'), self.region_mat)
        # change to cmap='gray' for grayscale color coding
        plt.imsave(os.path.join(regions_path, 'regions_col_coded.jpg'), self.region_mat, cmap='jet')
        if skip_bg:
            puzzle_mask = (self.region_mat > 0).astype(np.uint8)
        else:
            puzzle_mask = (self.region_mat + 1).astype(np.uint8)
        puzzle_mask = cv2.dilate(puzzle_mask, np.ones((5,5)))
        cv2.imwrite(os.path.join(regions_path, 'orig_image_cut.jpg'), np.round(self.img * 255).astype(np.uint8))
        
        # if len(self.img.shape) == len(puzzle_mask.shape):
        #     cut_puzzle_img = puzzle_mask * self.img
        #     breakpoint()
        #     if np.max(cut_puzzle_img) < 2:
        #         cut_puzzle_img = cut_puzzle_img * 255 / np.max(cut_puzzle_img)
        #     cv2.imwrite(os.path.join(regions_path, 'orig_image_cut.jpg'), np.round(cut_puzzle_img).astype(np.uint8)) #, cmap='gray')
        # else:
        #     if skip_bg == True:
        #         puzzle_mask = (puzzle_mask > 0).astype(int)
        #     else:
        #         puzzle_mask = (puzzle_mask > -1).astype(int)
            
        #     puzzle_mask3c = np.repeat(puzzle_mask, self.img.shape[2]).reshape(self.img.shape)
        #     cut_puzzle_img = (puzzle_mask3c * self.img).astype(np.uint8)
        #     plt.imsave(os.path.join(regions_path, 'orig_image_cut.jpg'), cut_puzzle_img)

    def get_pieces_from_puzzle_v2(self, start_from=0):
        """
        Get the pieces from a `generated` puzzle. It does not handle rotation at the moment. 
        Latest version @extract_pieces
        ---------
        2025 / 11
        """
        print("\nWARNING: DEPRECATED")
        print("This is an older version of the code, the new method is called: `extract_pieces`, please use that for best results\n")
        pieces = []
        bg_mat = np.zeros_like(self.img)
        h_max = 0
        w_max = 0
        dist_cm_max = 0
        padding = 3 #np.min(self.img.shape[:2]) // 30
        for i in range(start_from, self.region_cnt):
            mask_i = self.region_mat == i
            if len(self.img.shape) > 2: 
                image_i = self.img * np.repeat(mask_i, self.img.shape[2]).reshape(self.img.shape)
            else:
                image_i = np.where(mask_i, self.img, bg_mat)
            poly_i = get_polygon(mask_i)
            cm_i = get_cm(mask_i)[::-1]
            coords = np.argwhere(mask_i)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            h_i = y1-y0 
            w_i = x1-x0 
            
            dists_from_cm = np.linalg.norm(np.array(cm_i[::-1]) - np.array(poly_i.exterior.coords[:]), axis=1)
            if np.max(dists_from_cm) > dist_cm_max:
                dist_cm_max = np.max(dists_from_cm)
            if h_i > h_max:
                h_max = h_i 
            if w_i > w_max:
                w_max = w_i     

            ## centering
            centered_img = np.zeros_like(self.img)
            centered_mask = np.zeros_like(mask_i)
            center_i = np.asarray([self.img.shape[0] / 2, self.img.shape[1] / 2])
            shift2center = (center_i - cm_i)#[::1]
            x0c = np.round(x0+shift2center[1]).astype(int)
            x1c = np.round(x0c + w_i).astype(int)
            y0c = np.round(y0+shift2center[0]).astype(int)
            y1c = np.round(y0c + h_i).astype(int)
            centered_img[y0c:y1c, x0c:x1c] = image_i[y0:y1, x0:x1]
            centered_mask[y0c:y1c, x0c:x1c] = mask_i[y0:y1, x0:x1]
            centered_poly = get_polygon(centered_mask)
            pieces.append({
                'mask': mask_i,
                'centered_mask': centered_mask,
                'image': image_i,
                'centered_image': centered_img,
                'polygon': poly_i,
                'centered_polygon': centered_poly,
                'center_of_mass': cm_i,
                'height': h_i,
                'width': w_i,
                'shift2center': shift2center
            })

        # put pieces inside a square 
        diam_dist_cm = int(dist_cm_max * 2)
        sq_size = max(h_max, w_max, diam_dist_cm) + padding
        # it should always be dist_cm_max which is the maximum radius from the center of mass 
        # and is the radius of the circle where the piece can be included. Using this as the 
        # size of the image guarantees that the piece does not go out of the square even during rotation
        if sq_size % 2 > 0:
            sq_size += 1 # keep square size even! :)
        hsq = sq_size // 2
        # remember center ordering!
        from_idx = np.round(center_i-hsq).astype(int)
        to_idx = np.round(center_i+hsq).astype(int)
        for i in range(len(pieces)):
            squared_img = np.zeros((sq_size, sq_size, 3))
            squared_img = pieces[i]['centered_image'][from_idx[0]:to_idx[0], from_idx[1]:to_idx[1]]
            squared_mask = pieces[i]['centered_mask'][from_idx[0]:to_idx[0], from_idx[1]:to_idx[1]]
            # we remove the offset in the centered polygon to get it aligned
            xoffset = - (self.img.shape[1]-sq_size) / 2   # half of the distance from the square to the shape of the image!
            yoffset = - (self.img.shape[0]-sq_size) / 2
            squared_poly = shapely.affinity.translate(pieces[i]['centered_polygon'], xoff=xoffset, yoff=yoffset)
            pieces[i]['squared_image'] = squared_img
            pieces[i]['squared_mask'] = squared_mask
            pieces[i]['squared_polygon'] = squared_poly
            pieces[i]['shift2square'] = np.asarray([xoffset, yoffset])

        return pieces, sq_size

    def extract_pieces(self):
        """
        Extracts the pieces from the generated puzzle (either generated regions, or patterns map / polyomino below, as it has different center properties).
        It also handles rotations (depending on puzzle_type) and writes down the ground truth information
        -------
        2026 / 02
        """
        self.pieces = {}
        self.gt = {
            'pieces': {},
            'adjacency': []
        } 
        bg_mat = np.zeros_like(self.img)
        h_max = 0
        w_max = 0
        dist_cm_max = 0
        
        for i in range(self.start_from, self.region_cnt):
            j = i - self.start_from # useful if you start from values > 0
            # 1. Extract the piece from the region
            piece_name = f"piece_{j:03d}"
            mask_i = self.region_mat == i
            # 1b. Calculate adjacency matrix
            for k in range(i+1, self.region_cnt):
                mask_k = self.region_mat == k
                overlap = np.sum(cv2.dilate(mask_i.astype(np.uint8), self.dilation_kernel) * cv2.dilate(mask_k.astype(np.uint8), self.dilation_kernel))
                if overlap > self.minimum_overlap_for_adjacency: 
                    self.gt['adjacency'].append([j, k])
                # else:
                # plt.subplot(131); plt.imshow(mask_i)
                # plt.subplot(132); plt.imshow(mask_k)
                # plt.subplot(133); plt.imshow(cv2.dilate(mask_i.astype(np.uint8), self.dilation_kernel) * cv2.dilate(mask_k.astype(np.uint8), self.dilation_kernel))
                # plt.suptitle(f"overlap: {overlap}, threshold: {self.minimum_overlap_for_adjacency}")
                # plt.show()
                # breakpoint()
            if len(self.img.shape) > 2: 
                image_i = self.img * np.repeat(mask_i, self.img.shape[2]).reshape(self.img.shape)
            else:
                image_i = np.where(mask_i, self.img, bg_mat)
            poly_i = get_polygon(mask_i)
            cm_i = get_cm(mask_i)[::-1]
            coords = np.argwhere(mask_i)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            h_i = y1-y0 
            w_i = x1-x0 
            dists_from_cm = np.linalg.norm(np.array(cm_i[::-1]) - np.array(poly_i.exterior.coords[:]), axis=1)
            if np.max(dists_from_cm) > dist_cm_max:
                dist_cm_max = np.max(dists_from_cm)
            if h_i > h_max:
                h_max = h_i 
            if w_i > w_max:
                w_max = w_i   
            ## 2. Centering based on the center of mass
            centered_img = np.zeros_like(self.img)
            centered_mask = np.zeros((self.img.shape[0], self.img.shape[1]))
            center_i = np.asarray([self.img.shape[0] / 2, self.img.shape[1] / 2])
            shift2center = (center_i - cm_i)#[::1]
            x0c = np.round(x0+shift2center[1]).astype(int)
            x1c = np.round(x0c + w_i).astype(int)
            y0c = np.round(y0+shift2center[0]).astype(int)
            y1c = np.round(y0c + h_i).astype(int)
            centered_img[y0c:y1c, x0c:x1c] = image_i[y0:y1, x0:x1]
            centered_mask[y0c:y1c, x0c:x1c] = mask_i[y0:y1, x0:x1]
            centered_poly = get_polygon(centered_mask)
            ## 3. pieces in the dict
            self.pieces[piece_name] = {
                'mask': mask_i,
                'centered_mask': centered_mask,
                'image': image_i,
                'centered_image': centered_img,
                'polygon': poly_i,
                'centered_polygon': centered_poly,
                'center_of_mass': cm_i,
                'height': h_i,
                'width': w_i,
                'shift2center': shift2center
            }
            self.gt['pieces'][j] = {
                'name': piece_name,
                'x': float(cm_i[1]),
                'y': float(cm_i[0]),
                'theta': 0
            }

        # 4. Squared version of the pieces 
        diam_dist_cm = int(dist_cm_max * 2)
        self.sq_size = max(h_max, w_max, diam_dist_cm) + self.padding
        # it should always be dist_cm_max which is the maximum radius from the center of mass 
        # and is the radius of the circle where the piece can be included. Using this as the 
        # size of the image guarantees that the piece does not go out of the square even during rotation
        if self.sq_size % 2 > 0:
            self.sq_size += 1 # keep square size even! :)
        hsq = self.sq_size // 2
        # remember center ordering!
        from_idx = np.round(center_i-hsq).astype(int)
        to_idx = np.round(center_i+hsq).astype(int)
        for j, p_name in enumerate(self.pieces.keys()):
            squared_img = np.zeros((self.sq_size, self.sq_size, 4))
            squared_img[:,:,:3] = self.pieces[p_name]['centered_image'][from_idx[0]:to_idx[0], from_idx[1]:to_idx[1], ::-1]
            squared_img[:,:,3] = np.sum(squared_img[:,:,:3], axis=2) > 0
            squared_mask = self.pieces[p_name]['centered_mask'][from_idx[0]:to_idx[0], from_idx[1]:to_idx[1]]
            # we remove the offset in the centered polygon to get it aligned
            xoffset = - (self.img.shape[1]-self.sq_size) / 2   # half of the distance from the square to the shape of the image!
            yoffset = - (self.img.shape[0]-self.sq_size) / 2
            squared_poly = shapely.affinity.translate(self.pieces[p_name]['centered_polygon'], xoff=xoffset, yoff=yoffset)
            # 5. handling rotations
            if self.rotation_type > 1:
                if self.rotation_type == 2: # 90 deg rotation
                    degrees = int(np.round(np.random.uniform(0, 3)) * 90)
                elif self.rotation_type == 3: # free deg rotation
                    degrees = random.uniform(-self.rotation_range, self.rotation_range)
                else:
                    print("unknown rotation type!")
                    raise NotImplementedError()                
                self.gt['pieces'][j]['theta'] = degrees
                squared_img, squared_mask, squared_poly = self.rotate_piece(squared_img, squared_mask, squared_poly, degrees, method='ND')

            self.pieces[p_name]['squared_image'] = squared_img
            self.pieces[p_name]['squared_mask'] = squared_mask
            self.pieces[p_name]['squared_polygon'] = squared_poly
            self.pieces[p_name]['shift2square'] = np.asarray([xoffset, yoffset])

        return self.pieces, self.sq_size, self.gt

    def get_polyomino_pieces_from_puzzle(self, parameters):
        """
        This is very similar to the `get_pieces_from_puzzle_v2` method, but it uses the centers 
        (which were previously set when initializing the puzzle generator object)
        and centers the pieces there. It's not in the center of mass of the piece,
        they are centered on one of the squares of the polyominos!
        ----
        2026 / 02
        """
        self.pieces = {}
        self.gt = {
            'pieces': {},
            'adjacency': []
        } 
        verbosity = parameters.get('verbosity', 0)
        square_side = self.img.shape[0] + parameters['monomino_square_size'] + 10 # padding to be sure
        if square_side % 2 == 0:
            square_side += 1
        bg_mat = np.zeros_like(self.img)
        h_max = 0
        w_max = 0
        dist_cm_max = 0
        for i in range(self.start_from, self.region_cnt):
            j = i - self.start_from
            # use keys!
            piece_name = f"piece_{j:03d}"
            mask_i = self.region_mat == i
            # 1b. Calculate adjacency matrix
            for k in range(i+1, self.region_cnt):
                mask_k = self.region_mat == k
                overlap = np.sum(cv2.dilate(mask_i.astype(np.uint8), self.dilation_kernel) * cv2.dilate(mask_k.astype(np.uint8), self.dilation_kernel))
                if overlap > self.minimum_overlap_for_adjacency: 
                    self.gt['adjacency'].append([j, k])
            if len(self.img.shape) > 2: 
                image_i = self.img * np.repeat(mask_i, self.img.shape[2]).reshape(self.img.shape)
            else:
                image_i = np.where(mask_i, self.img, bg_mat)
            poly_i = get_polygon(mask_i)
            cm_i = np.asarray(self.pieces_centers[f"{i}"][::-1])
            coords = np.argwhere(mask_i)
            y0, x0 = coords.min(axis=0) 
            y1, x1 = coords.max(axis=0)#  + 1
            h_i = y1-y0 
            w_i = x1-x0 
            # print(f"width: {w_i}, height: {h_i}")
            # plt.imshow(image_i);
            # plt.scatter(cm_i[1], cm_i[0])
            # plt.plot([x0, x1], [y0, y1]), plt.plot([x1, x0], [y0, y1])
            # plt.show()
            # breakpoint()
            
            dists_from_cm = np.linalg.norm(np.array(cm_i[::-1]) - np.array(poly_i.exterior.coords[:]), axis=1)
            if np.max(dists_from_cm) > dist_cm_max:
                dist_cm_max = np.max(dists_from_cm)
            if h_i > h_max:
                h_max = h_i 
            if w_i > w_max:
                w_max = w_i     

            ## 2. Centering based on the center of mass
            centered_img = np.zeros((square_side, square_side, 3))
            centered_mask = np.zeros((square_side, square_side))
            # center_i is the center of centered_img! (not the center of image_i!?)
            # center_i = np.asarray([self.img.shape[0] / 2, self.img.shape[1] / 2])
            center_i = np.asarray([(centered_img.shape[1]) / 2, (centered_img.shape[1]) / 2]) # + 1
            shift2center = (center_i - cm_i)#[::1]
            # print(f"s2c_{i}:{shift2center}, cm_{i}: {cm_i}, w_{i}:{w_i}, h_{i}:{h_i}")
            x0c = np.floor(x0+shift2center[1]+0.5).astype(int) 
            x1c = np.floor(x0c + w_i + 1+0.5).astype(int) 
            y0c = np.floor(y0+shift2center[0]+0.5).astype(int) 
            y1c = np.floor(y0c + h_i + 1+0.5).astype(int)
            ## NEW: calculate x0c and x1c from the center!
            # x0c = np.ceil(center_i[1] - (w_i / 2)).astype(int) 
            # x1c = np.ceil(center_i[1] + (w_i / 2)).astype(int) 
            # y0c = np.ceil(center_i[0] - (h_i / 2)).astype(int) 
            # y1c = np.ceil(center_i[0] + (h_i / 2)).astype(int)
            if verbosity > 3:
                print(f"Will extract image_i[{y0}:{y1}, {x0}:{x1}] with shape: {image_i[y0:y1+1, x0:x1+1].shape}")
                print(f"Will paste in image_i[{y0c}:{y1c}, {x0c}:{x1c}] (center in {center_i}, shape: {centered_img[y0c:y1c, x0c:x1c].shape})")
            try:
                centered_img[y0c:y1c, x0c:x1c] = image_i[y0:y1+1, x0:x1+1]
            except:
                breakpoint()
            centered_mask[y0c:y1c, x0c:x1c] = mask_i[y0:y1+1, x0:x1+1]
            centered_poly = get_polygon(centered_mask)
            centered_poly = shapely.affinity.translate(centered_poly, xoff=0, yoff=0)
            # plt.subplot(121); plt.imshow(image_i)
            # plt.plot(*poly_i.boundary.xy, c='red')
            # plt.subplot(122); plt.imshow(centered_img)
            # plt.plot(*centered_poly.boundary.xy, c='red')
            # plt.show()
            # breakpoint()
            ## 3. pieces in the dict
            self.pieces[piece_name] = {
                'mask': mask_i,
                'centered_mask': centered_mask,
                'image': image_i,
                'centered_image': centered_img,
                'polygon': poly_i,
                'centered_polygon': centered_poly,
                'center_of_mass': cm_i,
                'height': h_i,
                'width': w_i,
                'shift2center': shift2center,
                'x0c': x0c,
                'x1c': x1c,
                'y0c': y0c,
                'y1c': y1c
            }
            self.gt['pieces'][j] = {
                'name': piece_name,
                'x': float(cm_i[1]),
                'y': float(cm_i[0]),
                'theta': 0
            }
        # 4. Squared version of the pieces 
        diam_dist_cm = int(dist_cm_max * 2)
        self.sq_size = max(h_max, w_max, diam_dist_cm) + self.padding
        # it should always be dist_cm_max which is the maximum radius from the center of mass 
        # and is the radius of the circle where the piece can be included. Using this as the 
        # size of the image guarantees that the piece does not go out of the square even during rotation
        # if self.sq_size % 2 > 0:
        #     self.sq_size += 1 # keep square size even! :)
        if self.sq_size % 2 == 0:
            self.sq_size += 1 # keep square size odd! :)
        hsq = self.sq_size / 2
        # breakpoint()
        # remember center ordering!
        # center_of_piece = np.asarray([center_i-w_i/2, center_i-h_i/2])

        # this is the part of the centered_image that goes into the squared image (complete square)
        from_idx = np.floor(center_i-hsq).astype(int)
        to_idx = np.floor(center_i+hsq).astype(int)
        if verbosity > 3:
            print(f"from {from_idx} (rounded {center_i-hsq}) to {to_idx} (rounded {center_i+hsq})")

        for j, p_name in enumerate(self.pieces.keys()):
            if verbosity > 3:
                print(f"\npiece {p_name}")
            squared_img = np.zeros((self.sq_size, self.sq_size, 4))
            w_j = self.pieces[p_name]['width']
            h_j = self.pieces[p_name]['height']
            if verbosity > 3:
                print(f"ci: {center_i}, w: {w_j}, h: {h_j}")
            center_of_rotation = np.asarray([self.pieces[p_name]['center_of_mass'][1] + self.pieces[p_name]['shift2center'][1], \
                self.pieces[p_name]['center_of_mass'][0] + self.pieces[p_name]['shift2center'][0]])
            # fill only the central part 
            x0c = self.pieces[p_name]['x0c']
            x1c = self.pieces[p_name]['x1c']
            y0c = self.pieces[p_name]['y0c']
            y1c = self.pieces[p_name]['y1c']

            try:
                if verbosity > 3:
                    print(f"taking centered_image[{from_idx[1]}:{to_idx[1]}, {from_idx[0]}:{to_idx[0]}] with shape ({self.pieces[p_name]['centered_image'][from_idx[1]:to_idx[1], from_idx[0]:to_idx[0], ::-1].shape})")
                    print(f"into squared_image with shape ({squared_img[:, :, :3].shape})")
                squared_img[:, :, :3] = self.pieces[p_name]['centered_image'][from_idx[1]:to_idx[1], from_idx[0]:to_idx[0], ::-1]
                squared_img[:,:,3] = np.sum(squared_img[:,:,:3], axis=2) > 0
                squared_mask = self.pieces[p_name]['centered_mask'][from_idx[0]:to_idx[0], from_idx[1]:to_idx[1]]
            except:
                row_start2 = (self.sq_size - square_side) // 2
                col_start2 = (self.sq_size - square_side) // 2
                if verbosity > 3:
                    print(f"taking centered_image of size {self.pieces[p_name]['centered_image'].shape}")
                    print(f"into squared_image[{row_start2}:{row_start2+square_side}, {col_start2}:{col_start2 + square_side}] with shape ({square_side}, {square_side}) ")
                squared_img[row_start2:row_start2 + square_side, col_start2:col_start2 + square_side, :3] = self.pieces[p_name]['centered_image']
                squared_img[:,:,3] = np.sum(squared_img[:,:,:3], axis=2) > 0
                squared_mask = squared_img[:,:,3] #self.pieces[p_name]['centered_mask'][from_idx[0]:to_idx[0], from_idx[1]:to_idx[1]]
            
            # we remove the offset in the centered polygon to get it aligned
            xoffset = - (self.img.shape[1]-self.sq_size) / 2 - parameters['monomino_square_size'] / 2  # half of the distance from the square to the shape of the image!
            yoffset = - (self.img.shape[0]-self.sq_size) / 2 - parameters['monomino_square_size'] / 2 
            squared_poly = shapely.affinity.translate(self.pieces[p_name]['centered_polygon'], xoff=xoffset, yoff=yoffset)
            if self.rotation_type > 1:
                if self.rotation_type == 2: # 90 deg rotation
                    degrees = int(np.round(np.random.uniform(0, 3)) * 90)
                elif self.rotation_type == 3: # free deg rotation
                    degrees = random.uniform(-self.rotation_range, self.rotation_range)
                else:
                    print("unknown rotation type!")
                    raise NotImplementedError()          
                self.gt['pieces'][j]['theta'] = degrees

                ################################################
                #   DEBUG VISUALIZATION 1 (continues below)
                ################################################
                # plt.subplot(3, 2, 1); plt.title("Image"); plt.imshow(squared_img);  plt.plot(*squared_poly.boundary.xy)
                # plt.subplot(3, 2, 2); plt.title("Mask"); plt.imshow(squared_mask); plt.plot(*squared_poly.boundary.xy)
                
                ################################################################################################################################################
                #   NOTE: this should not be done like this!
                #   why do we rotate `squared_img[2:, 2:]` ? 
                #       cannot explain, really. There is an issue with the center "value" (we have odd size images guaranteed, so floating value)
                #       which never aligns with any rotation method, and empirically I found out that this gentle nudge (+2) before rotation is needed for the 
                #       correct rotation. It seems simple (just move the "center" + 1!) but after losing a lot of time trying to find an explainable solution, 
                #       I gave up. If you find the solution and can explain, please fix the code and reach out, I will be grateful.
                #       The debug visualization parts are here to help "visualize" the issue if needed.
                #   also, I think now polygons are screwed up (of course, because of this push), and to correct, there should be an offset (dependent on the 
                #   angle). But they are not used, so we probably leave here this bomb ready to explode
                ################################################################################################################################################
                _squared_img_rotated, _squared_mask_rotated, squared_poly_rotated = self.rotate_piece(squared_img[2:, 2:, :], squared_mask[2:, 2:], squared_poly, degrees, method='CV')
                _squared_img = np.zeros_like(squared_img)
                _squared_img[2:, 2:, :] = _squared_img_rotated
                _squared_mask = np.zeros_like(squared_mask)
                _squared_mask[2:, 2:] = _squared_mask_rotated

                ################################################
                #   DEBUG VISUALIZATION 1 (continuing)
                ################################################
                # plt.subplot(3, 2, 3); plt.title("Image"); plt.imshow(_squared_img);  plt.plot(*squared_poly_rotated.boundary.xy)
                # plt.subplot(3, 2, 4); plt.title("Mask"); plt.imshow(_squared_mask); plt.plot(*squared_poly_rotated.boundary.xy)
                # plt.subplot(3, 2, 5); plt.title("Image Overlap"); plt.imshow(squared_img + _squared_img)
                # plt.subplot(3, 2, 6); plt.title("Mask Overlap"); plt.imshow(squared_mask + _squared_mask)
                # plt.show()
                # breakpoint()

                squared_img = _squared_img
                squared_mask = _squared_mask

                ################################################
                # This code rotates the "centered" image as well
                # centered_rotated_img2c, squared_mask2, centered_rotated_poly2 = self.rotate_piece(self.pieces[p_name]['centered_image'][2:, 2:, :], squared_mask, self.pieces[p_name]['centered_polygon'], degrees, center_of_rotation=center_of_rotation, method='CV')
                # centered_rotated_img2 = np.zeros_like(self.pieces[p_name]['centered_image'])
                # centered_rotated_img2[2:, 2:, :] = centered_rotated_img2c

            ################################################
            #   DEBUG VISUALIZATION 2
            ################################################
            # plt.suptitle(f"after rotation of {degrees} degrees")
            # # CENTERED
            # c_img = self.pieces[p_name]['centered_image'].copy()
            # c_img[np.floor(c_img.shape[0]/2 + 0.5).astype(int), :, :] = np.asarray([255, 0, 0])
            # plt.subplot(251); plt.title("centered"); plt.imshow(c_img); plt.scatter(square_side / 2, square_side / 2, marker='x', linewidths=120); plt.plot(*self.pieces[p_name]['centered_polygon'].boundary.xy); plt.scatter(cm_i[0] + shift2center[0], cm_i[1] + shift2center[1], marker='x', c='green', linewidths=60)
            # plt.plot([x0c, x1c], [y0c, y1c]), plt.plot([x1c, x0c], [y0c, y1c])
            # c_img = centered_rotated_img2.copy()
            # c_img[np.floor(c_img.shape[0]/2 + 0.5).astype(int), :, :] = np.asarray([255, 0, 0])
            # plt.subplot(252); plt.title("centered after rotation CV [1 --> 1]"); plt.imshow(c_img); plt.scatter(square_side / 2, square_side / 2, marker='x', linewidths=120); plt.plot(*centered_rotated_poly2.boundary.xy); plt.scatter(cm_i[0] + shift2center[0], cm_i[1] + shift2center[1], marker='x', c='green', linewidths=60)
            # plt.plot([x0c, x1c], [y0c, y1c]), plt.plot([x1c, x0c], [y0c, y1c])
            # c_img = centered_rotated_img3.copy()
            # c_img[np.floor(c_img.shape[0]/2 + 0.5).astype(int), :, :] = np.asarray([255, 0, 0])
            # plt.subplot(253); plt.title("centered after rotation CV [2 --> 2]"); plt.imshow(c_img); plt.scatter(square_side / 2, square_side / 2, marker='x', linewidths=120); plt.plot(*centered_rotated_poly3.boundary.xy); plt.scatter(cm_i[0] + shift2center[0], cm_i[1] + shift2center[1], marker='x', c='green', linewidths=60)
            # plt.plot([x0c, x1c], [y0c, y1c]), plt.plot([x1c, x0c], [y0c, y1c])
            # # SQUARED
            # s_img = squared_img.copy()
            # s_img[np.floor(s_img.shape[0]/2 + 0.5).astype(int), :, :] = np.asarray([255, 0, 0, 1])
            # plt.subplot(256); plt.title("squared before rotation"); plt.imshow(s_img); plt.scatter(self.sq_size / 2, self.sq_size / 2, marker='x', linewidths=120); plt.plot(*squared_poly.boundary.xy)
            # plt.subplot(257); plt.title("squared after rotation ND"); plt.imshow(squared_img2); plt.scatter(self.sq_size / 2, self.sq_size / 2, marker='x', linewidths=120); plt.plot(*squared_poly2.boundary.xy)
            # s_img = squared_img3.copy()
            # s_img[np.floor(s_img.shape[0]/2 + 0.5).astype(int), :, :] = np.asarray([255, 0, 0, 1])
            # plt.subplot(258); plt.title("squared after rotation CV"); plt.imshow(s_img); plt.scatter(self.sq_size / 2, self.sq_size / 2, marker='x', linewidths=120); plt.plot(*squared_poly3.boundary.xy)
            # # OVERLAP 
            # plt.subplot(254); plt.title("overlap centered img centered ND"); plt.imshow(self.pieces[p_name]['centered_image'] + centered_rotated_img2); plt.scatter(square_side / 2, square_side / 2, marker='x', linewidths = 120)
            # # plt.subplot(336); plt.title("overlap mask"); plt.imshow(squared_mask + squared_mask2); plt.scatter(self.sq_size / 2, self.sq_size / 2, marker='x', linewidths = 120)
            # plt.subplot(255); plt.title("overlap centered img centered CV"); plt.imshow(self.pieces[p_name]['centered_image'] + centered_rotated_img3); plt.scatter(self.sq_size / 2, self.sq_size / 2, marker='x', linewidths = 120)
            # plt.subplot(259); plt.title("overlap squared img rotated ND"); plt.imshow(squared_img + squared_img2); plt.scatter(self.sq_size / 2, self.sq_size / 2, marker='x', linewidths = 120)
            # plt.subplot(2,5,10); plt.title("overlap squared img rotated CV"); plt.imshow(squared_img + squared_img3); plt.scatter(self.sq_size / 2, self.sq_size / 2, marker='x', linewidths = 120)
            # plt.show()
            # breakpoint()

            self.pieces[p_name]['squared_image'] = squared_img
            self.pieces[p_name]['squared_mask'] = squared_mask
            self.pieces[p_name]['squared_polygon'] = squared_poly                   # not sure if this is "always" aligned :/
            self.pieces[p_name]['shift2square'] = np.asarray([xoffset, yoffset])    # suspicious (it is only for the polygon?)

        return self.pieces, self.sq_size, self.gt

    def rotate_piece(self, squared_img, squared_mask, squared_poly, degrees, center_of_rotation=None, method='ND'):
        """ Rotate a piece, including the mask and the polygon """
        rot_origin = [squared_img.shape[0] // 2, squared_img.shape[1] // 2]

        if method == 'ND' or method == 'scipy':
            rotated_square_img = ndimage.rotate(squared_img, degrees, reshape=False, mode='constant')
            rotated_square_mask = ndimage.rotate(squared_mask, degrees, reshape=False, mode='constant')
        elif method == 'OPENCV-WARP' or method == 'WARP':
            # region_rot = ndimage.rotate(region_pad, degree, reshape=False, cval=bg_color)
            if center_of_rotation is None:
                center_of_rotation = (np.array(squared_img.shape[:2])) / 2.0  # match scipy's center convention
            rotation_mat = cv2.getRotationMatrix2D(center_of_rotation, degrees, 1)
            rotated_square_img = cv2.warpAffine(squared_img, rotation_mat, (squared_img.shape[1], squared_img.shape[0]),  # keep original size
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rotated_square_mask = cv2.warpAffine(squared_mask, rotation_mat, (squared_img.shape[1], squared_img.shape[0]),  # keep original size
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif method == 'CV' or method == 'OPENCV-ROTATE':
            if degrees == 0 or degrees == 360:
                rotated_square_img, rotated_square_mask = squared_img, squared_mask
            elif degrees == 90:
                rotated_square_img = cv2.rotate(squared_img, cv2.ROTATE_90_COUNTERCLOCKWISE) 
                rotated_square_mask = cv2.rotate(squared_mask, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            elif degrees == 180:
                rotated_square_img = cv2.rotate(squared_img, cv2.ROTATE_180) 
                rotated_square_mask = cv2.rotate(squared_mask, cv2.ROTATE_180)  
            elif degrees == 270:
                rotated_square_img = cv2.rotate(squared_img, cv2.ROTATE_90_CLOCKWISE) 
                rotated_square_mask = cv2.rotate(squared_mask, cv2.ROTATE_90_CLOCKWISE) 
            else:
                print(f"ERROR: this method handles only 90deg rotations! (and degrees={degrees})\nChoose 'WARP' or 'scipy' for other rotations.")

        rotated_square_poly = shapely.affinity.rotate(squared_poly, -degrees, origin=rot_origin) #(np.asarray(rot_origin)-0.5).tolist())
        return rotated_square_img, rotated_square_mask, rotated_square_poly
        # plt.subplot(131); plt.imshow(squared_img); plt.title(f'will be rotated by {degree} degrees!')
        # plt.subplot(132); plt.imshow(rotated_square_img_nd); plt.title('Rotated with SCIPY NDIMAGE')
        # plt.subplot(133); plt.imshow(rotated_square_img); plt.title('Rotated with OPENCV')
        # plt.show()
        # breakpoint()

    def info(self):
        """ Creates a dictionary with the information about the puzzle and returns that """
        info_d = {
            'piece_size': self.sq_size,
            'num_pieces': len(self.pieces),
            'pieces_type': self.pieces_type,
            'pieces_type_description': self.pieces_type_description,
            'rotation_type': self.rotation_type ,
            'rotation_type_description': self.rotation_type_description 
        }
        if self.gt is not None:
            info_d['ground_truth_available'] = True 
            info_d['ground_truth_format'] = 'x, y, theta (degrees)'
        else:
            info_d['ground_truth_available'] = False 
        if 'mask' in self.pieces[list(self.pieces.keys())[0]]:
            info_d['binary_masks_available'] = True
        else:
            info_d['binary_masks_available'] = False
        if 'polygon' in self.pieces[list(self.pieces.keys())[0]]:
            info_d['polygons_available'] = True
        else:
            info_d['polygons_available'] = False

        return info_d   
     
    def save_puzzle(self, iter, bg_color, save_regions=False):

        pdb.set_trace()
        bg_mat = np.full(self.img.shape, bg_color, np.uint8)
        #region_mat_np = np.array(self.region_mat, np.uint32)

        region_rgbs = []
        w_max = 0
        h_max = 0
        groundtruth = []

        puzzle_path = os.path.join(self.puzzle_folder, str(iter))
        os.mkdir(puzzle_path)

        #pdb.set_trace()
        if save_regions:
            #pdb.set_trace()
            cv2.imwrite(os.path.join(puzzle_path, 'regions_uint8.png'), self.region_mat)
            # change to cmap='gray' for grayscale color coding
            plt.imsave(os.path.join(puzzle_path, 'regions_col_coded.jpg'), self.region_mat, cmap='jet')

        # Compute maximum boundary
        for i in range(self.region_cnt):

            region_map = self.region_mat == i
            region_map3 = np.repeat(region_map, 3).reshape(self.img.shape)
            rgb = np.where(region_map3, self.img, bg_mat)

            coords = np.argwhere(region_map)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1

            region_rgb = rgb[y0:y1, x0:x1]
            region_rgbs.append(region_rgb)
            groundtruth.append({
                'id': i,
                'dx': int(x0),
                'dy': int(y0)
            })

            w_max = max(w_max, x1 - x0)
            h_max = max(h_max, y1 - y0)
        r = int(math.sqrt(w_max ** 2 + h_max ** 2) + 5)

        groundtruth_path = os.path.join(puzzle_path, 'groundtruth.txt')
        outfile = open(groundtruth_path, 'w')
        
        # pdb.set_trace()
        # Compute groundtruth
        # Save groundtruth in txt
        # pdb.set_trace()
        for i in range(self.region_cnt):

            pad_top = (r - region_rgbs[i].shape[0]) // 2
            pad_left = (r - region_rgbs[i].shape[1]) // 2
            pad_bottom = r - region_rgbs[i].shape[0] - pad_top
            pad_right = r - region_rgbs[i].shape[1] - pad_left

            region_pad = cv2.copyMakeBorder(region_rgbs[i],
                pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=bg_color)

            ############
            # ROTATIONS
            if self.rotation_type == 1: # no rotation
                region_rot = region_pad
            else:
                if self.rotation_type == 2: # 90 deg rotation
                    degree = np.round(np.random.uniform(0, 3)).astype(np.uint8) * 90
                elif self.rotation_type == 3: # free deg rotation
                    degree = random.uniform(-self.rot_range, self.rot_range)
                # region_rot = ndimage.rotate(region_pad, degree, reshape=False, cval=bg_color)
                rotation_mat = cv2.getRotationMatrix2D((region_pad.shape[1]/2, region_pad.shape[0]/2), degree, 1)
                region_rot = cv2.warpAffine(region_pad, rotation_mat, (region_pad.shape[1], region_pad.shape[0]),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
            if self.alpha_channel:
                rgba = cv2.cvtColor(region_rot, cv2.COLOR_RGB2RGBA)
                rgba[:, :, 3] = 255*(1 - (region_rot[:,:] == bg_color)[:,:,0])
                region_rot = rgba

            cv2.imwrite(os.path.join(puzzle_path, 'piece-%d.png' % i), region_rot)

            groundtruth[i]['dx'] -= pad_left
            groundtruth[i]['dy'] -= pad_top
            groundtruth[i]['dx_region_to_img'] = (bg_mat.shape[1] - r) // 2
            groundtruth[i]['dy_region_to_img'] = (bg_mat.shape[0] - r) // 2
            groundtruth[i]['dx_full'] = groundtruth[i]['dx'] - groundtruth[i]['dx_region_to_img']
            groundtruth[i]['dy_full'] = groundtruth[i]['dy'] - groundtruth[i]['dy_region_to_img']
            groundtruth[i]['rotation'] = degree / 180 * math.pi
            groundtruth[i]['rotation_deg'] = degree

            outfile.write('%d %d %.3f\n' % (groundtruth[i]['dx'], groundtruth[i]['dy'], groundtruth[i]['rotation']))
            # rgb = np.ma.masked_equal(self.region_mat == i, self.img)
            # cv2.imshow('region_rgb', region_rgbs[i])
            # cv2.imshow('region_pad', region_pad)
            # cv2.imshow('region_rot', region_rot)
            # cv2.waitKey()
            # print(rgb)
            # break

        outfile.close()

        #pdb.set_trace()
        # general information about the generation process
        # this will be the starting point to solve the puzzle
        general_info = {
            'name': self.name,
            'orig_img_w': bg_mat.shape[1],
            'orig_img_h': bg_mat.shape[0],
            'region_side': r,
            'ref_fragment': groundtruth[0],
            'regions': self.region_cnt,
            'alpha_channel': self.alpha_channel,
            'rot_range': self.rot_range,
            'small_region_area_ratio': self.small_region_area_ratio,
            'num_of_missing_fragments': int(self.num_of_missing_fragments),
            'missing_indices': [int(ind) for ind in self.missing_indices]
        }

        gt = {
            'info': general_info,
            'fragments': groundtruth
        }
        # Save groundtruth in json
        groundtruth_path = os.path.join(puzzle_path, 'groundtruth.json')
        outfile = open(groundtruth_path, 'w')
        json.dump(groundtruth, outfile, indent=3)
        outfile.close()

        #for gk in general_info.keys(): print(gk, type(general_info[gk]))
        groundtruth_path = os.path.join(puzzle_path, 'groundtruth_extended.json')
        outfile = open(groundtruth_path, 'w')
        json.dump(gt, outfile, indent=3)
        outfile.close()

        # Save config file
        config_path = os.path.join(puzzle_path, 'config.txt')
        outfile = open(config_path, 'w')

        outfile.write('piece-\n') # Prefix
        outfile.write('%d\n' % self.region_cnt) # Piece number
        outfile.write('%d %d %d\n' % (bg_color[0], bg_color[1], bg_color[2])) # bg color in BGR

        outfile.close()

        # save a file for the challenge
        challenge_path = os.path.join(puzzle_path, 'challenge.json')
        outfile = open(challenge_path, 'w')
        json.dump(general_info, outfile, indent=3)
        outfile.close()

    def save_zip(self, iter):

        puzzle_path = os.path.join(self.puzzle_folder, str(iter))
        zip_path = os.path.join(puzzle_path, 'puzzle-%d.zip' % iter)

        zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
        for i in range(self.region_cnt):
            piece_name = 'piece-%d.png' % i
            zipf.write(os.path.join(puzzle_path, piece_name), piece_name)
        zipf.write(os.path.join(puzzle_path, 'config.txt'), 'config.txt')
        zipf.close()

    def save_challenge_zip(self, iter):

        puzzle_path = os.path.join(self.puzzle_folder, str(iter))
        zip_path = os.path.join(puzzle_path, f'challenge-{self.name}-%d.zip' % iter)

        zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
        for i in range(self.region_cnt):
            if i not in self.missing_indices:
                piece_name = 'piece-%d.png' % i
                zipf.write(os.path.join(puzzle_path, piece_name), piece_name)
        zipf.write(os.path.join(puzzle_path, 'challenge.json'), 'challenge.json')
        zipf.close()


    def run(self, piece_n, offset_rate_h=0.2, offset_rate_w=0.2, small_region_area_ratio=0.25, rot_range=180,
            smooth_flag=False, alpha_channel=True, perc_missing_fragments=0, erosion=0, borders=False):
        """
        Generate irregular puzzle pieces by creating random cutting masks.
        
        This is the main method for irregular piece generation. It creates a grid-based
        cutting pattern with smooth or segmented curves, then extracts the resulting regions.
        
        Parameters
        ----------
        piece_n : int
            Approximate number of pieces to create (actual count may vary)
        offset_rate_h : float, optional
            Vertical offset randomness as fraction of piece height (default: 0.2)
        offset_rate_w : float, optional
            Horizontal offset randomness as fraction of piece width (default: 0.2)
        small_region_area_ratio : float, optional
            Threshold for removing small pieces as fraction of average piece area (default: 0.25)
        rot_range : float, optional
            Maximum rotation angle in degrees for type 3 puzzles (default: 180)
        smooth_flag : bool, optional
            If True, use smooth curves; if False, use linear segments (default: False)
        alpha_channel : bool, optional
            If True, pieces include alpha channel for transparency (default: True)
        perc_missing_fragments : float, optional
            Percentage of pieces to mark as missing (0-100) (default: 0)
        erosion : int, optional
            Erosion level to apply to pieces (default: 0)
        borders : bool, optional
            If True, save extrapolated piece borders (default: False)
        
        Notes
        -----
        The algorithm works by:
        1. Creating a grid of approximate dimensions sqrt(piece_n) x sqrt(piece_n)
        2. Drawing random curves (vertical and horizontal) to cut the image
        3. Finding connected regions in the resulting mask
        4. Filtering out regions that are too small
        5. Optionally applying erosion and border effects
        
        The actual number of pieces may differ from piece_n due to curve randomness
        and small region filtering.
        """

        self.rot_range = rot_range
        self.piece_n = piece_n
        # Calculate grid dimensions (approximately square)
        self.w_n = math.floor(math.sqrt(piece_n))
        self.h_n = self.w_n
        self.smooth_flag = smooth_flag
        self.alpha_channel = alpha_channel
        self.small_region_area_ratio = small_region_area_ratio
        self.missing_indices = []
        self.erosion = erosion
        self.borders = borders

        # Generate the cutting mask with random curves
        self.get_mask(offset_rate_h, offset_rate_w)
        
        # Extract regions from the mask
        self.get_regions()

        # Calculate how many pieces to mark as missing
        self.num_of_missing_fragments = np.floor(self.region_cnt * perc_missing_fragments / 100).astype(int)
        if self.num_of_missing_fragments > 0:
            # Randomly select pieces to mark as missing (skip index 0)
            self.missing_indices = random.sample(set(np.arange(1, self.region_cnt)), self.num_of_missing_fragments)
            self.missing_indices = np.sort([int(ind) for ind in self.missing_indices])

    def save(self, bg_color=(0,0,0), save_regions=False):

        exist_data_len = len(glob(os.path.join(self.raw_regions, '*.npy')))
        self.save_raw_regions(exist_data_len)
        self.save_puzzle(exist_data_len, bg_color, save_regions)
        if self.borders:
            self.save_extrapolated_regions(exist_data_len)
        self.save_zip(exist_data_len)
        self.save_challenge_zip(exist_data_len)

def crop_extrapolated(image, padding=1, return_vals=False):
    """
    Crop an extrapolated piece image to its bounding box.
    
    Finds the tight bounding box around non-transparent pixels and crops
    the image, with optional padding.
    
    Parameters
    ----------
    image : np.ndarray
        RGBA image with shape (H, W, 4)
    padding : int, optional
        Number of pixels to include as padding around the piece (default: 1)
    return_vals : bool, optional
        If True, also return bounding box coordinates (default: False)
    
    Returns
    -------
    cropped_image : np.ndarray
        Cropped RGBA image
    x0, x1, y0, y1 : int, optional
        Bounding box coordinates (only if return_vals=True)
    
    Notes
    -----
    Uses the alpha channel (channel 3) to determine which pixels are part
    of the piece. Assumes background has alpha=0.
    """
    # Find bounding box of non-transparent pixels
    x0 = np.min(np.where(image[:,:,3] > 0)[1]) - padding
    x1 = np.max(np.where(image[:,:,3] > 0)[1]) + padding
    y0 = np.min(np.where(image[:,:,3] > 0)[0]) - padding
    y1 = np.max(np.where(image[:,:,3] > 0)[0]) + padding

    if return_vals == True:
        return image[y0:y1, x0:x1, :], x0, x1, y0, y1
    return image[y0:y1, x0:x1, :]