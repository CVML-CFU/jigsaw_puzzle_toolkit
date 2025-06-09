from enum import Enum
import cv2
from shapely import Polygon, transform
import numpy as np 
import os 
import matplotlib.pyplot as plt

class PuzzleType(Enum):
    type0 = 'type0'
    type0r = 'type0R'
    type1 = 'type1'
    type1r = 'type1R'
    type2 = 'type2'
    type2r = 'type2R'

    def __str__(self):
        return self.value

# @staticmethod
def extract_binary_mask(irregular_image: np.ndarray, background: int = 0):
    if irregular_image.shape[2] == 4:
        binary_mask = 1 - (irregular_image[:,:,3] == background).astype(np.uint8)
    else:
        binary_mask = 1 - (irregular_image[:,:,0] == background).astype(np.uint8)
    return binary_mask

# @staticmethod
def calculate_center_of_mass(binary_mask: np.ndarray):
    mass_y, mass_x = np.where(binary_mask >= 0.5)
    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)
    # center = [ np.average(indices) for indices in np.where(th1 >= 255) ]
    return [cent_x, cent_y]

# @staticmethod
def extract_polygon(binary_mask: np.ndarray):
    bin_img = binary_mask.copy()
    bin_img = cv2.dilate(bin_img.astype(np.uint8), np.ones((2,2)), iterations=1)
    contours, _ = cv2.findContours(bin_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = contours[0]
    # should we remove 0.5 or it's just visualization?
    #shapely_points = [(point[0][0]-0.5, point[0][1]-0.5) for point in contour_points]  # Shapely expects points in the format (x, y)
    shapely_points = [(point[0][0]-0.5, point[0][1]-0.5) for point in contour_points]  # Shapely expects points in the format (x, y)
    if len(shapely_points) < 4:
        print('we have a problem, too few points', shapely_points)
        raise ValueError('\nWe have fewer than 4 points on the polygon, so we cannot create a Shapely polygon out of this points! Maybe something went wrong with the mask?')
    polygon = Polygon(shapely_points)
    return polygon

class Puzzle:

    def __init__(self, input_path: str, puzzle_type: PuzzleType, output_path: str):    

        self.input_path = input_path
        self.puzzle_type = puzzle_type
        self.output_path = output_path
        self.names = []
        self.images = []
        self.masks = []
        self.polygons = []
        self.input_data = {}

    def load_input_data(self):
        """ Loads the data and handles the different use-cases. """
        if self.input_path.endswith('json'):
            print("Got the json dictionary with the data, loading from there..")
            import json 
            with open (self.input_path, 'r') as jf:
                input_dict = json.load(jf)
            
            fragments = input_dict['fragments']
            print(f"found {len(fragments)} fragments")
            for fragment in fragments:
                fragment_path = os.path.join(os.path.dirname(os.path.join(self.input_path)), fragment['filename'].replace('obj', 'png'))
                raw_image = plt.imread(fragment_path)
                frag_name = fragment['filename'].split('.')[0]
                image, mask, polygon = self.preprocess_irregular_piece(raw_image)
                self.names.append(frag_name)
                self.images.append(image)
                self.masks.append(mask)
                self.polygons.append(polygon)
                self.input_data[frag_name] = {
                    'image': image,
                    'mask': mask,
                    'polygon': polygon
                }
            self.show_piece(3)
            breakpoint()

    def preprocess_irregular_piece(self, raw_image):
        bmask = extract_binary_mask(raw_image)
        polygon = extract_polygon(bmask)
        cm = calculate_center_of_mass(bmask)
        cen_image, cen_bmask, cen_polygon = Puzzle.center_piece(raw_image, bmask, polygon, cm)    
        return cen_image, cen_bmask, cen_polygon 

    @staticmethod
    def center_piece(raw_image, bmask, polygon, cm):
        """ Center the piece image, mask and polygon """
        # breakpoint()
        cen_image = np.zeros_like(raw_image)
        cen_bmask = np.zeros_like(bmask)
        half_image_side = np.round(raw_image.shape[0]/2).astype(int)
        center_pos = [half_image_side, half_image_side]
        shift_x, shift_y = -np.round(np.array(cm) - center_pos).astype(int)
        if shift_x == 0 and shift_y == 0:
            cen_image = raw_image
            cen_bmask = bmask
        if shift_x >= 0 and shift_y >= 0:
            cen_image[shift_y:, shift_x:] = raw_image[:raw_image.shape[0]-shift_y, :raw_image.shape[1]-shift_x]
            cen_bmask[shift_y:, shift_x:] = bmask[:bmask.shape[0]-shift_y, :bmask.shape[1]-shift_x]
        elif shift_x >= 0 and shift_y < 0:
            cen_image[:shift_y, shift_x:] = raw_image[-shift_y:, :raw_image.shape[1]-shift_x]
            cen_bmask[:shift_y, shift_x:] = bmask[-shift_y:, :bmask.shape[1]-shift_x]
        elif shift_x < 0 and shift_y >= 0:
            cen_image[shift_y:, :shift_x] = raw_image[:raw_image.shape[0]-shift_y, -shift_x:]
            cen_bmask[shift_y:, :shift_x] = bmask[:bmask.shape[0]-shift_y, -shift_x:]
        elif shift_x < 0 and shift_y < 0:
            cen_image[:shift_y, :shift_x] = raw_image[-shift_y:, -shift_x:]
            cen_bmask[:shift_y, :shift_x] = bmask[-shift_y:, -shift_x:]

        # polygon
        cen_polygon = transform(polygon, lambda f: f + [+shift_x,+shift_y])

        return cen_image, cen_bmask, cen_polygon 

    def save_pieces(self):
        output_dir = os.path.join(self.output_path, os.path.basename(os.path.dirname(os.path.join(self.input_path))))
        os.makedirs(output_dir, exist_ok=True)
        images_out_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_out_dir, exist_ok=True)
        bmasks_out_dir = os.path.join(output_dir, 'binary_masks')
        os.makedirs(bmasks_out_dir, exist_ok=True)        
        polygons_out_dir = os.path.join(output_dir, 'polygons')
        os.makedirs(polygons_out_dir, exist_ok=True)
        for name, image, bmask, polygon in zip(self.names, self.images, self.masks, self.polygons):
            plt.imsave(os.path.join(images_out_dir, f"{name}.png"), image)
            plt.imsave(os.path.join(bmasks_out_dir, f"{name}.png"), bmask)
            np.save(os.path.join(polygons_out_dir, f"{name}.png"), polygon)

        print("Done!")

    def show_piece(self, index: int = 0):
        plt.suptitle(f"Piece {index}: {self.names[index]}", fontsize=32)
        plt.subplot(121)
        plt.title("Image")
        plt.imshow(self.images[index])
        plt.plot(*(self.polygons[index].boundary.xy), linewidth=3)
        
        plt.subplot(122)
        plt.title("Binary Mask")
        plt.imshow(self.masks[index])
        plt.plot(*(self.polygons[index].boundary.xy), linewidth=3)
        plt.show()

    def create_pieces(self):
        # puzzle_data, pieces, solution = 
        return 1,1,1