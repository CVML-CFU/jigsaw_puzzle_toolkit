from enum import Enum
import cv2
from shapely import Polygon, transform
import numpy as np 
import os, json
import matplotlib.pyplot as plt

class PuzzleType(Enum):
    """
    Type 0:
    - given an image, we create squared pieces 
    Type 0R:
    - same as Type 0 but with rotation
    Type 1:
    - given an image, we create polyomino pieces
    Type 1R:
    - same as Type 1 but with rotation
    Type 2: 
    - given an image or a set of fragments, we create irregular fragments
    Type 2R:
    - same as Type 2 but with rotation
    """
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
def extract_polygon(binary_mask: np.ndarray, return_max_dist_from_center: bool = False):
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
    x,y,w,h = cv2.boundingRect(contour_points)
    img_center = np.asarray(bin_img.shape[:2]) / 2
    dist_from_center = np.sqrt(np.square(w/2) + np.square(h/2))
    # # breakpoint()
    # dists_from_center = [np.abs(x - img_center[0]), np.abs(x+w - img_center[0]), 
    #                      np.abs(y - img_center[1]), np.abs(y+h - img_center[1])] 

    # plt.suptitle(f"Max: {np.max(dists_from_center)}")
    # plt.subplot(121)
    # plt.imshow(binary_mask)
    # plt.plot(*(polygon.boundary.xy))
    # plt.subplot(122)
    # img2 = np.zeros_like(binary_mask)
    # img2 = cv2.rectangle(img2, (x,y), (w,h), (0,255,0), 15)    
    # plt.imshow(img2)
    # plt.show()
    # breakpoint()
    if return_max_dist_from_center == True:
        return polygon, dist_from_center
    else:
        return polygon
class Puzzle:

    def __init__(self, input_path: str, puzzle_type: PuzzleType, output_path: str, padding: int = 0):    

        self.input_path = input_path
        self.puzzle_type = puzzle_type
        self.output_path = output_path
        self.names = []
        self.images = []
        self.masks = []
        self.polygons = []
        self.input_data = {}
        self.gt = {}
        self.xs = []
        self.ys = []
        self.thetas = []
        self.positions = []
        self.puzzle_info = {}
        self.sizes = []
        self.max_dist_from_center = 0
        self.padding = padding

    def load_input_data(self, crop_pieces: bool = False):
        """ Loads the data and handles the different use-cases. """


        if self.input_path.endswith('json'):
            print("Got the json dictionary with the data, loading from there..")
            import json 
            with open (self.input_path, 'r') as jf:
                input_dict = json.load(jf)
            
            self.gt['adjacency'] = input_dict['adjacency']
            
            fragments = input_dict['fragments']
            print(f"found {len(fragments)} fragments")
            self.puzzle_info['num_pieces'] = len(fragments)
            print("loading..")
            for fragment in fragments:
                fragment_path = os.path.join(os.path.dirname(os.path.join(self.input_path)), fragment['filename'].replace('obj', 'png'))
                raw_image = plt.imread(fragment_path)
                frag_name = fragment['filename'].split('.')[0]
                image, mask, polygon, dist_from_center = self.preprocess_irregular_piece(raw_image)
                self.names.append(frag_name)
                self.images.append(image)
                self.masks.append(mask)
                self.polygons.append(polygon)
                self.sizes.append(dist_from_center)
                if dist_from_center > self.max_dist_from_center:
                    self.max_dist_from_center = dist_from_center
                self.input_data[frag_name] = {
                    'image': image,
                    'mask': mask,
                    'polygon': polygon
                }
                self.gt[frag_name] = {
                    'idx': fragment['idx'],
                    'x': fragment['position'][0],
                    'y': fragment['position'][1],
                    'theta': fragment['position'][2]
                }
                self.xs.append(fragment['position'][0])
                self.ys.append(fragment['position'][1])
                self.thetas.append(fragment['position'][2])
                self.positions.append(fragment['position'])

            self.puzzle_info['pieces_image_size'] = self.images[0].shape
            self.puzzle_info['binary_masks_available'] = True
            self.puzzle_info['polygons_available'] = True
            self.puzzle_info['ground_truth_available'] = True
            self.puzzle_info['max_dist_from_center'] = int(self.max_dist_from_center)
            self.puzzle_info['padding'] = self.padding
            # self.show_piece(3)
            # breakpoint()
            # print(self.sizes)

            if crop_pieces == True:
                center_pix = np.round(self.puzzle_info['pieces_image_size'][0] / 2).astype(int)
                range_crop = self.padding + np.round(self.max_dist_from_center).astype(int)
                polygon_translation = center_pix - range_crop
                for frag_key in self.input_data.keys():
                    orig_image_shape = self.input_data[frag_key]['image'].shape
                    self.input_data[frag_key]['image'] = self.input_data[frag_key]['image'][center_pix-range_crop:center_pix+range_crop, center_pix-range_crop:center_pix+range_crop]
                    self.input_data[frag_key]['mask'] = self.input_data[frag_key]['mask'][center_pix-range_crop:center_pix+range_crop, center_pix-range_crop:center_pix+range_crop]
                    self.input_data[frag_key]['polygon'] = transform(self.input_data[frag_key]['polygon'], lambda f: f - [+polygon_translation,+polygon_translation])
                    
                    plt.suptitle(f"Piece: {frag_key}\nPieces image size: {orig_image_shape} --> {self.puzzle_info['pieces_image_size']}\nFrag max size: {self.max_dist_from_center*2}", fontsize=24)
                    plt.subplot(121)
                    plt.title("Image")
                    plt.imshow(self.input_data[frag_key]['image'])
                    plt.scatter(self.input_data[frag_key]['image'].shape[0] / 2, self.input_data[frag_key]['image'].shape[1] / 2, s=15)
                    plt.plot(*(self.input_data[frag_key]['polygon'].boundary.xy), linewidth=3)
                    
                    plt.subplot(122)
                    plt.title("Binary Mask")
                    plt.imshow(self.input_data[frag_key]['mask'])
                    plt.plot(*(self.input_data[frag_key]['polygon'].boundary.xy), linewidth=3)
                    plt.show()
                    breakpoint()

    def preprocess_irregular_piece(self, raw_image):
        bmask = extract_binary_mask(raw_image)
        polygon, dist_from_center = extract_polygon(bmask, return_max_dist_from_center=True)
        cm = calculate_center_of_mass(bmask)
        cen_image, cen_bmask, cen_polygon = Puzzle.center_piece(raw_image, bmask, polygon, cm)    
        return cen_image, cen_bmask, cen_polygon, dist_from_center

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

    def save(self):
        print("saving..")
        self.output_dir = os.path.join(self.output_path, os.path.basename(os.path.dirname(os.path.join(self.input_path))))
        os.makedirs(self.output_dir, exist_ok=True)
        images_out_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(images_out_dir, exist_ok=True)
        bmasks_out_dir = os.path.join(self.output_dir, 'binary_masks')
        os.makedirs(bmasks_out_dir, exist_ok=True)        
        polygons_out_dir = os.path.join(self.output_dir, 'polygons')
        os.makedirs(polygons_out_dir, exist_ok=True)
        for frag_key in self.input_data.keys():
            frag_gt = self.gt[frag_key]
            frag_data = self.input_data[frag_key]

            # for name, image, bmask, polygon in zip(self.names, self.images, self.masks, self.polygons):
            plt.imsave(os.path.join(images_out_dir, f"{frag_gt['idx']}_{frag_key}.png"), frag_data['image'])
            plt.imsave(os.path.join(bmasks_out_dir, f"{frag_gt['idx']}_{frag_key}.png"), frag_data['mask'])
            np.save(os.path.join(polygons_out_dir, f"{frag_gt['idx']}_{frag_key}.png"), frag_data['polygon'])
        
        np.savetxt(os.path.join(self.output_dir, "ground_truth.txt"), np.asarray(self.positions))
        with open(os.path.join(self.output_dir, "ground_truth.json"), 'w') as jf:
            json.dump(self.gt, jf, indent=2)
        with open(os.path.join(self.output_dir, "puzzle_info.json"), 'w') as jf:
            json.dump(self.puzzle_info, jf, indent=2)

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