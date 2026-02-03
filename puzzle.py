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


class PuzzleType(Enum):
    """
    Type S1: SQUARED PIECES, NO ROTATION
    Type S2: SQUARED PIECES, 90 Degrees ROTATIONS

    Type P1: POLYOMINO PIECES, NO ROTATION
    Type P2: POLYOMINO PIECES, 90 Degrees ROTATIONS
    Type P3: POLYOMINO PIECES, free ROTATIONS

    Type M1: PIECES from PATTERN MAP, NO ROTATION
    Type M2: PIECES from PATTERN MAP, 90 Degrees ROTATIONS
    Type M3: PIECES from PATTERN MAP, free ROTATIONS

    Type I1: IRREGULAR PIECES, NO ROTATION
    Type I2: IRREGULAR PIECES, 90 Degrees ROTATIONS
    Type I3: IRREGULAR PIECES, free ROTATIONS
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
    
    def _rot(self): # the rotations 1 --> no rotation, 2 --> 90 deg rotations, 3 --> free rotations
        return int(self.value[-1])
    
    def _rot_str(self):
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

    def _type(self): # the letter corresponding to the type
        return self.value[0]

    def _type_str(self):
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
    

# @staticmethod
def extract_binary_mask(irregular_image: np.ndarray, background: int = 0, close: bool = True):
    if irregular_image.shape[2] == 4:
        binary_mask = 1 - (irregular_image[:,:,3] == background).astype(np.uint8)
    else:
        binary_mask = 1 - (irregular_image[:,:,0] == background).astype(np.uint8)
    if close == True:
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones_like((5,5)))
    return binary_mask

# @staticmethod
def calculate_center_of_mass(binary_mask: np.ndarray, method: str = 'np'):

    if method == 'scipy':
        import scipy
        cent_y, cent_x = scipy.ndimage.center_of_mass(bmask)
    else: #if method == 'np':
        mass_y, mass_x = np.where(binary_mask >= 0.5)
        cent_x = np.average(mass_x)
        cent_y = np.average(mass_y)
        # center = [ np.average(indices) for indices in np.where(th1 >= 255) ]
    return [cent_x, cent_y]

# @staticmethod
def extract_polygon(binary_mask: np.ndarray, return_vals: bool = False):
    bin_img = binary_mask.copy()
    bin_img = cv2.dilate(bin_img.astype(np.uint8), np.ones((2,2)), iterations=1)
    contours, _ = cv2.findContours(bin_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = contours[0]
    if len(contours) > 0:
        for cnt in contours:
            if len(cnt) > len(contour_points):
                contour_points = cnt    

    # breakpoint()
    # should we remove 0.5 or it's just visualization?
    #shapely_points = [(point[0][0]-0.5, point[0][1]-0.5) for point in contour_points]  # Shapely expects points in the format (x, y)
    shapely_points = [(point[0][0]-0.5, point[0][1]-0.5) for point in contour_points]  # Shapely expects points in the format (x, y)
    if len(shapely_points) < 4:
        print('we have a problem, too few points', shapely_points)
        raise ValueError('\nWe have fewer than 4 points on the polygon, so we cannot create a Shapely polygon out of this points! Maybe something went wrong with the mask?')
    polygon = Polygon(shapely_points)
    x,y,w,h = cv2.boundingRect(contour_points)
    img_center = np.asarray(bin_img.shape[:2]) / 2

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
    # breakpoint()

    if return_vals == True:
        return polygon, bounding_box_half_diagonal, [h,w]
    else:
        return polygon

class Puzzle:

    def __init__(self, input_path: str, puzzle_type: PuzzleType, output_path: str, input_type: str, padding: int = 1000, \
        target_size: tuple[int, int] = (0, 0), puzzle_name: str = None, output_folder_name:str = None):    

        self.input_type = input_type
        if self.input_type == 'repair' or self.input_type == 'json':
            input_path = os.path.join(root_path, 'data.json')
        self.input_path = input_path
        if not puzzle_name:
            puzzle_name = os.path.basename(input_path)
        self.puzzle_name = puzzle_name
        self.puzzle_type = puzzle_type
        self.rotation_type = self.puzzle_type._rot()
        self.pieces_type = self.puzzle_type._type()
        #self.output_path = output_path
        if not output_folder_name:
            output_folder_name = os.path.basename(self.input_path)
        if self.input_type == 'image':
            output_folder_name = output_folder_name.split(".")[0] # remove .jpg or .png
        self.output_dir = os.path.join(output_path, output_folder_name)
        os.makedirs(self.output_dir, exist_ok=True)
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
        self.radii = []
        self.max_enclosing_radius = 0
        self.padding = padding
        self.target_size = target_size
        self.random_rotations = []

    def prepare_puzzle(self, num_pieces: int = 9, crop_pieces: bool = True, pattern_map_path: str = None):
        if self.input_type == 'repair' or self.input_type == 'json':
            self.prepare_puzzle_from_json(crop_pieces = crop_pieces, new_size = self.target_size[0])
        if self.input_type == 'image':
            self.prepare_puzzle_from_image(num_pieces = num_pieces, pattern_map_path = pattern_map_path)

    def process_region_map(self, region_map, perc_min=0.01):
        """
        It eliminates small regions and keep only the one who are "big" enough 
        """
        uvals = np.unique(region_map)
        rmap = np.zeros_like(region_map)
        rc = 1
        min_pixels = region_map.shape[0] * region_map.shape[1] * perc_min
        for uval in np.unique(region_map): 
            # print(f"region with value:{uval} has {np.sum(region_map==uval)} pixels")
            # plt.imshow(region_map==uval)
            # plt.show()
            if np.sum(region_map==uval) > min_pixels and uval > 0:
                rmap += (region_map==uval).astype(np.uint8) * rc
                rc += 1
            elif uval > 0:
                print("region too small! check threshold")

        # plt.subplot(121); plt.imshow(region_map, vmin=0, vmax=255)
        # plt.subplot(122); plt.imshow(rmap, vmin=0, vmax=31)
        # plt.show()
        # pdb.set_trace()
        return rmap, rc-1

    def adapt_to_pattern_size(self, image, pattern_map, crop=True):
        
        if crop == True:
            target_size = pattern_map.shape[0]
            # Get current dimensions
            h, w = image.shape[:2]
            # Calculate scale factor to match the smaller dimension
            scale = target_size / min(h, w)
            # Resize so smaller side becomes target_size
            resized = resize(image, (int(h * scale), int(w * scale)), anti_aliasing=True)
        
            # Crop center to target_size x target_size
            h_new, w_new = resized.shape[:2]
            start_h = (h_new - target_size) // 2
            start_w = (w_new - target_size) // 2
            adapted = resized[start_h:start_h + target_size, start_w:start_w + target_size]
            # Optional: Convert to uint8 if needed (skimage.resize returns float64 in [0,1])
            # cropped = (cropped * 255).astype(np.uint8)
        else:
            adapted = resize(image, (pattern_map.shape[0], pattern_map.shape[1]), anti_aliasing=True) 
        # image = np.round(image * 255).astype(np.uint8)
        return adapted

    def prepare_puzzle_from_image(self, num_pieces: int = 9, pattern_map_path: str = None):
        """ 
        Starting from an image, creates the pieces by cutting it 
        it uses the parameters given in the creation of the Puzzle object
        """
        image = cv2.imread(self.input_path)
        parameters = {
            'name': self.puzzle_name,
            'num_pieces': num_pieces,
            'pattern_map_path': pattern_map_path,
            'padding': 9,
            'rotation_type': self.puzzle_type._rot(),
            'rotation_type_s': self.puzzle_type._rot_str(),
            'pieces_type': self.puzzle_type._type(),
            'pieces_type_s': self.puzzle_type._type_str(),
            'rotation_range': 180
        }
        # np.savetxt(os.path.join(self.output_dir, "random_rot.txt"), np.asarray(self.random_rotations))
        # np.savetxt(os.path.join(self.output_dir, "ground_truth.txt"), np.asarray(self.positions))
        # with open(os.path.join(self.output_dir, "ground_truth.json"), 'w') as jf:
        #     json.dump(self.gt, jf, indent=2)
        # with open(os.path.join(self.output_dir, "puzzle_info.json"), 'w') as jf:
        #     json.dump(self.puzzle_info, jf, indent=2)

        if self.pieces_type == 'I': # == 'irregular':
            generator = PuzzleGenerator(image, parameters=parameters)
            # generator.generate_regions(puzzle_parameters, save_image=True)
            generated_puzzle = generator.run(num_pieces, offset_rate_h=0.2, offset_rate_w=0.2, small_region_area_ratio=0.25, rot_range=0,
                smooth_flag=True, alpha_channel=True, perc_missing_fragments=0, erosion=0, borders=False)
            generator.save_jpg_regions(self.output_dir, skip_bg=False)
            parameters['start_from'] = 0
            self.pieces, self.patch_size = generator.extract_pieces()
            
        if self.pieces_type == 'M' and pattern_map_path is not None: # if shape == 'pattern'
            patterns_map = cv2.imread(pattern_map_path)
            pattern_map, num_pieces = self.process_region_map(patterns_map)
            generator = PuzzleGenerator(image, parameters=parameters)
            generator.region_cnt = num_pieces + 1
            generator.region_mat = pattern_map # processed version 
            generator.save_jpg_regions(self.output_dir, skip_bg=True)
            parameters['start_from'] = 1
            self.pieces, self.patch_size = generator.extract_pieces()

        if self.pieces_type == 'P' and pattern_map_path is not None: # if shape == 'polyominos'
            region_map = cv2.imread(f"{pattern_map_path}.png", cv2.IMREAD_GRAYSCALE)
            print("pattern map: ", pattern_map_path)
            with open(f"{pattern_map_path}.json", 'r') as jc:
                pieces_centers = json.load(jc)
            
            image = self.adapt_to_pattern_size(image, region_map)
            pattern_map, num_pieces = self.process_region_map(region_map)
            generator = PuzzleGenerator(image, parameters=parameters, pieces_centers=pieces_centers)
            generator.region_cnt = num_pieces + 1
            generator.region_mat = pattern_map # processed version 
            # breakpoint()
            parameters['start_from'] = 1
            generator.save_jpg_regions(self.output_dir, skip_bg=True)
            self.pieces, self.patch_size, self.gt = generator.get_polyomino_pieces_from_puzzle(parameters=parameters)
            self.puzzle_info = generator.info()
            self.puzzle_info['puzzle_image_size'] = image.shape 

    def prepare_puzzle_from_json(self, crop_pieces: bool = False, add_random_rotations: bool = False, theta_step: int = 45):
        """ Loads the data and handles the different use-cases. """

        if not self.input_path.endswith('json'):
            print("In this case, we need the json file!")
        else:
            print("Got the json dictionary with the data, loading from there..")
            import json 
            with open (self.input_path, 'r') as jf:
                input_dict = json.load(jf)
            
            self.gt['adjacency'] = input_dict['adjacency']
            self.gt['transform'] = input_dict['transform']
            self.gt['pieces'] = {}
            
            fragments = input_dict['fragments']
            print(f"found {len(fragments)} fragments")
            self.puzzle_info['num_pieces'] = len(fragments)
            print("loading the pieces..")
            self.raw_images = []
            self.extended_raw_images = []
            self.extended_raw_masks = []
            self.extended_raw_polygons = []
            self.raw_masks = []
            self.raw_polygons = []
            if add_random_rotations == True:
                self.random_rotations = []
            for j, fragment in enumerate(fragments):
                fragment_path = os.path.join(os.path.dirname(os.path.join(self.input_path)), fragment['filename'].replace('obj', 'png'))
                raw_image = plt.imread(fragment_path)
                
                self.raw_images.append(raw_image)
                self.extended_raw_images.append(raw_image)
                frag_name = fragment['filename'].split('.')[0]
                self.gt['pieces'][j] = {
                    'idx': fragment['idx'],
                    'name': frag_name,
                    'x': fragment['pixel_position'][0], # + shift[0],
                    'y': -fragment['pixel_position'][1], # - shift[1],
                    'theta': fragment['pixel_position'][2]
                }
                raw_mask, raw_polygon, enclosing_radius, wh = self.extract_mask_and_polygon_irregular_piece(raw_image)
                self.raw_masks.append(raw_mask)
                self.extended_raw_masks.append(raw_mask)
                self.raw_polygons.append(raw_polygon)
                self.extended_raw_polygons.append(raw_polygon)
                self.sizes.append(wh)
                self.radii.append(enclosing_radius)
                if enclosing_radius > self.max_enclosing_radius:
                    self.max_enclosing_radius = enclosing_radius

            print("processing..")
            for j, fragment in enumerate(fragments):
                
                frag_name = fragment['filename'].split('.')[0]
                print(f"fragment {frag_name}")
                self.extended_raw_images[j] = cv2.copyMakeBorder(self.raw_images[j], self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT, (0, 0, 0, 0))
                # np.zeros((self.raw_images[j].shape[0] + self.padding, self.raw_images[j].shape[1] + self.padding, self.raw_images[j].shape[2]))
                self.extended_raw_masks[j] = cv2.copyMakeBorder(self.raw_masks[j], self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT, (0))
                # np.zeros((self.raw_masks[j].shape[0] + self.padding, self.raw_masks[j].shape[1] + self.padding))
                self.extended_raw_polygons[j] = transform(self.raw_polygons[j], lambda f: f + [self.padding, self.padding])

                pixel_position = np.asarray(fragment['pixel_position'][:2]) + np.asarray([self.padding, self.padding])

                ### temporal solution .... ONLY FOR RENDERING output FROM THEO !!! USE line 177 otherwise
                # pixel_position = np.asarray(fragment['pixel_position'][:2])
                # pixel_position[1] = self.raw_images[j].shape[1] - pixel_position[1]
                # pixel_position += np.asarray([self.padding, self.padding])
                #################################

                # this is the pixel position of the center of mass of the fragment (so we can process it and maintain the GT)
                # pixel_position = np.asarray([self.gt['pieces'][j]
                # pnts_hom = np.hstack((np.asarray(pnts), np.ones((np.asarray(pnts).shape[0], 1)))) # transform position points to homogeneous
                # pixels = (inv_tform @ pnts_hom.T).T[:,0:2]

                # print(f"pixels: {pixels}, pixel_position: {fragment['pixel_position']}")

                # plt.subplot(121)
                # plt.imshow(self.extended_raw_images[j])
                # plt.scatter(pixel_position[0], pixel_position[1], s=25, c='red')
                # plt.plot(*self.extended_raw_polygons[j].boundary.xy)
                
                # plt.subplot(122)
                # plt.imshow(self.extended_raw_masks[j])
                # plt.scatter(pixel_position[0], pixel_position[1], s=25, c='red')
                # plt.plot(*self.extended_raw_polygons[j].boundary.xy)
                # plt.show()
                # breakpoint()
                
                self.names.append(frag_name)
                image, mask, polygon, rescaling_factor = self.extract_everything_centered_at(piece_index=j, center=pixel_position, half_image_side=self.max_enclosing_radius, new_size=self.target_size)
                
                if add_random_rotations == True:
                    # breakpoint()
                    random_rot = np.floor(random.uniform(0, 8)) * theta_step
                    # plt.subplot(221); plt.imshow(image); plt.plot(*polygon.boundary.xy)
                    # plt.subplot(222); plt.imshow(mask)
                    image = scipy.ndimage.rotate(image, random_rot, reshape=False, mode='constant', order=0)
                    mask = scipy.ndimage.rotate(mask, random_rot, reshape=False, mode='constant', order=0)
                    polygon = shapely.affinity.rotate(polygon, -random_rot, origin=tuple([image.shape[0]/2, image.shape[1]/2]))
                    # plt.subplot(223); plt.imshow(rot_image); plt.plot(*rot_polygon.boundary.xy); plt.title(f"rotated {random_rot} degrees"); 
                    # plt.subplot(224); plt.imshow(rot_mask); plt.title(f"rotated {random_rot} degrees")
                    # plt.show()
                    # breakpoint()

                    self.random_rotations.append(random_rot)
                # plt.subplot(121)
                # plt.imshow(image)
                # # plt.scatter(pixel_position[0], pixel_position[1], s=25, c='red')
                # plt.plot(*polygon.boundary.xy)
                #
                # plt.subplot(122)
                # plt.imshow(mask)
                # # plt.scatter(pixel_position[0]+1000, pixel_position[1]+1000, s=25, c='red')
                # plt.plot(*polygon.boundary.xy)
                # plt.show()
                # breakpoint()

                self.images.append(image)
                self.masks.append(mask)
                self.polygons.append(polygon)                
                self.input_data[frag_name] = {
                    'idx': fragment['idx'],
                    'name': frag_name,
                    'image': image,
                    'mask': mask,
                    'polygon': polygon
                }

                self.gt['pieces'][j]['x'] = self.gt['pieces'][j]['x'] / rescaling_factor
                self.gt['pieces'][j]['y'] = image.shape[1] - self.gt['pieces'][j]['y'] / rescaling_factor
                
                self.xs.append(fragment['pixel_position'][0])
                self.ys.append(fragment['pixel_position'][1])
                fragment_rotation = fragment['pixel_position'][2]
                if add_random_rotations == True:
                    fragment_rotation += random_rot
                self.thetas.append(fragment['pixel_position'][2])
                self.positions.append(fragment['pixel_position'])

            self.puzzle_info['pieces_image_size'] = self.images[0].shape
            self.puzzle_info['binary_masks_available'] = True
            self.puzzle_info['polygons_available'] = True
            self.puzzle_info['ground_truth_available'] = True
            self.puzzle_info['max_dist_from_center'] = np.ceil(self.max_enclosing_radius).astype(float)
            self.puzzle_info['padding'] = self.padding
            self.puzzle_info['rescaling_factor'] = rescaling_factor
            # self.show_piece(3)
            # breakpoint()
            # print(self.sizes)

            # if crop_pieces == True:
            #     center_pix = np.round(self.puzzle_info['pieces_image_size'][0] / 2).astype(int)
            #     range_crop = self.padding + np.round(self.max_dist_from_center).astype(int)
            #     polygon_translation = center_pix - range_crop

            #     for frag_key in self.input_data.keys():

            #         orig_image_shape = self.input_data[frag_key]['image'].shape
            #         cropped_image = self.input_data[frag_key]['image'][center_pix-range_crop:center_pix+range_crop, center_pix-range_crop:center_pix+range_crop]
            #         cropped_mask = self.input_data[frag_key]['mask'][center_pix-range_crop:center_pix+range_crop, center_pix-range_crop:center_pix+range_crop]
            #         cropped_polygon = transform(self.input_data[frag_key]['polygon'], lambda f: f - [+polygon_translation,+polygon_translation])

            #         if new_size > 0:
            #             from skimage.transform import resize 
                        
            #             cropped_imcrop_everything_atage_size = cropped_image.shape[0]
            #             rescaling_factor = cropped_image_size / new_size
            #             cropped_image = resize(cropped_image, (new_size, new_size), anti_aliasing=True)
            #             cropped_mask = (resize(cropped_mask, (new_size, new_size), anti_aliasing=True, preserve_range=True) > 0.5).astype(np.uint8)
            #             cropped_polygon = transform(cropped_polygon, lambda f: f * new_size / cropped_image_size)

            #         # breakpoint()
            #         # self.gt['pieces'][self.input_data[frag_key]['idx']]['x'] /= rescaling_factor
            #         # self.gt['pieces'][self.input_data[frag_key]['idx']]['y'] /= rescaling_factor
            #         self.input_data[frag_key]['image'] = cropped_image
            #         self.input_data[frag_key]['mask'] = cropped_mask
            #         self.input_data[frag_key]['polygon'] = cropped_polygon
                
            #         # plt.suptitle(f"After resizing to {new_size}", fontsize=24)
            #         # plt.subplot(121)
            #         # plt.title("Image")
            #         # plt.imshow(self.input_data[frag_key]['image'])
            #         # plt.scatter(self.input_data[frag_key]['image'].shape[0] / 2, self.input_data[frag_key]['image'].shape[1] / 2, s=15)
            #         # plt.plot(*(self.input_data[frag_key]['polygon'].boundary.xy), linewidth=3)
                    
            #         # plt.subplot(122)
            #         # plt.title("Binary Mask")
            #         # plt.imshow(self.input_data[frag_key]['mask'])
            #         # plt.plot(*(self.input_data[frag_key]['polygon'].boundary.xy), linewidth=3)
            #         # plt.show()
            #         # breakpoint()

    def extract_everything_centered_at(self, piece_index:int, center:np.ndarray, half_image_side:float, new_size:int = 0):
        """ extract all the information from the given center of mass """

        # breakpoint()
        # int_square_size = np.ceil(square_size + padding).astype(int)
        # if int_square_size % 2 == 1:
        #     int_square_size += 1
        half_image_side = np.ceil(half_image_side).astype(int)
        center = np.round(center).astype(int)

        # breakpoint()
        # plt.suptitle(f"Before", fontsize=24)
        # plt.subplot(421)
        # plt.title("Before")
        # plt.imshow(self.raw_images[piece_index])
        # plt.scatter(center[0], center[1], s=15, c='red')
        # plt.scatter(self.raw_images[piece_index].shape[0] / 2, self.raw_images[piece_index].shape[1] / 2, s=15)
        # plt.plot(*(self.raw_polygons[piece_index].boundary.xy), linewidth=3)
        
        # plt.subplot(422)
        # plt.imshow(self.raw_masks[piece_index])
        # plt.scatter(center[0], center[1], s=15, c='red')
        # plt.scatter(self.raw_masks[piece_index].shape[0] / 2, self.raw_masks[piece_index].shape[1] / 2, s=15)
        # plt.plot(*(self.raw_polygons[piece_index].boundary.xy), linewidth=3)

        # plt.subplot(423)
        # plt.title("Before")
        # plt.imshow(self.extended_raw_images[piece_index])
        # plt.scatter(center[0], center[1], s=15, c='red')
        # plt.scatter(self.extended_raw_images[piece_index].shape[0] / 2, self.raw_images[piece_index].shape[1] / 2, s=15)
        # plt.plot(*(self.raw_polygons[piece_index].boundary.xy), linewidth=3)
        
        # plt.subplot(424)
        # plt.imshow(self.extended_raw_masks[piece_index])
        # plt.scatter(center[0], center[1], s=15, c='red')
        # plt.scatter(self.extended_raw_masks[piece_index].shape[0] / 2, self.raw_masks[piece_index].shape[1] / 2, s=15)
        # plt.plot(*(self.raw_polygons[piece_index].boundary.xy), linewidth=3)


        # cropped_image = self.raw_images[piece_index][center[1]-self.sizes[piece_index][1]//2: center[1]+self.sizes[piece_index][1]//2+1, center[0]-self.sizes[piece_index][0]//2: center[0]+self.sizes[piece_index][0]//2+1, :]
        # cropped_mask = self.raw_masks[piece_index][center[1]-self.sizes[piece_index][1]//2: center[1]+self.sizes[piece_index][1]//2+1, center[0]-self.sizes[piece_index][0]//2: center[0]+self.sizes[piece_index][0]//2+1]
        # polygon_translation = center - np.asarray([half_image_side, half_image_side])
        # cropped_polygon = transform(self.raw_polygons[piece_index], lambda f: f - polygon_translation)

        # breakpoint()
        cropped_image = self.extended_raw_images[piece_index][center[1]-half_image_side: center[1]+half_image_side+1, center[0]-half_image_side: center[0]+half_image_side+1, :]
        cropped_mask = self.extended_raw_masks[piece_index][center[1]-half_image_side: center[1]+half_image_side+1, center[0]-half_image_side: center[0]+half_image_side+1]
        polygon_translation = center - np.asarray([half_image_side, half_image_side])
        cropped_polygon = transform(self.extended_raw_polygons[piece_index], lambda f: f - polygon_translation)

        # # # plt.suptitle(f"After cropping", fontsize=24)
        # plt.subplot(425)
        # plt.title(f"Crop (ihs:{half_image_side}, s:{cropped_image.shape}")
        # plt.imshow(cropped_image)
        # plt.scatter(cropped_image.shape[0] / 2, cropped_image.shape[1] / 2, s=15)
        # plt.plot(*(cropped_polygon.boundary.xy), linewidth=3)

        # plt.subplot(426)
        # plt.imshow(cropped_mask)
        # plt.scatter(cropped_mask.shape[0] / 2, cropped_mask.shape[1] / 2, s=15)
        # plt.plot(*(cropped_polygon.boundary.xy), linewidth=3)

        if all(x > 0 for x in new_size):
            new_size = new_size[0] 
                
            cropped_image_size = cropped_image.shape[0]
            rescaling_factor = cropped_image_size / new_size
            cropped_image = resize(cropped_image, (new_size, new_size), anti_aliasing=True)
            cropped_mask = (resize(cropped_mask, (new_size, new_size), anti_aliasing=True, preserve_range=True) > 0.5).astype(np.uint8)
            cropped_polygon = transform(cropped_polygon, lambda f: f * new_size / cropped_image_size)


        # # # plt.suptitle(f"After resizing to {new_size}", fontsize=24)
        # plt.subplot(427)
        # plt.title("Resize")
        # plt.imshow(cropped_image)
        # plt.scatter(cropped_image.shape[0] / 2, cropped_image.shape[1] / 2, s=15)
        # plt.plot(*(cropped_polygon.boundary.xy), linewidth=3)

        # plt.subplot(428)
        # plt.imshow(cropped_mask)
        # plt.scatter(cropped_mask.shape[0] / 2, cropped_image.shape[1] / 2, s=15)
        # plt.plot(*(cropped_polygon.boundary.xy), linewidth=3)
        # plt.show()
        # breakpoint()
        return cropped_image, cropped_mask, cropped_polygon, rescaling_factor

    def preprocess_and_center_irregular_piece(self, raw_image, method='np'):
        bmask = extract_binary_mask(raw_image)
        polygon, enclosing_radius = extract_polygon(bmask, return_vals=True)
        cm = calculate_center_of_mass(bmask, method=method)
        cen_image, cen_bmask, cen_polygon, shift = Puzzle.center_piece(raw_image, bmask, polygon, cm)  
       
        # plt.subplot(131)
        # plt.imshow(raw_image)
        # plt.scatter(raw_image.shape[0] / 2, raw_image.shape[1] / 2, s=15)
        # plt.plot(*(polygon.boundary.xy), linewidth=3)

    def extract_mask_and_polygon_irregular_piece(self, raw_image):

        bmask = extract_binary_mask(raw_image)
        polygon, enclosing_radius, wh = extract_polygon(bmask, return_vals=True)
        return bmask, polygon, enclosing_radius, wh

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

        return cen_image, cen_bmask, cen_polygon, np.asarray([shift_x, shift_y])

    def save(self):
        # breakpoint()
        print("saving..")
        images_out_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(images_out_dir, exist_ok=True)
        bmasks_out_dir = os.path.join(self.output_dir, 'binary_masks')
        os.makedirs(bmasks_out_dir, exist_ok=True)        
        polygons_out_dir = os.path.join(self.output_dir, 'polygons')
        os.makedirs(polygons_out_dir, exist_ok=True)
        if self.input_type == 'repair' or self.input_type == 'json':
            for frag_key in self.input_data.keys():
                frag_data = self.input_data[frag_key]

                # for name, image, bmask, polygon in zip(self.names, self.images, self.masks, self.polygons):
                plt.imsave(os.path.join(images_out_dir, f"{frag_data['idx']}_{frag_data['name']}.png"), frag_data['image'])
                cv2.imwrite(os.path.join(bmasks_out_dir, f"{frag_data['idx']}_{frag_data['name']}.png"), frag_data['mask'])
                np.save(os.path.join(polygons_out_dir, f"{frag_data['idx']}_{frag_data['name'][:-4]}"), frag_data['polygon'])
        else:
            for p_name in self.pieces.keys():
                piece = self.pieces[p_name]
                plt.imsave(os.path.join(images_out_dir, f"{p_name}.png"), piece['centered_image'][:,:,::-1])
                # cv2.imwrite(os.path.join(images_out_dir, f"piece_{j:04d}.png"), np.round(piece['centered_image']*255).astype(np.uint8))
                cv2.imwrite(os.path.join(bmasks_out_dir, f"mask_{p_name}.png"), piece['centered_mask'])
                np.save(os.path.join(polygons_out_dir, f"polygon_{p_name}.png"), piece['centered_polygon'])

        with open(os.path.join(self.output_dir, "ground_truth.json"), 'w') as jf:
            json.dump(self.gt, jf, indent=2)
        with open(os.path.join(self.output_dir, "puzzle_info.json"), 'w') as jf:
            json.dump(self.puzzle_info, jf, indent=2)
        np.savetxt(os.path.join(self.output_dir, "ground_truth.txt"), np.asarray(list(self.gt.values())))
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

    # def create_pieces(self):
    #     # puzzle_data, pieces, solution = 
    #     return 1,1,1