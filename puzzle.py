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

    def __init__(self, input_path: str, puzzle_type: PuzzleType, output_path: str, padding: int = 1000):    

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
        self.radii = []
        self.max_enclosing_radius = 0
        self.padding = padding

    def load_input_data(self, crop_pieces: bool = False, new_size: int = 0):
        """ Loads the data and handles the different use-cases. """


        if self.input_path.endswith('json'):
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
                #np.zeros((self.raw_masks[j].shape[0] + self.padding, self.raw_masks[j].shape[1] + self.padding))
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
                image, mask, polygon, rescaling_factor = self.extract_everything_centered_at(piece_index=j, center=pixel_position, half_image_side=self.max_enclosing_radius, new_size=new_size)
                
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

        if new_size > 0:
            from skimage.transform import resize 
                
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
        self.output_dir = os.path.join(self.output_path, os.path.basename(os.path.dirname(os.path.join(self.input_path))))
        os.makedirs(self.output_dir, exist_ok=True)
        # images_out_dir = os.path.join(self.output_dir, 'images')
        # os.makedirs(images_out_dir, exist_ok=True)
        # bmasks_out_dir = os.path.join(self.output_dir, 'binary_masks')
        # os.makedirs(bmasks_out_dir, exist_ok=True)        
        # polygons_out_dir = os.path.join(self.output_dir, 'polygons')
        # os.makedirs(polygons_out_dir, exist_ok=True)
        # for frag_key in self.input_data.keys():
        #     frag_data = self.input_data[frag_key]

        #     # for name, image, bmask, polygon in zip(self.names, self.images, self.masks, self.polygons):
        #     plt.imsave(os.path.join(images_out_dir, f"{frag_data['idx']}_{frag_data['name']}.png"), frag_data['image'])
        #     cv2.imwrite(os.path.join(bmasks_out_dir, f"{frag_data['idx']}_{frag_data['name']}.png"), frag_data['mask'])
        #     np.save(os.path.join(polygons_out_dir, f"{frag_data['idx']}_{frag_data['name']}"), frag_data['polygon'])
        
        # np.savetxt(os.path.join(self.output_dir, "ground_truth.txt"), np.asarray(self.positions))
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

    # def create_pieces(self):
    #     # puzzle_data, pieces, solution = 
    #     return 1,1,1