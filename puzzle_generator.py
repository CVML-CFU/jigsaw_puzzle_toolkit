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
# utils method
def check_outside(x, y, width, height):
    if x < 0 or x >= width or y < 0 or y >= height:
        return True
    else:
        return False

def clip_rect(x, y, width, height):
    x_new = max(0, min(x, width-1))
    y_new = max(0, min(y, height-1))
    return x_new, y_new

def new_array(dims, val):
    assert(type(dims) is int or type(dims) is tuple or type(dims) is list)
    if type(dims) is int:
        return [val for i in range(dims)]
    elif len(dims) == 1:
        return [val for i in range(dims[0])]
    else:
        return [new_array(dims[1:], val) for i in range(dims[0]) ]

def get_cm(mask):
    mass_y, mass_x = np.where(mask >= 0.5)
    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)
    return [cent_x, cent_y]

def get_polygon(binary_image):
    bin_img = binary_image.copy()
    bin_img = cv2.dilate(bin_img.astype(np.uint8), np.ones((2,2)), iterations=1)
    contours, _ = cv2.findContours(bin_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = contours[0]
    # should we remove 0.5 or it's just visualization?
    #shapely_points = [(point[0][0]-0.5, point[0][1]-0.5) for point in contour_points]  # Shapely expects points in the format (x, y)
    shapely_points = [(point[0][0]-0.5, point[0][1]-0.5) for point in contour_points]  # Shapely expects points in the format (x, y)
    if len(shapely_points) < 4:
        print('we have a problem, too few points', shapely_points)
        raise ValueError('\nWe have fewer than 4 points on the polygon, so we cannot create a Shapely polygon out of this points! Maybe something went wrong with the mask?')
    polygon = shapely.Polygon(shapely_points)
    return polygon
##############################
##############################
class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return "Vector(%r, %r)" % (self.x, self.y)
    
    def __abs__(self):
        return hypot(self.x, self.y)
    
    def __bool__(self):
        return bool(abs(self))
    
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
##############################
##############################
class PuzzleGenerator:

    def __init__(self, img, parameters:dict, pieces_centers=None):
        #img_name:str, rotations_type:int, pieces_type:str, pieces_centers=None, ):

        self.img = img
        self.img_size = self.img.shape[:2] # Height, Width, Channel
        self.aspect_ratio = self.img_size[0] / self.img_size[1]
        self.erosion_kernel_size = 7
        self.dilation_kernel_size = 51

        # name of the file without extension
        self.name = parameters.get('name', "no_name")
        self.padding = parameters.get('padding', 9)  #np.min(self.img.shape[:2]) // 30
        self.rotation_range = parameters.get('rotation_range', 180)  
        self.rotation_type = parameters.get('rotation_type', 1)  
        self.rotation_type_description = parameters.get('rotation_type_s', "no rotation")  
        self.pieces_type = parameters.get('pieces_type', "S")  
        self.pieces_type_description = parameters.get('pieces_type_s', "squared")  
        self.start_from = 0
        if self.pieces_type == "M" or self.pieces_type == "P":
            self.start_from = 1
        if pieces_centers is not None:
            self.pieces_centers = pieces_centers


    def get_smooth_curve(self, x_len, x_pt_n, x_offset, y_offset, x_step):

        x_arr = []
        y_arr = []

        for i in range(x_pt_n+1):

            if i == 0:
                x = 0
            elif i == x_pt_n:
                x = x_len - 1
            else:
                x = round(x_step * i + random.uniform(-x_offset, x_offset))
            y = round(random.uniform(-y_offset, y_offset))

            x_arr.append(x)
            y_arr.append(y)

        x_arr = list(set(x_arr))
        y_arr = y_arr[:len(x_arr)]
        x_arr.sort()

        if self.smooth_flag:
            if len(x_arr) >= 4:
                smooth_func = interpolate.interp1d(x_arr, y_arr, kind='cubic')
            elif len(x_arr) == 3:
                smooth_func = interpolate.interp1d(x_arr, y_arr, kind='quadratic')
            elif len(x_arr) == 2:
                smooth_func = interpolate.interp1d(x_arr, y_arr, kind='slinear')
            else:
                raise ValueError("The length of cutting points in x_arr must be larger than 0.")


        else:
            smooth_func = interpolate.interp1d(x_arr, y_arr, kind='linear')

        x_arr = np.arange(0, x_len, dtype=np.int32)
        y_arr = smooth_func(x_arr).astype(np.int32)
        # plt.plot(x_arr, y_arr, 'r')
        # plt.plot(x_arr_s, y_arr_s, 'b')
        # plt.show()

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
            # TODO: adjacency = {}
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
        for p_name in self.pieces.keys():
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
                    degrees = np.round(np.random.uniform(0, 3)).astype(np.uint8) * 90
                elif self.rotation_type == 3: # free deg rotation
                    degrees = random.uniform(-self.rotation_range, self.rotation_range)
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
            # TODO: adjacency = {}
        } 
        square_side = self.img.shape[0]
        if square_side // 2 == 0:
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
            if len(self.img.shape) > 2: 
                image_i = self.img * np.repeat(mask_i, self.img.shape[2]).reshape(self.img.shape)
            else:
                image_i = np.where(mask_i, self.img, bg_mat)
            poly_i = get_polygon(mask_i)
            cm_i = np.asarray(self.pieces_centers[f"{i}"][::-1]) + 1
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
            centered_img = np.zeros((square_side, square_side, 3))
            centered_mask = np.zeros((square_side, square_side))
            center_i = np.asarray([self.img.shape[0] / 2, self.img.shape[1] / 2])
            shift2center = (center_i - cm_i)#[::1]
            # print(f"{i}:{shift2center}")
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
        for p_name in self.pieces.keys():
            squared_img = np.zeros((self.sq_size, self.sq_size, 4))
            squared_img[:,:,:3] = self.pieces[p_name]['centered_image'][from_idx[0]:to_idx[0], from_idx[1]:to_idx[1], ::-1]
            squared_img[:,:,3] = np.sum(squared_img[:,:,:3], axis=2) > 0
            squared_mask = self.pieces[p_name]['centered_mask'][from_idx[0]:to_idx[0], from_idx[1]:to_idx[1]]
            # we remove the offset in the centered polygon to get it aligned
            xoffset = - (self.img.shape[1]-self.sq_size) / 2   # half of the distance from the square to the shape of the image!
            yoffset = - (self.img.shape[0]-self.sq_size) / 2
            squared_poly = shapely.affinity.translate(self.pieces[p_name]['centered_polygon'], xoff=xoffset, yoff=yoffset)
            if self.rotation_type > 1:
                if self.rotation_type == 2: # 90 deg rotation
                    degrees = np.round(np.random.uniform(0, 3)).astype(np.uint8) * 90
                elif self.rotation_type == 3: # free deg rotation
                    degrees = random.uniform(-self.rotation_range, self.rotation_range)
                self.gt['pieces'][j]['theta'] = degrees
                squared_img, squared_mask, squared_poly = self.rotate_piece(squared_img, squared_mask, squared_poly, degrees, method='ND')

            self.pieces[p_name]['squared_image'] = squared_img
            self.pieces[p_name]['squared_mask'] = squared_mask
            self.pieces[p_name]['squared_polygon'] = squared_poly
            self.pieces[p_name]['shift2square'] = np.asarray([xoffset, yoffset])

        return self.pieces, self.sq_size, self.gt

    def rotate_piece(self, squared_img, squared_mask, squared_poly, degrees, method='ND'):
        """ Rotate a piece, including the mask and the polygon """
        rot_origin = [squared_img.shape[0] // 2, squared_img.shape[1] // 2]

        if method == 'ND' or method == 'scipy':
            rotated_square_img = ndimage.rotate(squared_img, degrees, reshape=False, mode='constant')
            rotated_square_mask = ndimage.rotate(squared_mask, degrees, reshape=False, mode='constant')
        elif method == 'OPENCV' or method == 'WARP':
            # region_rot = ndimage.rotate(region_pad, degree, reshape=False, cval=bg_color)
            rotation_mat = cv2.getRotationMatrix2D((squared_img.shape[1]/2, squared_img.shape[0]/2), degrees, 1)
            rotated_square_img = cv2.warpAffine(squared_img, rotation_mat, rot_origin,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rotated_square_mask = cv2.warpAffine(squared_mask, rotation_mat, rot_origin,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_square_poly = shapely.affinity.rotate(squared_poly, -degrees, origin=rot_origin)
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

        self.rot_range = rot_range
        self.piece_n = piece_n
        self.w_n = math.floor(math.sqrt(piece_n)) # / self.aspect_ratio))
        self.h_n = self.w_n #math.floor(self.w_n * self.aspect_ratio)
        self.smooth_flag = smooth_flag
        self.alpha_channel = alpha_channel
        self.small_region_area_ratio = small_region_area_ratio
        self.missing_indices = []
        self.erosion = erosion
        self.borders = borders

        # print('\tInitial block in hori: %d, in vert: %d' % (self.w_n, self.h_n))
        # print('\tOffset rate h: %.2f, w: %.2f, small region: %.2f, rot: %.2f' %
        #     (offset_rate_h, offset_rate_w, small_region_area_ratio, rot_range))

        self.get_mask(offset_rate_h, offset_rate_w)
        self.get_regions()

        self.num_of_missing_fragments = np.floor(self.region_cnt * perc_missing_fragments / 100).astype(int)
        if self.num_of_missing_fragments > 0:
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

    x0 = np.min(np.where(image[:,:,3] > 0)[1]) - padding
    x1 = np.max(np.where(image[:,:,3] > 0)[1]) + padding
    y0 = np.min(np.where(image[:,:,3] > 0)[0]) - padding
    y1 = np.max(np.where(image[:,:,3] > 0)[0]) + padding

    if return_vals == True:
        return image[y0:y1, x0:x1, :], x0, x1, y0, y1
    return image[y0:y1, x0:x1, :]