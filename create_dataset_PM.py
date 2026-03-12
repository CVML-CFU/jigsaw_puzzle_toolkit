import os
import argparse
import natsort
from puzzle import PuzzleType, Puzzle
import random 
import yaml


def main(cfg):
    
    # IMAGES 
    input_images = natsort.natsorted(os.listdir(cfg['input']))
    # PATTERN MAPS
    pattern_maps = natsort.natsorted(os.listdir(cfg['pattern_maps_path']))
    puzzle_type = PuzzleType(cfg['puzzle_type'])
    if puzzle_type._type() == 'P':
        pattern_maps = [pm[:-4] for pm in pattern_maps if pm.endswith('.png')]
    # if puzzle_type._type() == 'M': 
    assert(len(pattern_maps) == len(input_images)), "You should provide a pattern map for each image!"
    total_size = len(pattern_maps)
    print(f"Got {total_size} objects!")
    # if puzzle_type._type() == 'P':
    #     assert(len(pattern_maps) == 2*len(input_images)), "You should provide a pattern map and the centers for each image! (2 files for each image!)"
    target_size = cfg.get('target_size', 0)
    monomino_square_size = cfg.get('monomino_square_size', 0)

    # SHUFFLING!
    if cfg['shuffle']:
        random.seed(cfg['seed'])
        random.shuffle(input_images)
        random.shuffle(pattern_maps)

    puzzle_id = 1
    
    for input_image, pattern_map in zip(input_images, pattern_maps):
        
        print('-' * 60)
        
        full_path_image = os.path.join(cfg['input'], input_image)
        full_path_pattern_map = os.path.join(cfg['pattern_maps_path'], pattern_map)
        # PM_name, PM_size = pattern_map.split("__")
        img_pt, img_id, img_cat, img_name = input_image.split("__")
        puzzle_name = f"{puzzle_id:05d}__{puzzle_type}__img__{img_id}__{img_cat}__{img_name.split('.')[0]}__pmap__{pattern_map}"
        if os.path.exists(os.path.join(cfg['output'], 'preprocessing', puzzle_name)) and cfg['skip_done']:
            print(f"Skipping {puzzle_name} as configured!")
        else:
            print(f"Creating puzzle from {input_image} and {pattern_map}\n\t --> {puzzle_name}\n")    
            puzzle = Puzzle(input_path=full_path_image, puzzle_name=puzzle_name, puzzle_type=puzzle_type, output_path=cfg['output'], input_type='image', target_size=target_size,
                            save_masks=cfg['save_masks'], save_polygons=cfg['save_polygons'])
            puzzle.prepare_puzzle(num_pieces = 0, crop_pieces = True, pattern_map_path = full_path_pattern_map, monomino_square_size = monomino_square_size)
            puzzle.save()
            # breakpoint()
        print(f"*** Completed {puzzle_id:05d}/{total_size:05d} ({((puzzle_id/total_size)*100):.03f} %) ***")
        puzzle_id += 1
        # breakpoint()

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a large dataset with different settings from a config file.\
     If you want to create a single puzzle run `create.py`. \
     If you want to create some puzzles, run `create_from_image_collection.py`')
    parser.add_argument('--input', '-I', type=str, default='dataset_config.yaml', help='path to the YAML file with all configs')

    args = parser.parse_args()

    with open(args.input, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
