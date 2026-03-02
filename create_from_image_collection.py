import os
import argparse
import natsort
from puzzle import PuzzleType, Puzzle

def main(args):

    input_images = natsort.natsorted(os.listdir(args.input))
    if args.puzzle_type._type() == 'M' or args.puzzle_type._type() == 'P':
        num_available_pattern_maps = len(os.listdir(args.pattern_map))
        assert(len(num_available_pattern_maps) == len(input_images)), "You should provide a pattern map for each image!"
    
    for input_image in input_images:
        
        print(f"Creating puzzle from {input_image}")
        full_path_image = os.path.join(args.input, input_image)
        puzzle = Puzzle(input_path=full_path_image, puzzle_type=args.puzzle_type, output_path=args.output, input_type='image', target_size=args.new_size)
        puzzle.prepare_puzzle(num_pieces = args.num_pieces, crop_pieces = True, pattern_map_path = args.pattern_map)
        puzzle.save()

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a puzzle')
    parser.add_argument('--input', '-I', type=str, default='input', help='path to a folder with images')
    parser.add_argument('--pattern_map', '-M', type=str, default='', help='path to a folder with pattern maps') 
    parser.add_argument('--num_pieces', '-NP', type=int, default=9)
    parser.add_argument('--puzzle_type', '-PT', type=PuzzleType, choices=list(PuzzleType), help='puzzle type')
    parser.add_argument('--output', '-O', type=str, default='output', help='path to the output where the puzzle files will be placed')
    parser.add_argument('--new_size', '-S', type=int, default=251, help='new size for rescaling the input image')

    args = parser.parse_args()
    main(args)
