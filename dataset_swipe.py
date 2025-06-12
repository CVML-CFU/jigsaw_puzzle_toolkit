"""
It creates a puzzle, given one image and some data.
A `puzzle` consists in a set of pieces (and possibly the solution)
`
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

import os 
import argparse 
from puzzle import PuzzleType, Puzzle
import natsort 
import shutil 

def main(args):
    
    puzzles = os.listdir(args.input_root)
    sorted_puzzles = natsort.natsorted(puzzles)
    for puzzle_name in sorted_puzzles:
        target_folder = os.path.join(args.output_root, puzzle_name)
        if os.path.exists(target_folder):
            print(f"puzzle {os.path.basename(target_folder)} already exists!")
        else:
            print("creating puzzle", os.path.basename(puzzle_name))
            puzzle_data_file = os.path.join(args.input_root, puzzle_name, 'data.json')
            puzzle = Puzzle(puzzle_data_file, args.puzzle_type, args.output_root)
            puzzle.load_input_data(crop_pieces=True)
            puzzle.save()
            # copy preview
            shutil.copy2(os.path.join(args.input_root, puzzle_name, 'adjacency_preview.png'), os.path.join(puzzle.output_dir, 'adjacency_preview.png'))
            shutil.copy2(os.path.join(args.input_root, puzzle_name, 'preview.png'), os.path.join(puzzle.output_dir, 'preview.png'))
            breakpoint()
        
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a puzzle')
    parser.add_argument('--input_root', '-I', type=str, default='input', help='path to dataset root folder (contains multiple folders, the puzzles)')
    parser.add_argument('--puzzle_type', '-PT', type=PuzzleType, choices=list(PuzzleType), help='puzzle type')
    parser.add_argument('--output_root', '-O', type=str, default='output', help='path to the output where the puzzle files will be placed')

    args = parser.parse_args()
    main(args)