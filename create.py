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
import shutil
from puzzle import PuzzleType, Puzzle

def main(args):
    input_path = os.path.join(args.input, 'data.json')
    puzzle = Puzzle(input_path, args.puzzle_type, args.output)
    #puzzle = Puzzle(args.input, args.puzzle_type, args.output)
    puzzle.load_input_data(crop_pieces=True, new_size=args.new_size)
    # puzzle.create_pieces()
    puzzle.save()
    # copy preview
    shutil.copy2(os.path.join(args.input, 'adjacency_preview.png'),
                 os.path.join(puzzle.output_dir, 'adjacency_preview.png'))
    shutil.copy2(os.path.join(args.input, 'preview.png'),
                 os.path.join(puzzle.output_dir, 'preview.png'))
    
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a puzzle')
    parser.add_argument('--input', '-I', type=str, default='input', help='path to an image or a set of fragments')
    parser.add_argument('--puzzle_type', '-PT', type=PuzzleType, choices=list(PuzzleType), help='puzzle type')
    parser.add_argument('--output', '-O', type=str, default='output', help='path to the output where the puzzle files will be placed')
    parser.add_argument('--new_size', '-S', type=int, default=251, help='new size for rescaling')

    args = parser.parse_args()
    main(args)