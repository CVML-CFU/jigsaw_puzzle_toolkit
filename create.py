"""
Single Puzzle Creation Script

This script creates a jigsaw puzzle from a single input source (image or existing pieces).
It handles the complete workflow: loading input, generating/processing pieces, and saving
the output puzzle data.

Usage
-----
Basic command:
    python create.py -I input.jpg -PT I2 -O output/ -S 500

With pattern map:
    python create.py -I input.jpg -PT M2 -M pattern.png -O output/ -S 500

From existing pieces:
    python create.py -I pieces_dir/ -IT repair -PT I2 -O output/ -S 500

Arguments
---------
--input, -I : str
    Path to input image or directory containing puzzle pieces
--input_type, -IT : str
    Type of input: 'image' (default), 'repair', or 'json'
--num_pieces, -NP : int
    Number of pieces to create (default: 9). Only used for 'image' input type.
--pattern_map, -M : str
    Path to pattern map image for M-type or P-type puzzles (default: '')
--puzzle_type, -PT : PuzzleType
    Type of puzzle to create (e.g., I2, S1, P3, M2)
--output, -O : str
    Path to output directory where puzzle files will be saved
--new_size, -S : int
    Target size for rescaling pieces in pixels (default: 251)

Puzzle Types
------------
Format: {SHAPE}{ROTATION}
- SHAPE: S (squared), P (polyomino), M (pattern map), I (irregular)
- ROTATION: 1 (none), 2 (90° increments), 3 (free rotation)

Examples: S1, I2, P3, M2

Output Structure
----------------
The script creates the following output structure:
    output/
        {PUZZLE_TYPE}_{IMAGE_NAME}_{SIZE}/
            images/           # PNG files of all pieces
            binary_masks/     # Binary masks for each piece
            polygons/         # Numpy arrays of piece polygons
            ground_truth.json # Solution information
            puzzle_info.json  # Puzzle metadata

Examples
--------
Create irregular puzzle with 25 pieces, 90° rotations allowed:
    $ python create.py -I landscape.jpg -PT I2 -NP 25 -O puzzles/ -S 512

Create polyomino puzzle from pattern map:
    $ python create.py -I photo.jpg -PT P2 -M patterns/tetromino.png -O puzzles/

Process existing puzzle pieces:
    $ python create.py -I old_puzzle/ -IT repair -PT I2 -O new_format/
"""

import os
import argparse
import shutil
from puzzle import PuzzleType, Puzzle

def main(args):
    """
    Main function to create a puzzle.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing puzzle configuration
    
    Returns
    -------
    int
        1 on successful completion
    
    Notes
    -----
    The function creates a Puzzle object, prepares it with the specified
    configuration, and saves all output files. For 'repair' or 'json' input
    types, it also copies preview images if they exist.
    """
    # Create Puzzle object with specified configuration
    puzzle = Puzzle(
        input_path=args.input,
        puzzle_type=args.puzzle_type,
        output_path=args.output,
        input_type=args.input_type,
        target_size=args.new_size
    )
    
    # Prepare the puzzle (generate or process pieces)
    puzzle.prepare_puzzle(
        num_pieces=args.num_pieces,
        crop_pieces=True,
        pattern_map_path=args.pattern_map
    )
    
    # Save all puzzle data to disk
    puzzle.save()
    
    # Copy preview images if available (for repair/json input types)
    if args.input_type == 'repair' or args.input_type == 'json':
        # Copy adjacency preview if it exists
        adjacency_preview = os.path.join(args.input, 'adjacency_preview.png')
        if os.path.exists(adjacency_preview):
            shutil.copy2(
                adjacency_preview,
                os.path.join(puzzle.output_dir, 'adjacency_preview.png')
            )
        
        # Copy general preview if it exists
        preview = os.path.join(args.input, 'preview.png')
        if os.path.exists(preview):
            shutil.copy2(
                preview,
                os.path.join(puzzle.output_dir, 'preview.png')
            )
    
    return 1


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Create a jigsaw puzzle from an image or existing pieces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 25-piece irregular puzzle with 90° rotations
  python create.py -I photo.jpg -PT I2 -NP 25 -O output/ -S 500
  
  # Create polyomino puzzle from pattern map
  python create.py -I image.jpg -PT P2 -M pattern.png -O output/
  
  # Process existing puzzle pieces
  python create.py -I old_puzzle/ -IT repair -PT I2 -O new_format/
        """
    )
    
    # Input/output arguments
    parser.add_argument(
        '--input', '-I',
        type=str,
        default='input',
        help='Path to input image or directory containing puzzle pieces'
    )
    parser.add_argument(
        '--input_type', '-IT',
        type=str,
        default='image',
        choices=['image', 'repair', 'json'],
        help="Type of input: 'image' for new puzzle, 'repair'/'json' for existing pieces"
    )
    parser.add_argument(
        '--output', '-O',
        type=str,
        default='output',
        help='Path to output directory where puzzle files will be saved'
    )
    
    # Puzzle configuration arguments
    parser.add_argument(
        '--num_pieces', '-NP',
        type=int,
        default=9,
        help='Number of pieces to create (only used when input_type is "image")'
    )
    parser.add_argument(
        '--pattern_map', '-M',
        type=str,
        default='',
        help='Path to pattern map image (required for M-type and P-type puzzles)'
    )
    parser.add_argument(
        '--puzzle_type', '-PT',
        type=PuzzleType,
        choices=list(PuzzleType),
        required=True,
        help='Puzzle type: format {SHAPE}{ROTATION} where SHAPE in [S,P,M,I] and ROTATION in [1,2,3]'
    )
    parser.add_argument(
        '--new_size', '-S',
        type=int,
        default=251,
        help='Target size in pixels for rescaling pieces (default: 251)'
    )
    
    # Parse arguments and run
    args = parser.parse_args()
    main(args)
