"""
Batch Puzzle Creation Script

This script creates multiple jigsaw puzzles from a collection of images.
It processes each image in the input directory and generates a separate puzzle
for each one, maintaining consistent puzzle configuration across all images.

Usage
-----
Basic command:
    python create_from_image_collection.py -I images/ -PT I2 -O output/ -S 500

With pattern maps:
    python create_from_image_collection.py -I images/ -PT P2 -M patterns/ -O output/

Arguments
---------
--input, -I : str
    Path to directory containing input images
--pattern_map, -M : str
    Path to directory containing pattern maps (one per image, required for M/P types)
--num_pieces, -NP : int
    Number of pieces per puzzle (default: 9)
--puzzle_type, -PT : PuzzleType
    Type of puzzle to create for all images (e.g., I2, S1, P3, M2)
--output, -O : str
    Path to output directory where all puzzle folders will be created
--new_size, -S : int
    Target size for rescaling pieces in pixels (default: 251)

Pattern Map Requirements
------------------------
When using M-type (pattern map) or P-type (polyomino) puzzles:
- The pattern_map directory must contain the same number of pattern images
  as there are input images
- Pattern maps are matched to images by natural sort order
- Each pattern map defines the cutting pattern for its corresponding image

Output Structure
----------------
For each input image, creates:
    output/
        {PUZZLE_TYPE}_{IMAGE_NAME_1}_{SIZE}/
            images/           # Piece images
            binary_masks/     # Piece masks
            polygons/         # Piece polygons
            ground_truth.json # Solution
            puzzle_info.json  # Metadata
        {PUZZLE_TYPE}_{IMAGE_NAME_2}_{SIZE}/
            ...
        ...

Examples
--------
Create irregular puzzles from all images in a folder:
    $ python create_from_image_collection.py -I photos/ -PT I2 -NP 25 -O puzzles/ -S 512

Create polyomino puzzles with pattern maps:
    $ python create_from_image_collection.py -I images/ -PT P2 -M patterns/ -NP 16 -O out/

Create squared puzzles with 90° rotations:
    $ python create_from_image_collection.py -I dataset/ -PT S2 -NP 9 -O output/ -S 300

Notes
-----
- Images are processed in natural sort order (1, 2, 10 instead of 1, 10, 2)
- Each puzzle is independent and saved in its own subdirectory
- Failed puzzles will raise an error and stop processing
- All images must be valid image files (PNG, JPG, etc.)
"""

import os
import argparse
import natsort
from puzzle import PuzzleType, Puzzle

def main(args):
    """
    Create puzzles from a collection of images.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing batch puzzle configuration
    
    Returns
    -------
    int
        1 on successful completion of all puzzles
    
    Raises
    ------
    AssertionError
        If using M/P type puzzles and the number of pattern maps doesn't
        match the number of input images
    
    Notes
    -----
    Images are processed sequentially in natural sort order. Each image
    produces an independent puzzle with identical configuration parameters.
    """
    # Get sorted list of input images
    input_images = natsort.natsorted(os.listdir(args.input))
    
    # Validate pattern map requirement for M and P type puzzles
    if args.puzzle_type._type() == 'M' or args.puzzle_type._type() == 'P':
        num_available_pattern_maps = len(os.listdir(args.pattern_map))
        assert(num_available_pattern_maps == len(input_images)), \
            "You should provide a pattern map for each image!"
    
    # Process each image
    for input_image in input_images:
        print(f"Creating puzzle from {input_image}")
        
        # Construct full path to current image
        full_path_image = os.path.join(args.input, input_image)
        
        # Create Puzzle object for this image
        puzzle = Puzzle(
            input_path=full_path_image,
            puzzle_type=args.puzzle_type,
            output_path=args.output,
            input_type='image',  # Always 'image' for batch processing
            target_size=args.new_size
        )
        
        # Prepare the puzzle (generate pieces)
        puzzle.prepare_puzzle(
            num_pieces=args.num_pieces,
            crop_pieces=True,
            pattern_map_path=args.pattern_map
        )
        
        # Save puzzle data
        puzzle.save()
        
        print(f"  ✓ Completed {input_image}")
    
    print(f"\nSuccessfully created {len(input_images)} puzzles!")
    return 1


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Create multiple jigsaw puzzles from a collection of images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 25-piece irregular puzzles from all images
  python create_from_image_collection.py -I photos/ -PT I2 -NP 25 -O puzzles/ -S 512
  
  # Create polyomino puzzles with pattern maps
  python create_from_image_collection.py -I images/ -PT P2 -M patterns/ -O output/
  
  # Create 9-piece squared puzzles with rotations
  python create_from_image_collection.py -I dataset/ -PT S2 -NP 9 -O output/
        """
    )
    
    # Input/output arguments
    parser.add_argument(
        '--input', '-I',
        type=str,
        default='input',
        help='Path to directory containing input images'
    )
    parser.add_argument(
        '--output', '-O',
        type=str,
        default='output',
        help='Path to output directory where puzzle folders will be created'
    )
    
    # Pattern map argument
    parser.add_argument(
        '--pattern_map', '-M',
        type=str,
        default='',
        help='Path to directory with pattern maps (one per image, for M/P type puzzles)'
    )
    
    # Puzzle configuration arguments
    parser.add_argument(
        '--num_pieces', '-NP',
        type=int,
        default=9,
        help='Number of pieces to create per puzzle (default: 9)'
    )
    parser.add_argument(
        '--puzzle_type', '-PT',
        type=PuzzleType,
        choices=list(PuzzleType),
        required=True,
        help='Puzzle type for all images: format {SHAPE}{ROTATION} where SHAPE in [S,P,M,I] and ROTATION in [1,2,3]'
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
