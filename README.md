# Jigsaw Puzzle Toolkit 

This repo is intended to be used for creating puzzles from images (breaking them into pieces) or directly from pieces. It comes from the work done in the repair project. 

## The Idea 
Starting from either image or folder with images, you create a `Puzzle` object which contains the functions to load data and save it (so look at `puzzle.py` for storage, paths, puzzle types and so on).
You then prepare the puzzle using the `PuzzleGenerator` to extract the pieces ( so look at `puzzle_generator.py` for the algorithm to generate puzzle, which now supports irregular pieces cut by generating random lines (smooth or segmented), using a pattern map, and also creating polyomino pieces centered on one of the square which constitutes them!)

## Using it
The simplest way is:
```bash
python create.py -I '~/path/image.jpeg' -PT I2 -O ~/output -S 500
```
- `-I` (or `--input`) is the input image to cut into pieces
- `-O` (or `--output`) is the input folder (new folders will be created here)
- `-PT` (or `--puzzle_type`) is the type of puzzle (see below for more info)
- `-S` (or `--new_size`) is the desired size for the image (it is cropped and resized before cutting)

## PuzzleTypes
We created a `PuzzleType` class with couple of helpful methods, and we classified puzzles like this:
``` 
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
```
You should use the two characters:
- for piece type (`I` for irregular, `M` for pattern map)
- for rotation type (`1` no rotations, `3` random rotations)

For example `-PT I1` creates irregular pieces without rotating them.