> [!IMPORTANT]  
> This branch `AI-docs` is an experiments where Claude has been used to generate README, example and documentation (from the code, but hopefully without modifying the code!) - treat it with caution as it is experimental, the idea is to see whether this extra documentation is useful when using the code.

# Jigsaw Puzzle Toolkit 

A comprehensive Python toolkit for creating jigsaw puzzles from images. This toolkit supports multiple puzzle types (squared, irregular, polyomino, pattern-based) with configurable rotation options, and provides complete ground truth data for puzzle assembly research.

## Overview

This repository provides tools to:
- **Generate puzzles** from images by cutting them into pieces
- **Process existing puzzles** from fragment collections
- **Create datasets** with multiple puzzle configurations
- **Extract ground truth** information for puzzle assembly algorithms

The toolkit originated from research in puzzle assembly and fragment reconstruction, providing a standardized way to create benchmark datasets.

## Features

- **Multiple Piece Types**: Squared, irregular (smooth/segmented cuts), polyomino, and pattern map-based
- **Rotation Support**: No rotation, 90° increments, or free rotation
- **Adjacency Detection**: Automatic calculation of which pieces are neighbors
- **Ground Truth**: Complete solution data including positions, rotations, and adjacency
- **Flexible Input**: Work with images or existing puzzle piece collections
- **Batch Processing**: Create multiple puzzles from image collections

## Installation

```bash
# Clone the repository
git clone https://github.com/CVML-CFU/jigsaw_puzzle_toolkit.git
cd jigsaw_puzzle_toolkit

# Install dependencies
pip install numpy opencv-python shapely matplotlib scipy scikit-image natsort
```

## Quick Start

### Basic Usage

Create an irregular puzzle with 25 pieces and 90° rotations:

```bash
python create.py -I ~/path/image.jpeg -PT I2 -NP 25 -O ~/output -S 500
```

### Command-Line Arguments

- `-I, --input`: Path to input image or directory
- `-PT, --puzzle_type`: Puzzle type (e.g., I2, S1, P3) - **required**
- `-O, --output`: Output directory for puzzle files
- `-NP, --num_pieces`: Number of pieces to create (default: 9)
- `-S, --new_size`: Target size for pieces in pixels (default: 251)
- `-M, --pattern_map`: Path to pattern map (for M/P types)
- `-IT, --input_type`: Input type: 'image', 'repair', or 'json' (default: 'image')

## Puzzle Types

Puzzles are identified by a two-character code: `{SHAPE}{ROTATION}`

### Piece Shapes

- **S** - Squared pieces: Regular grid cutting
- **I** - Irregular pieces: Smooth or segmented random curves
- **P** - Polyomino pieces: Tetris-like shapes centered on grid squares
- **M** - Pattern Map pieces: Custom cutting patterns from an image

### Rotation Types

- **1** - No rotation: Pieces maintain their original orientation
- **2** - 90° rotations: Pieces can be rotated by 0°, 90°, 180°, or 270°
- **3** - Free rotation: Pieces can be rotated to any angle

### Available Combinations

```
S1: Squared pieces, no rotation
S2: Squared pieces, 90° rotations

P1: Polyomino pieces, no rotation
P2: Polyomino pieces, 90° rotations
P3: Polyomino pieces, free rotations

M1: Pattern map pieces, no rotation
M2: Pattern map pieces, 90° rotations
M3: Pattern map pieces, free rotations

I1: Irregular pieces, no rotation
I2: Irregular pieces, 90° rotations
I3: Irregular pieces, free rotations
```

## Examples

### Single Puzzle from Image

Create a 16-piece irregular puzzle with free rotations:

```bash
python create.py \
    -I photos/landscape.jpg \
    -PT I3 \
    -NP 16 \
    -O puzzles/ \
    -S 512
```

### Polyomino Puzzle with Pattern Map

Create a polyomino puzzle using a tetromino pattern:

```bash
python create.py \
    -I images/photo.jpg \
    -PT P2 \
    -M patterns/tetromino.png \
    -O output/ \
    -S 400
```

### Batch Processing

Create multiple puzzles from a directory of images:

```bash
python create_from_image_collection.py \
    -I dataset/images/ \
    -PT I2 \
    -NP 25 \
    -O dataset/puzzles/ \
    -S 500
```

### Processing Existing Pieces

Convert an existing puzzle to a new format:

```bash
python create.py \
    -I old_puzzle/ \
    -IT repair \
    -PT I2 \
    -O reformatted/ \
    -S 256
```

## Output Structure

Each puzzle is saved in its own directory with the following structure:

```
{PUZZLE_TYPE}_{IMAGE_NAME}_{SIZE}/
├── images/              # PNG images of all pieces
│   ├── piece_0000.png
│   ├── piece_0001.png
│   └── ...
├── binary_masks/        # Binary masks for each piece
│   ├── piece_0000.png
│   └── ...
├── polygons/            # Numpy arrays of piece boundaries
│   ├── piece_0000.npy
│   └── ...
├── ground_truth.json    # Solution information
└── puzzle_info.json     # Puzzle metadata
```

### Ground Truth Format

```json
{
  "pieces": {
    "0": {
      "name": "piece_0000",
      "x": 245.3,
      "y": 312.7,
      "theta": 45.2
    },
    ...
  },
  "adjacency": [
    [0, 1],
    [0, 4],
    [1, 2],
    ...
  ]
}
```

- `x, y`: Center of mass coordinates in the original image
- `theta`: Rotation angle in degrees
- `adjacency`: List of adjacent piece pairs

## Architecture

### Core Components

- **`puzzle.py`**: Core puzzle utilities
  - `PuzzleType` enum for puzzle configurations
  - `Puzzle` class for puzzle creation and management
  - Helper functions for mask/polygon extraction

- **`puzzle_generator.py`**: Piece extraction algorithms
  - Curve generation for irregular pieces
  - Region segmentation and extraction
  - Adjacency calculation
  - Rotation handling

- **`create.py`**: Single puzzle creation
  - Command-line interface for single puzzles
  - Handles images and existing piece collections

- **`create_from_image_collection.py`**: Batch puzzle creation
  - Processes multiple images in one run
  - Maintains consistent configuration across puzzles

### Workflow

1. **Input Loading**: Image or existing pieces are loaded
2. **Piece Generation**: Image is cut into pieces using the specified algorithm
3. **Piece Processing**: Pieces are centered, cropped, and rescaled
4. **Rotation (optional)**: Pieces are rotated based on puzzle type
5. **Ground Truth**: Positions, rotations, and adjacency are calculated
6. **Output**: All data is saved to disk

## Advanced Usage

### Custom Pattern Maps

For M-type (pattern map) puzzles, you can provide a custom cutting pattern:

```python
# Pattern map should be a grayscale image where each connected region
# becomes a piece. Create it programmatically or manually.
import numpy as np
from PIL import Image

# Example: Create a 4-piece pattern
pattern = np.zeros((100, 100), dtype=np.uint8)
pattern[:50, :50] = 1  # Top-left
pattern[:50, 50:] = 2  # Top-right
pattern[50:, :50] = 3  # Bottom-left
pattern[50:, 50:] = 4  # Bottom-right

Image.fromarray(pattern * 60).save('custom_pattern.png')
```

Then use it:

```bash
python create.py -I image.jpg -PT M2 -M custom_pattern.png -O output/
```

### Programmatic Usage

```python
from puzzle import PuzzleType, Puzzle

# Create a puzzle programmatically
puzzle = Puzzle(
    input_path='image.jpg',
    puzzle_type=PuzzleType.type_I2,
    output_path='output/',
    target_size=512
)

# Prepare the puzzle
puzzle.prepare_puzzle(num_pieces=25, crop_pieces=True)

# Access piece data
for piece_name, piece_data in puzzle.pieces.items():
    image = piece_data['squared_image']
    mask = piece_data['squared_mask']
    polygon = piece_data['squared_polygon']
    # Process pieces...

# Save when done
puzzle.save()
```

## Known Issues and Notes

### GTFS-Kit RuntimeError

If you encounter `RuntimeError: dictionary changed size during iteration` when using this toolkit alongside `gtfs_kit`, avoid placing breakpoints inside puzzle generation functions. Let piece generation complete before debugging.

### Rotation Artifacts (Polyomino)

Polyomino pieces with free rotations may have minor alignment artifacts at certain angles. This is a known issue documented in the code and does not affect most use cases.

### Pattern Map Requirements

- Pattern maps must have the same number of regions as the target number of pieces
- Regions should be well-separated (at least a few pixels apart)
- Each region becomes one piece

## Contributing

Contributions are welcome! Areas for improvement:
- Additional piece cutting algorithms
- Better rotation handling for polyomino pieces
- Optimization of adjacency detection
- Support for 3D puzzle visualization
- Web-based puzzle creator interface

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{jigsaw_puzzle_toolkit,
  title = {Jigsaw Puzzle Toolkit},
  author = {CVML-CFU},
  year = {2026},
  url = {https://github.com/CVML-CFU/jigsaw_puzzle_toolkit}
}
```

## License

[Add your license information here]

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- [Add contact information if desired]

## Acknowledgments

This toolkit was developed as part of research in puzzle assembly and fragment reconstruction at CVML-CFU.
