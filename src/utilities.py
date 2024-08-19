# Importing Libraries and Packages
import os
import csv
import cv2
import math
import pyproj
import streamlit as st
import rasterio
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from rasterio.plot import show
from rasterio.enums import ColorInterp
from shapely.geometry import Polygon
from shapely import ops
import geopandas as gpd
import geopandas as gpd
from detectron2 import *
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# ------------------------------------------------------------------------------------------------ #
def create_working_dir(unique_name, base_dir):
    """
    Create a working directory for storing outputs related to a specific project or task.

    Parameters:
    - unique_name (str): A unique name for the directory.
    - base_dir (Path): The base directory where the working directory will be created.

    Returns:
    - Path: The path to the created working directory.
    """
    global WORKING_DIR
    WORKING_DIR = base_dir / unique_name
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    return WORKING_DIR

# ------------------------------------------------------------------------------------------------ #
def save_tif_as_png(file_path, png_path):
    """
    Convert a TIFF image to PNG format and save it.

    Parameters:
    - file_path (str): Path to the input TIFF file.
    - png_path (str): Path to save the converted PNG file.

    Returns:
    - str: Path to the saved PNG file.
    """
    with rasterio.open(file_path) as src:
        band1 = src.read(1)
        colormap = src.colormap(1)
        
        if colormap:
            num_colors = len(colormap)
            palette = []
            for i in range(num_colors):
                color = colormap[i]
                palette.extend(color[:3])
            palette = palette + [0] * (256 * 3 - len(palette))
            img = Image.fromarray(band1, mode='P')
            img.putpalette(palette)
        else:
            img = Image.fromarray(band1, mode='L')
    
    img.save(png_path)
    st.write(f"Converted TIF to PNG and saved into {png_path}")
    return png_path  

# ------------------------------------------------------------------------------------------------ #
def calculate_square_tile_size(image_width, image_height, num_tiles):
    """
    Calculate the size of square tiles to split an image into a specified number of tiles.

    Parameters:
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.
    - num_tiles (int): Total number of tiles to create.

    Returns:
    - int: The size of each tile (in pixels).
    """
    num_tiles_per_dimension = math.ceil(math.sqrt(num_tiles))
    tile_size_width = math.ceil(image_width / num_tiles_per_dimension)
    tile_size_height = math.ceil(image_height / num_tiles_per_dimension)
    tile_size = min(tile_size_width, tile_size_height)
    return tile_size

# ------------------------------------------------------------------------------------------------ #
def create_tiles(image_path, tile_size, tile_dir):
    """
    Create tiles from an image and save them in a specified directory.

    Parameters:
    - image_path (str): Path to the input image file.
    - tile_size (int): Size of each tile (in pixels).
    - tile_dir (Path): Directory where the tiles will be saved.

    Returns:
    - list: A list of paths to the saved tile images.
    """
    img = Image.open(image_path)
    img_width, img_height = img.size
    tiles = []

    for i in range(0, img_width, tile_size):
        for j in range(0, img_height, tile_size):
            box = (i, j, min(i + tile_size, img_width), min(j + tile_size, img_height))
            tile = img.crop(box)
            tile_path = tile_dir / f"tile_{i}_{j}.png"
            tile.save(tile_path)
            tiles.append(tile_path)

    return tiles

# ------------------------------------------------------------------------------------------------ #
def initialize_predictor(cfg_path, model_weights, device="cpu"):
    """
    Initialize the Detectron2 predictor with the specified configuration and model weights.

    Parameters:
    - cfg_path (str): Path to the Detectron2 configuration file.
    - model_weights (str): Path to the model weights file.
    - device (str): Device to use for inference ("cuda" for GPU, "cpu" for CPU).

    Returns:
    - DefaultPredictor: The initialized Detectron2 predictor.
    """
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = device  # Set to "cuda" for GPU or "cpu" for CPU
    
    predictor = DefaultPredictor(cfg)
    return predictor

# ------------------------------------------------------------------------------------------------ #
def extract_coordinates_from_tile(tile_path, predictor, epsilon=6.0, output_dir='coordinates'):
    """
    Extract polygon coordinates from an image tile using the specified predictor, and save them to a CSV file.

    Parameters:
    - tile_path (str): Path to the image tile file.
    - predictor (DefaultPredictor): The Detectron2 predictor instance.
    - epsilon (float): Epsilon parameter for contour approximation. (default value = 6.0)
    - output_dir (str): Directory to save the output CSV file.

    Returns:
    - str: Path to the saved CSV file containing polygon coordinates, or None if no masks are found.
    """
    img = cv2.imread(str(tile_path))
    outputs = predictor(img)

    if "pred_masks" in outputs["instances"]._fields:
        masks = outputs["instances"].pred_masks.to("cpu").numpy()
        all_polygons = []
        for mask in masks:
            polygons = mask_to_polygons(mask, epsilon)
            all_polygons.extend(polygons)

        polygon_data = []
        for i, polygon in enumerate(all_polygons):
            if isinstance(polygon, Polygon):
                coords = list(polygon.exterior.coords)
                formatted_coords = ["[{:.0f},{:.0f}]".format(point[0], point[1]) for point in coords]
                joined_coords = " ".join(formatted_coords)
                polygon_data.append([i+1, joined_coords])  # BuildingID starts from 1

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_csv = Path(output_dir) / (f"{Path(tile_path).stem}_coordinates.csv")

        df = pd.DataFrame(polygon_data, columns=["BuildingID", "px_coordinates"])
        df.to_csv(output_csv, index=False)
        return output_csv
    return None

# ------------------------------------------------------------------------------------------------ #
def mask_to_polygons(mask, epsilon=5.0):
    """
    Convert a binary mask to a list of polygons using contour approximation.

    Parameters:
    - mask (numpy.ndarray): The binary mask array.
    - epsilon (float): Epsilon parameter for contour approximation.

    Returns:
    - list: A list of Shapely Polygon objects representing the contours of the mask.
    """
    mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 4:
            polygon = Polygon(shell=approx[:, 0, :])
            if polygon.is_valid and not polygon.is_empty:
                polygons.append(polygon)
        else:
            print("Warning: Contour does not have enough points to form a polygon.")

    return polygons

# ------------------------------------------------------------------------------------------------ #
def adjust_coordinates_for_large_image(csv_files):
    """
    Adjust the coordinates in CSV files for large images by accounting for tile offsets.

    Parameters:
    - csv_files (list): List of paths to CSV files containing building coordinates.
    """
    for csv_file in csv_files:
        try:
            filename = os.path.basename(csv_file)
            tile_name, _ = os.path.splitext(filename)
            parts = tile_name.split('_')
            x_offset = int(parts[1])
            y_offset = int(parts[2])

            updated_rows = []

            with open(csv_file, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    building_id = row['BuildingID']
                    coords_str = row['px_coordinates']

                    coords_pairs = coords_str.split('] [')
                    coords_pairs = [coord.strip('[] ') for coord in coords_pairs]
                    adjusted_coords = []
                    for coord in coords_pairs:
                        x, y = map(int, coord.split(','))
                        adjusted_x = x + x_offset
                        adjusted_y = y + y_offset
                        adjusted_coords.append(f"[{adjusted_x},{adjusted_y}]")

                    updated_coords_str = ' '.join(adjusted_coords)
                    updated_rows.append({'BuildingID': building_id, 'px_coordinates': updated_coords_str})

            updated_csv_file = csv_file.replace('.csv', '_adjusted.csv')
            with open(updated_csv_file, mode='w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['BuildingID', 'px_coordinates']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(updated_rows)

            print(f"Adjusted coordinates saved to {updated_csv_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

# ------------------------------------------------------------------------------------------------ #
def convert_coordinates_to_shapefiles(csv_files, output_shapefile):
    """
    Convert CSV files with building coordinates into a single shapefile.

    Parameters:
    - csv_files (list): List of paths to CSV files containing building coordinates.
    - output_shapefile (str): Path to the output shapefile.
    """
    all_polygons = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                building_id = row['BuildingID']
                coords_str = row['px_coordinates']
                coords = eval(f"[{coords_str}]") 
                polygon = Polygon(coords)
                all_polygons.append({'geometry': polygon, 'BuildingID': building_id})
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    gdf = gpd.GeoDataFrame(all_polygons, geometry='geometry', crs='EPSG:4326')
    gdf.to_file(output_shapefile)
    print(f"Shapefile saved to {output_shapefile}")

# ------------------------------------------------------------------------------------------------ #