# Importing Libraries and Packages
import streamlit as st
import rasterio
import numpy as np
import os
import math
from pathlib import Path
from PIL import Image
from rasterio.plot import show
from rasterio.enums import ColorInterp
import cv2
from shapely.geometry import Polygon
import geopandas as gpd
from utilities import *
import pyproj

# Define paths and directories
BASE_DIR = Path("./data")
MODEL_CFG_PATH = r"custom_cfg.yaml" # path to model config file - custom_cfg.yaml
MODEL_WEIGHTS_PATH = r"model_final.pth" # path to model weights file - model_final.pth

def main():
    """
    Main function to run the Streamlit application.

    This function sets up the Streamlit UI and manages the application's workflow:
    - Upload Raster Files
    - Tiling
    - Processing
    - Georeferencing
    - Download
    """
    st.title("Building Footprint Extraction")
    page = st.sidebar.selectbox("Select Page", ["Upload Raster", "Tiling", "Processing", "Georeferencing", "Download"])
    global WORKING_DIR

    if page == "Upload Raster":
        st.subheader("Upload Raster Files")
        st.text("NOTE: Upload only .tiff or .tif Files")
        st.text("It is recommended to rename the file")
        name = st.text_input("Enter unique name for this raster file")
        if name:
            WORKING_DIR = Path(create_working_dir(name, BASE_DIR))
            st.session_state['working_dir'] = WORKING_DIR  # Save to session state
        
        uploaded_file = st.file_uploader("Choose a Raster file", type=["tiff", "tif"])
        if uploaded_file and 'working_dir' in st.session_state:
            WORKING_DIR = st.session_state['working_dir']
            FILE_PATH = WORKING_DIR / uploaded_file.name

            with open(FILE_PATH, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File saved to {FILE_PATH}")
            
            with rasterio.open(FILE_PATH) as src:
                st.write("Number of bands:", src.count)
                st.write("Image width:", src.width)
                st.write("Image height:", src.height)
                st.write("Data type:", src.dtypes[0])
                st.write("CRS:", src.crs)
                st.write("Transform:", src.transform)
                st.write("Transform refers to Affine Transformation, which maps pixel coordinates to geographic coordinates.")
                img_width = src.width
                img_height = src.height
            
            PNG_PATH = WORKING_DIR / (uploaded_file.name.replace(".tif", ".png"))
            PNG_PATH = save_tif_as_png(FILE_PATH, PNG_PATH)
            st.success("Uploading Raster successful, you can move on to Tiling.")
            
    elif page == "Tiling":
        st.subheader("Tiling the Image")
        if 'working_dir' in st.session_state:
            WORKING_DIR = st.session_state['working_dir']
            img_files = list(WORKING_DIR.glob("*.png"))
            if img_files:
                PNG_PATH = img_files[0]  
                img = Image.open(PNG_PATH)
                img_width, img_height = img.size
                num_tiles_slider = st.slider("Select the number of tiles", min_value=9, max_value=100, value=0)
                st.text("Preferable to keep the number of tiles to be a square number")
        
                if st.button("Confirm Tiling Configuration"):
                    num_tiles = num_tiles_slider
                    st.success(f"Number of tiles set to: {num_tiles}")
                
                    if num_tiles:
                        TILE_SIZE = calculate_square_tile_size(img_width, img_height, num_tiles)
                        TILE_DIR = WORKING_DIR / "tiles"
                        TILE_DIR.mkdir(parents=True, exist_ok=True)  
                        st.session_state['tile_dir'] = TILE_DIR  
                        tiles = create_tiles(PNG_PATH, TILE_SIZE, TILE_DIR)
                        st.text("Tiling has been completed successfully.")
                        st.write(f"Created {len(tiles)} tiles.")
                        st.success("You can move on to Processing")
            else:
                st.error("No PNG files found in the working directory.")
        else:
            st.error("Please upload a raster file first.")       
  
    elif page == "Processing":
        """
        Page for processing tiles using the selected device.
        """
        device = st.selectbox("Select device to use for processing:", ("CPU", "GPU"))
        device_option = "cuda" if device == "GPU" else "cpu"
        predictor = initialize_predictor(MODEL_CFG_PATH, MODEL_WEIGHTS_PATH, device_option)
        
        if 'tile_dir' in st.session_state:  
            TILE_DIR = st.session_state['tile_dir']
            st.subheader("Processing Tiles")
            tile_files = list(TILE_DIR.glob("*.png"))
            if tile_files:
                st.write(f"Processing {len(tile_files)} tiles using {device}")
                for tile_path in tile_files:
                    csv_path = extract_coordinates_from_tile(tile_path, predictor, output_dir=TILE_DIR)
                    if csv_path:
                        st.write(f"Building Footprint Coordinates saved to {csv_path}")
                    else:
                        st.error(f"Failed to process {tile_path}")
                st.text("All tiles are processed successfully.")
                st.success("You can move on to Georeferencing.")
            else:
                st.warning("No tile images found in the directory.")
        else:
            st.error("Please complete tiling first.")
            
    elif page == "Georeferencing":
        """
        Page for georeferencing the processed tiles.
        """
        st.subheader("Georeferencing Tiles")
        
        if 'working_dir' in st.session_state:
            WORKING_DIR = st.session_state['working_dir']
            tif_files = list(WORKING_DIR.glob("*.tif"))
            
            if tif_files:
                tif_file = tif_files[0]  
                st.write(f"Georeferencing {tif_file}")
                
                if 'tile_dir' in st.session_state:
                    TILE_DIR = st.session_state['tile_dir']
                    csv_files = list(TILE_DIR.glob("*.csv"))
                    if csv_files:
                        for csv_file in csv_files:
                            try:
                                adjust_coordinates_for_large_image(csv_files)
                            except Exception as e:
                                st.error(f"Error georeferencing {csv_file}: {e}")
                        st.success("Georeferencing completed, move on to Download.")
                    else:
                        st.error("No CSV files found in the tile directory.")
                else:
                    st.error("Tile directory not found. Please complete the tiling and processing steps first.")
            else:
                st.error("No TIF files found in the working directory.")
        else:
            st.error("Please upload a raster file first and complete tiling and processing.")

    elif page == "Download":
        """
        Page for downloading the final output files.
        """
        st.write("Output Summary")
        TILE_DIR = st.session_state['tile_dir']
        WORKING_DIR = st.session_state['working_dir']
        output_filename = "compiled_px_coordinates.csv"
        combine_csv_files(TILE_DIR, WORKING_DIR, output_filename)
        st.success(f"Combined CSV file saved to {WORKING_DIR / output_filename}")
        
        tif_files = list(WORKING_DIR.glob("*.tif"))
        tif_file_path = tif_files[0]
        csv_files = list(WORKING_DIR.glob("*.csv"))
        csv_file_path = csv_files[0]
        output_file_name = "wkt_coordinates.csv"
        output_csv_file_path = Path(WORKING_DIR) / output_file_name
        convert_polygons_from_csv(tif_file_path, csv_file_path, output_csv_file_path)
    else:
        return
        
if __name__ == "__main__":
    main()
