# Building Footprint Extraction  

## Overview  

This application automates the extraction of building footprints from raster images using deep learning and geospatial processing techniques. The tool provides a streamlined workflow for users to:  

- Upload raster images (.tif, .tiff)  
- Tile large images into smaller segments  
- Process tiles using a deep learning model  
- Georeference extracted building footprints  
- Download the final processed results  

## Features  

- **Raster Image Upload**: Supports geospatial raster formats (.tif, .tiff).  
- **Tiling System**: Splits large images into smaller tiles for efficient processing.  
- **Deep Learning-based Processing**: Uses a trained model to extract building footprints.  
- **Georeferencing**: Aligns extracted features with real-world coordinates.  
- **Downloadable Results**: Outputs structured CSV files containing extracted footprint coordinates.  

## Installation  

### Prerequisites  

Ensure you have the following installed:  

- Python 3.8+  
- pip  
- Virtual environment (recommended)  

## Usage  

The application follows a step-by-step workflow:  

1. **Upload Raster**  
   - Upload a geospatial raster file (.tif or .tiff).  
   - The system extracts metadata such as number of bands, dimensions, data type, and CRS (Coordinate Reference System).  

2. **Tiling**  
   - Select the number of tiles for processing.  
   - The tool divides the raster into smaller images for efficient computation.  

3. **Processing**  
   - Choose a processing device (CPU/GPU).  
   - The model extracts building footprints from tiles and saves them as CSV files.  

4. **Georeferencing**  
   - Converts extracted pixel coordinates into real-world geospatial coordinates.  

5. **Download**  
   - Outputs a compiled CSV file containing extracted building footprints.  

## Technologies Used  

- **Streamlit** – UI for interactive processing.  
- **Rasterio** – Geospatial raster file handling.  
- **OpenCV** – Image processing.  
- **Detectron2** – Deep learning model for feature extraction.  
- **Shapely & GeoPandas** – Geospatial data manipulation.  
- **PyProj** – Coordinate transformations.  

## License  

This is a proprietary project and any suspection of copying of IP wil lead to legal consequences.

## Contact  

For any inquiries or issues, please reach out via [tanushtm.work@gmail.com].  
