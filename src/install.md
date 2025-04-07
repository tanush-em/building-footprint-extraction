
For a detailed installation guide, use: [https://helloshreyas.com/how-to-install-detectron2-on-windows-machine](https://helloshreyas.com/how-to-install-detectron2-on-windows-machine)

Ensure that you have the following components before you start running the project:
- Anaconda Environment
- Visual Studio Build Tools (Desktop development with C++)
- Libraries and Packages:
  - torch
  - torchvision
  - cython
  - OpenCV
  - pyyaml
  - ninja
  - rasterio
  - streamlit
  - geopandas
  - pyproj
  - PIL
- Cloned detectron2 repository
- Download the model components from the below drive link

Step-by-Step Guide to Installing Detectron2 on Windows

1. Install Anaconda
   - Download from: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
   - Run and follow the installation prompts.

2. Install Visual Studio Build Tools
   - Download from: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Run the installer, and choose "Desktop development with C++."

3. Install Microsoft Visual C++ Build Tools
   - Download from: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Run the installer, select "Desktop development with C++," and complete the installation.

4. Create and Activate a Conda Environment
   - Open Anaconda Prompt.
   - Create a new environment:
     ```
     conda create --name detectron_env python=3.9
     ```
   - Activate the environment:
     ```
     conda activate detectron_env
     ```

5. Install Required Dependencies

   Without GPU
   - Install dependencies:
     ```
     pip install torch==1.13.1 torchvision==0.14.1 Cython==0.29.33 opencv-python==4.7.0.68 matplotlib==3.6.3 PyYAML==6.0 protobuf==4.21.12 ninja==1.11.1
     ```

   With GPU
   - Install CUDA and cuDNN from: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Install PyTorch with GPU support (adjust URL for different CUDA versions):
     ```
     pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu11.0/torch_stable.html
     ```
   - Install additional dependencies:
     ```
     pip install Cython==0.29.33 opencv-python==4.7.0.68 matplotlib==3.6.3 PyYAML==6.0 protobuf==4.21.12 ninja==1.11.1
     ```

6. Clone the Detectron2 Repository
   - Clone from GitHub:
     ```
     git clone https://github.com/facebookresearch/detectron2.git
     ```

7. Build and Install Detectron2
   - Navigate to Detectron2 directory:
     ```
     cd detectron2
     ```
   - Build Detectron2:
     ```
     python setup.py build develop
     ```
   - Install Detectron2:
     ```
     pip install -e .
     ```

8. Test the Installation
   - Verify installation:
     ```
     python -c "import detectron2; print(detectron2.__version__)"
     ```

9. After all the setup and installation, make sure the model components are downloaded and paths are setup in "app.py"
GDrive link - https://drive.google.com/drive/folders/1NfEOJ0nRyk7lmX5sN8A4gssq8TIBW-rh?usp=sharing
(Contains model weights and config file)

