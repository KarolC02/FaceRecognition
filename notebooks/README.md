# Jupyter Notebooks

## Face Recognition Demo

**File**: `face_recognition_demo.ipynb`

A comprehensive notebook that walks through the complete face recognition pipeline:

1. **Face Detection** - See if MTCNN can detect faces in your images
2. **Face Extraction** - Visualize how faces are cropped and resized
3. **Preprocessing Visualization** - See how images are transformed (resize, normalize)
4. **Embedding Extraction** - Generate and visualize 128-dimensional face encodings
5. **Face Recognition** - Compare two faces and see if they match
6. **Interactive Testing** - Test with different thresholds

## Usage

1. **Start Jupyter**:
   ```bash
   source venv/bin/activate
   jupyter notebook
   ```

2. **Open the notebook**:
   - Navigate to `notebooks/face_recognition_demo.ipynb`
   - Click to open

3. **Update image paths**:
   - Set `your_image_path` to your reference photo
   - Set `test_image` to another photo to test

4. **Run cells**:
   - Run cells sequentially (Shift+Enter)
   - See visualizations at each step

## Features

- **Visual face detection** - See bounding boxes and landmarks
- **Preprocessing pipeline** - Step-by-step transformation visualization
- **Embedding visualization** - See the 128-dimensional face encoding
- **Recognition comparison** - Side-by-side face comparison with results
- **Interactive testing** - Try different thresholds

## Requirements

Make sure you have:
- Trained model at `data/models/best_model.pth`
- Jupyter installed: `pip install jupyter ipywidgets`
- Your test images ready

Enjoy exploring the face recognition pipeline! 🎉

