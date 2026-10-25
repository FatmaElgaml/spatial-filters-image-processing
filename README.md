# Spatial Filters – Image Processing Assignment

Python GUI (Tkinter + Matplotlib + OpenCV). All steps are shown in a single grid plot.

## Run
```bash
pip install -r requirements.txt
python app.py
```
Put `veggies.png` and `sunflower.png` in `images/` (loaded automatically) or use **Load image**.

## Gray / RGB
A black & white image is processed in Grayscale. For a colour image you choose Grayscale or RGB (RGB filters each channel independently).

## Experiment 1 – `veggies.png`
Gaussian noise → Average, Median, Gaussian filters with 3x3 and 5x5 windows.

## Experiment 2 – `sunflower.png`
Prewitt, Sobel, Laplacian (8-neighbour mask) edge detection → sharpening with each result.

## Observations (verify against your own run)
### Experiment 1
- Gaussian noise appears as grain over the whole image.
- Average reduces noise but blurs edges; Gaussian blurs less harshly (centre pixels weigh more); Median keeps edges better.
- 5x5 removes more noise than 3x3 but gives a blurrier image.

### Experiment 2
- Sobel and Prewitt (1st derivative) give similar gradient-magnitude edges; Sobel is slightly smoother (centre row/column weighted more).
- Laplacian (2nd derivative) highlights fine details and is the most sensitive to noise.
- Sharpening adds the edges back to the original, making edges and texture crisper; Laplacian sharpening is the strongest but can amplify noise.
