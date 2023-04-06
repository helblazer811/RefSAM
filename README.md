
# RefSAM

This repository is for evaluating the basic performance of SAM on the Referring Image Segmementation task. Check out the SAM project [here]().

# The Naive Zero Shot Approach

The very basic approach we use is to:
1. Produce a referring expression representation using the CLIP language transformer.
2. Extract SAM masks from an image.
3. Embed the masked sections into a CLIP model to produce a representation of the section.
4. Compare the masked section representation to the representation of the referring expression.

The code for the approach can be found in ```model.py```
# Setup
## Install SAM
```
    pip install git+https://github.com/facebookresearch/segment-anything.git
```
## Load the SAM model in the ```pretrained/``` folder
I used the ```sam_vit_h_4b8939.pth``` model from the SAM repository. It can be found [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

## Load the data

Follow the directions in ```prepare_dataset.md``` to download and setup the evaluation dataset. 

# Run the evaluation
To evaluate the approach run. 

```
    python evaluate_on_refcoco.py
```