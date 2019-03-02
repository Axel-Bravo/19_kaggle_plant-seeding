# [Plant Seeding](https://www.kaggle.com/c/plant-seedlings-classification)
## Overview
Can you differentiate a weed from a crop seedling?

The ability to do so effectively can mean better crop yields and better stewardship of the environment.

The Aarhus University Signal Processing group, in collaboration with University of Southern Denmark, has recently released a dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages.

We're hosting this dataset as a Kaggle competition in order to give it wider exposure, to give the community an opportunity to experiment with different image recognition techniques, as well to provide a place to cross-pollenate ideas.

## Data
You are provided with a training set and a test set of images of plant seedlings at various stages of grown. Each image has a filename that is its unique id. The dataset comprises 12 plant species. The goal of the competition is to create a classifier capable of determining a plant's species from a photo. The list of species is as follows:

`Black-grass | Charlock | Cleavers | Common Chickweed | Common wheat | Fat Hen
Loose Silky-bent | Maize | Scentless Mayweed | Shepherds Purse | Small-flowered Cranesbill | Sugar beet`

### File description
__train.csv__ - the training set, with plant species organized by folder

__test.csv__ - the test set, you need to predict the species of each image

__sample_submission.csv__ - a sample submission file in the correct format

## Code
For execution use the `main.py` file.
