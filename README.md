# AgriSentinel — End-to-End Rice Disease Prototype

## Goal
Build a working demo:
Image -> Vision Model -> Evidence Fusion -> Risk -> Advisory -> Follow-up -> Public Web App

## Recommended dataset
Rice Leaf Disease Classification Dataset:
https://www.kaggle.com/datasets/anshuhlm257/rice-disease-dataset/data

It contains 3,829 RGB rice-leaf images in six categories:
healthy_rice_leaf, bacterial_leaf_blight, brown_spot, leaf_blast, leaf_scald, sheath_blight.

For the first hackathon MVP, train on FOUR classes:
- healthy_rice_leaf
- bacterial_leaf_blight
- brown_spot
- leaf_blast

The source repository states CC BY 4.0 and provides the dataset structure and split information.

## Folder after download

Put the downloaded dataset under:
data/raw/

Expected source structure:
data/raw/rice_leaves/<class>/images/*.jpg

## Install

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Prepare four classes

python prepare_data.py

This creates:
data/processed/train/<class>
data/processed/val/<class>
data/processed/test/<class>

## Train

python train.py

Output:
models/agrisentinel_rice_efficientnet.pth
outputs/classification_report.txt
outputs/confusion_matrix.png
outputs/training_curve.png

## Test one image

python predict.py "path/to/image.jpg"

## Run the prototype

streamlit run app.py

Local URL:
http://localhost:8501

## Deploy
1. Create a GitHub repository.
2. Upload app.py, model, requirements.txt and required project files.
3. Open Streamlit Community Cloud.
4. Select repository, branch and app.py.
5. Deploy.
6. Share the generated streamlit.app URL.

## Important
Do not present illustrative risk scores or demo weather values as measured research results.
Do not invent pesticide dosage. Use verified agricultural authority/label information.
