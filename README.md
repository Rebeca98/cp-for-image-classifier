# Proyect structure
```
|-- LICENSE
|-- Makefile
|-- README.md
|-- data
|	|-- processed		<- data sets for modeling obtained by Megadetector and conabio_ml_vision library
|	|__ raw				<- Original data from Lila Repository
|	|__ test			<- data used for testing classification model and conformal-calibration algorithm
|
|-- models				<- Trained and serialized models for predictions
|
|-- notebooks			<- Jupyter notebooks. Naming convention is a number (for ordering), and a short `-` delimited description, e.g. `1.0-image-analysis`.
|
|-- reports				<- Generated analysis as Markdown.
|	|__ figures			<- Generated graphics and figures to be used in reporting
|
|-- setup.py			<- Make this project pip installable with `pip install -e`
|-- src					<- Source code for use int his project.
|-- data				<- Scripts to download or generate data.
|	|__split_data.py	<- Script to split data for classification model and conformal-calibration algorithm
|
|-- models				<- Scripts to train models and use the model for inference
|	|
|	|-- datasets.py		<- Script to transform data and create dataloader used in models.	
|	|-- train.py		<- Script to train image classifier
|	|-- inference.py	<- Script to test predictions of image classifier model
|	|-- utils.py		<- Script with functions used in other scripts.
|
|__ requierements.txt	<- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
```
