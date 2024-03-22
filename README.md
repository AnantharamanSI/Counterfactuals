# Constrained & Optimised Counterfactuals for interpretable ML and AI bias detection
by Anantharaman S. Iyer  
(Submitted to IRIS National Science Fair on 6th  November 2021)

Machine Learning models are usually black boxes for their users. The biases in their decision-making can be difficult to detect. I describe a constrained-optimization method for generating counterfactuals that can be used on a wide variety of datasets and ML models to make those models interpretable and uncover any biases lurking in them. I demonstrate the effectiveness of my method with several different ML models using MNIST image dataset as well as numerical and categorical datasets in the form of  Indian Liver dataset and German Credit dataset.

This repository demonstrates my method on the MNIST image dataset.

## How to download and run the code
1. Simply download the code and copy it your desired folder path.

2. Create a virtual python environment by running the following:
```
python -m venv counterfactuals_env
pip install -r requirements.txt
```
4. Activate your environment:
```
# for windows 
counterfactuals_env\Scripts\activate

# for mac
source counterfactuals_env/bin/activate
```
4. Run `python generate_counterfactuals.py`

# Constrained & Optimised Counterfactuals for interpretable ML and AI bias detection
by Anantharaman S. Iyer  
(Submitted to IRIS National Science Fair on 6th  November 2021)

Machine Learning models are usually black boxes for their users. The biases in their decision-making can be difficult to detect. I describe a constrained-optimization method for generating counterfactuals that can be used on a wide variety of datasets and ML models to make those models interpretable and uncover any biases lurking in them. I demonstrate the effectiveness of my method with several different ML models using MNIST image dataset as well as numerical and categorical datasets in the form of  Indian Liver dataset and German Credit dataset.

This repository demonstrates the method on the MNIST image dataset.

## How to download and run the code
1. Simply download the code and copy it your desired folder path.

2. Create a virtual python environment by running the following:
```
python -m venv counterfactuals_env
pip install -r requirements.txt
```
4. Activate your environment:
```
# for windows 
counterfactuals_env\Scripts\activate

# for mac
source counterfactuals_env/bin/activate
```
4. Run `python generate_counterfactuals.py`

## How to interpret the results
We begin with a handwritten number three and seek to find the minimum tweaks to the pixels of the image such that the model recognizes the new digits as being the digits 8 and 7.
`results/counterfactual_8_iter.jpeg` will display the progression of tweaks to get to 8. Similarily for 7.
