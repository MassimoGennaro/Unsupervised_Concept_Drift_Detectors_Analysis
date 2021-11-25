# Unsupervised Concept Drift Detectors Analysis
This project was developed during my master thesis at [Politecnico di Milano](https://www.polimi.it/), in collaboration with [Bip](https://www.bipconsulting.com/).

## Goal
The goal of the thesis is understanding the abilities of state of the arts unsupervised concept drift detection used in the scenario of the classic batch machine learning.
To do so a testing framework was developed in order to simulate the aforementioned use case and the techniques found has been adapted to the working assumptions.

## Unsupervised Concept Drift Detectors
The tested methodologies can be found [here](https://github.com/MassimoGennaro/Unsupervised-Concept-Drift-Detectors-Analysis/tree/main/papers), they are:
- Hellinger Distance Drift Detection Method - https://github.com/FabianHinder/drifting-data-in-continuous-time
- Dynamic Adapting Window Independence Drift Detection - https://github.com/FabianHinder/DAWIDD
- Discriminative Drift Detector - https://github.com/ogozuacik/d3-discriminative-drift-detector-concept-drift
- Student Teacher Approach - https://github.com/vcerqueira/unsupervised_concept_drift

## Usage
All the experiments can be launched using [main.py](https://github.com/MassimoGennaro/Unsupervised-Concept-Drift-Detectors-Analysis/blob/main/main.py), inserting which models and data sets must be used and how many repetitions of the experiment perform.

The concept drift detectors will be launched with default hyper parameters, which can be modified in the implementation file of each methods (<method_name>.py) changing the values inside the "<method_name>_inference" function. 

The results will be output in the 'result' folder as a json file for each data set.
