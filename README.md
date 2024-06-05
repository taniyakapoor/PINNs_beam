# PINNs_beam
Collection of code and data intended for replicating the experiments described in this research paper: (10.1109/TNNLS.2023.3310585)

## Folder Structure

The Data and code underlying the publication: Physics-Informed Neural Networks for Solving Forward and Inverse Problems in Complex Beam Systems are organized into five main folders: Advanced_DNN, Failure_first_case, Forward_Problems, Inverse_Problems and Numerical_methods. Below is a detailed description of each folder and their contents.

### Advanced_DNN

This folder contains codes for PGNN(physics guided neural networks) and gPINN(gradient enhanced physics informed neural networks). All implementation is done using jupyter notebook (.ipynb). To run the notebooks only need to run the cell. ".pth" files are trained model.

### Failure_first_case Folder

This folder contains codes to reproduce failure case of Euler-Bernoulli with PINNs in ipynb notebook.

### Forward_Problems

This folder contains five different folders which are cases for the forward problems of Euler-Bernoulli and Timoshenko single and double beam connected system. All these also contain jupyter notebooks (.ipynb) and trained models (.pth).

### Inverse_Problems
 
This folder also contains two folder for inverse problem of Timoshenko single and double beams. The csv files in the folders are data for deflection and rotations for beam systems. This dataset is simulated from forward problems. 

### Numerical_methods

In this folder finite difference method is implemented for Euler-Bernoulli and Timoshenko beam. 



## General Notes

- All codes, when run, will use the trained models to output the reported results.
- To retrain the networks, ensure to store old models elsewhere as the current code replaces the previous trained models, and figures.
- For retraining, uncomment the training part containing the block "history" and comment out the load model line.
- times-new-roman.ttf file is used for figures axis-label formating as "times new roman format".
