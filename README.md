# ML4641_Robotics_Project



Currently considered dataset:
https://www.kaggle.com/datasets/samehraouf/fault-detection-in-hexapod-robot-joints-dataset/data



# Methods

Data Preprocessing:

Based on the dataset provided, we will first find data points with pose or slope values drastically different from most other points in the dataset through RANSAC, which will iteratively select random values in the dataset and fit a model based on these random values. Inliers will then be determined based on a margin around the generated line, and outliers will be discarded. RANSAC will be run until the line with the largest number of inliers is found, and all outliers remaining will be discarded. We may use PyTorch’s torch-ransac3d for this purpose, which will assist in cleaning our data to ensure only important values are considered.

To fill in missing gaps in our data and ensure a smooth dataset, we may also use techniques such as linear regression or K-nearest neighbors based on the poses and slopes present in our dataset ensuring the data the model uses accounts for all possibilities that may occur when the robot is being run.

For effective data analysis, we may also have to normalize the pose data of the robot to assist in generating a baseline for the robot to work off of when it no longer can rely on our labeled, normalized, smoothed dataset given we will transition the robot to a less neat unsupervised dataset when we transition it to a true simulation and, subsequently, the real world.
