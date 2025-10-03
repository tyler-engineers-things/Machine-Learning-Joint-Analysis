# ML4641_Robotics_Project

Currently considered dataset:
https://www.kaggle.com/datasets/samehraouf/fault-detection-in-hexapod-robot-joints-dataset/data

## Introduction

Robotic systems are deployed often operate in demanding environments where they must maintain high reliability and safety. In many contexts including manufacturing, search and rescue, medicine, and more, failure in a robot join or actuator/sensor can cause large disruptions or damage. The ability to detect, classify, and isolate faults in robotics subsystems in real time is critical to maintaining this standard. Fault detection and diagnosis (FDD) in robotic systems had been studied both from model-based and data-driven perspectives. In model-based approaches, observers, parity checks, and residual generation techniques are used to detect deviation from expected dynamics. In data-driven approaches, sensor and control signals are treated as signals to be classified or regressed using machine learning models. 


### Dataset

In this project we propose to build upon data-driven fault detection for robotic joins using the public “Falt Detection in Hexapod Robot Joints” dataset from Kaggle [3]. We aim to apply machine learning models to this dataset to find and detect robotic faults. This dataset provides timeseries readings from hexapod robot joints under various fault conditions including offset error, gain error, and combined faults. Features include position data of the joins and slopes with respect to different axis. The dataset is labeled indicating the fault class of the sample making this a supervised learning problem. We aim to augment this dataset with simulated dataset from Georgia Tech’s Robotarium where we will create similar conditions to the ones in the Hexapod Robot Joints and work to optimize and find enhancements in real world robotic systems [4].

### Background

Recent work in data-driven robotic optimization approaches include a Two-States Random Forest (TSRF) algorithm applied to hexapod robot join fault detection [1]. By layering two random forest classifiers and using class-probability vectors from the first stage as meta-features for the second, they claim 99.7% accuracy over baseline random forest models. Another approach appled a backpropagation neural network approach to robotic fault diagnosis, exploring how different error magnitudes and sampling frequencies influence accuracy [2]. They report a diagnostic accuracy of up to 99.17% in their simulation. 


## Problem Definition

From these works, several recurring themes emerge. First, faults often include constant offset, gain error, stuck or drift behavior in sensors or actuators, and sometimes abrupt failures. Second, real-world systems have measurement noise, so robustness to noise or disturbance is critical. Third, distinguishing which joint or subsystem is faulty (sensor vs. actuator) is more challenging than simply detecting a fault. These challenges can be solved by using data-driven machine learning approaches which we aim to do with the hexapod dataset. 

Given sensor and actuator readings from the joins of a hexapod robot, the goal is to detect and classify faults. We plan to benchmark several models on this task, compare their performance, and analyze feature importance, misclassifications, and robustness to noise. 

This problem is worth tackling due to its practical relevance, extensions to other research fields, and potential for novelty. Fault detection in robotic joins is crucial for maintaining uptime, safety, and reliability. Even subtle faults (sensor drift or gain error) can degrade performance gradually and lead to wear or failure. The hexapod is a suitable test bed with multiple joints. Analyzing this dataset might show that certain features and methods yield better diagnostic accuracy over previous methods providing a novel insight into fault detection that can be generalized to other systems as well. 


## Methods

### Data Preprocessing:

Based on the dataset provided, we will first find data points with pose or slope values drastically different from most other points in the dataset through RANSAC, which will iteratively select random values in the dataset and fit a model based on these random values. Inliers will then be determined based on a margin around the generated line, and outliers will be discarded. RANSAC will be run until the line with the largest number of inliers is found, and all outliers remaining will be discarded. We may use PyTorch’s torch-ransac3d for this purpose, which will assist in cleaning our data to ensure only important values are considered.

To fill in missing gaps in our data and ensure a smooth dataset, we may also use techniques such as linear regression or K-nearest neighbors based on the poses and slopes present in our dataset ensuring the data the model uses accounts for all possibilities that may occur when the robot is being run.

For effective data analysis, we may also have to normalize the pose data of the robot to assist in generating a baseline for the robot to work off of when it no longer can rely on our labeled, normalized, smoothed dataset given we will transition the robot to a less neat unsupervised dataset when we transition it to a true simulation and, subsequently, the real world.

### ML Algorithms:

We will be using a supervised model as the dataset we are analyzing contains data labeled with respect to the number of joint faults that have occurred in the pose transformation. To effectively analyze the pose and slope data provided, we will use logistic regression to determine which feature combinations result in faulty joints and which ones are more conducive to the current design of the robot. Logistic regression will likely be done through PyTorch’s torch.nn.Module.

As logistic regression is a supervised algorithm, when we transition the model to the simulation we may not be able to use it as effectively as the data generated here will likely be unlabeled. Therefore, when transitioning past the original dataset we may begin using K-means or other similar unsupervised learning algorithms to ensure the model continues to function correctly based on new pose and slope estimates. Our simulations will likely be run through MuJoCo, and K-means will likely be run using PyTorch’s torch_kmeans.

The algorithms we choose for simulations should be quicker and lower time complexity due to the constraints often present in the field of robotics.

## Results:

## Quantitative Metrics:

To quantify the effectiveness of our utilized ML techniques, we may want to consider using various loss and scoring functions for the various parts of our project. When working with logistic regression, for example, we may want to use Log Loss as our loss function as it is designed for quantifying logistic regression algorithms, and may also want to consider Area Under the ROC Curve (AUC) as another valuable loss function. When working with Linear Regression for various purposes throughout our project, we may also want to consider the Mean Squared Error (MSE) to quantify how far from the average value our Linear Regression result was. One last metric to consider for both of our regressions is the Root MSE, which is another popular metric for quantifying the quality of linear and logarithmic regressions. When working with our unsupervised algorithm, K-means, it may be valuable to also consider utilizing Silhouette Analysis to determine whether points generally fall within or close to within their respective clusters.

### Project Goals and Expected Results:

With this project, we are hoping to find methods for which joint failures can be minimized or eliminated all-together to ensure robots are gentler and more capable of working with humans and other fragile objects. With the proliferation of robots in every facet of our lives, it is crucial that projects be put forth that ensure these robots remain safe to use by the average layperson, and failing to do so may result in unintended and unnecessary injuries and similar consequences. Our project’s intent is to promote the sustainability and ethical use of robotic technologies in various consumer-facing economic spheres, and if successful may contribute greatly to ensuring ML techniques result in robotics-related injuries reducing to a minimum.

Based on our research of ML algorithms, techniques, and metrics, we expect our project will result in a successful reduction in the proliferation and overexertion of faulty joints, ensuring greater robotic stability which will contribute to greater robotic safety in all facets of our economic life.


## References

[1]Q. Fang, Y. Men, K. Zhang, M. Yu, and Y. Liu, “Real-Time Joint Fault Detection and Diagnosis of Hexapod Robot Based on Improved Random Forest,” Processes, vol. 13, no. 9, pp. 2762–2762, Aug. 2025, doi: https://doi.org/10.3390/pr13092762.

[2]M. Hu, J. Wu, J. Yang, L. Zhang, and F. Yang, “Fault diagnosis of robot joint based on BP neural network,” Robotica, vol. 40, no. 12, pp. 4388–4404, Jul. 2022, doi: https://doi.org/10.1017/s0263574722000984.

[3]S. Raouf, “Fault Detection in Hexapod Robot Joints(One Joint),” Kaggle.com, 2024. https://www.kaggle.com/datasets/samehraouf/fault-detection-in-hexapod-robot-joints-dataset/data (accessed Oct. 03, 2025).

[4]“Robotarium,” Gatech.edu, 2025. https://www.robotarium.gatech.edu/ (accessed Oct. 03, 2025).
