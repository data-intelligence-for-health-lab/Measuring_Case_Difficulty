# Measuring Case Difficulty Project

This project introduces novel case difficulty calculation metrics designed to perform well across various datasets. The metrics were developed using neural networks and tailored to different definitions of prediction difficulty.

## Case Difficulty Metrics

- Case Difficulty Model Complexity **(CDmc)**
  - CDmc is based on the complexity of the neural network required for accurate predictions.

- Case Difficulty Double Model **(CDdm)**
  - CDdm utilizes a pair of neural networks: one predicts a given case, and the other assesses the likelihood that the prediction made by the first model is correct.

- Case Difficulty Predictive Uncertainty **(CDpu)**
  - CDpu evaluates the variability of the neural network's predictions.

**Note:**
CDmc, CDdm, and CDpu were originally named Approach 1, Approach 2, and Approach 3, respectively. Some code results may still refer to the previous names.

## Dataset Information
- **Simulated Datasets:**
  Results include the data. 

- **Real-World Datasets:**
  Only calculated values are included. The original datasets can be found at the following addresses:
  - [UCI Breast Cancer Data](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)
  - [Telco Data](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
  - [Customer Data](https://www.kaggle.com/datasets/vetrirah/customer)

  You can merge these results with the result data since the index orders are the same.

