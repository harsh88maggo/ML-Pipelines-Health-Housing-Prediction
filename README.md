# Machine Learning Health and Housing Predictions

This project implements three machine learning algorithms using Apache Spark ML for various prediction tasks, including diabetes risk prediction, cancer classification, and housing price prediction. Each algorithm utilizes a different machine learning model, data preprocessing steps, and evaluation techniques.

## Project Overview

The project contains three distinct machine learning algorithms:

1. **Diabetes Prediction**: Predicts if a person is at risk for diabetes based on features like glucose levels, BMI, and age.
2. **Cancer Classification**: Classifies tumors as cancerous or not based on various features such as clump thickness, cell size, and shape.
3. **Housing Price Prediction**: Predicts housing prices based on features like square footage, number of bathrooms, etc.

Each algorithm is contained in its respective directory with the dataset, model code, and a report summarizing the approach, results, and evaluation.

## Setup Instructions

### Prerequisites

Before running the algorithms, ensure you have the following set up:

- **Apache Spark**: Set up a Spark environment (either standalone or on a cluster).
- **Hadoop**: Ensure Hadoop is configured and running for storing datasets.
- **Scala**: The code is written in Scala using SparkML.

Upload Datasets to Hadoop HDFS
For each algorithm (Diabetes, Cancer, Housing), you need to upload the corresponding dataset to your Hadoop file system using the following command:

```bash
hadoop fs -mkdir /<your_directory_name>
hadoop fs -copyFromLocal <dataset_file> /<your_directory_name>/
```
### Step 1: Data Preparation

The datasets for each algorithm are located in the corresponding directories:

- `Diabetes_Prediction/diabetes.csv`
- `Cancer_Classification/cancer.csv`
- `HousePrice_Prediction/train.csv`

### Step 2: Running the Algorithms

#### Diabetes Algorithm (`Diabetes_Prediction/code/diabetes_model.txt`)

- **Model**: RandomForestClassifier in SparkML to predict if a person is at risk for diabetes based on features like glucose levels, BMI, and age.
- **Steps**:
  - Data preprocessing and cleaning are done before splitting the dataset into training and testing sets.
  - Hyperparameter tuning is performed using CrossValidation with a grid of `maxDepth` and `numTrees` parameters.

- **Execution**:
  1. Set up your Spark environment.
  2. Run the Spark code inside `diabetes_model.txt`.

#### Cancer Algorithm (`Cancer_Classification/code/cancer_model.txt`)

- **Model**: LogisticRegression in SparkML to classify cancerous tumors based on features such as clump thickness, cell size, and shape.
- **Steps**:
  - The dataset is preprocessed.
  - Cross-validation is used for hyperparameter tuning.

- **Execution**:
  1. Set up your Spark environment.
  2. Run the Spark code inside `cancer_model.txt`.

#### Housing Algorithm (`HousePrice_Prediction/code/house_price_model.txt`)

- **Model**: LinearRegression in SparkML to predict housing prices based on features like square footage, number of bathrooms, etc.
- **Steps**:
  - The dataset is cleaned by handling missing values and then split into training and test sets.
  - Hyperparameters are tuned using a CrossValidator.

- **Execution**:
  1. Set up your Spark environment.
  2. Run the Spark code inside `housing_model.txt`.

### Step 3: Evaluation

After training each model, the evaluation results are printed to the console:

- For the **Diabetes** and **Cancer** models, accuracy is calculated using a `MulticlassClassificationEvaluator`.
- For the **Housing** model, an `R2` evaluation is performed.

### Step 4: Reports

Reports for each algorithm can be found in the respective directories in PDF format. These reports include:

- Screenshots of code execution.
- Preprocessing steps.
- Model evaluation metrics.

## Notes

- All code is implemented in **Scala** with **SparkML** for machine learning models.
- Make sure to replace the `hdfs://10.128.0.7/` path with the actual path to your HDFS or Spark setup.

## Future Enhancements

- Explore other machine learning models like **Support Vector Machines (SVM)** or **Gradient Boosting Machines (GBM)** for improved performance.
- Investigate **feature engineering** techniques to enhance prediction accuracy.

