# Overall
This is a multi-classification question. We use four methods(SVM, naive bayes, Random Forest and XGBoost) in R language to solve a simple dataset.
## 1.Dataset
Source: https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre/data

Licence: https://creativecommons.org/publicdomain/zero/1.0/
## 2.Data pre-processing(Processor.R and DataPreprocessor.R)
### Process.R:
Handling Missing Values: It can replace missing values in the dataset and remove rows with missing values.

Removing Duplicates: It can eliminate duplicate rows from the dataset.

Data Type Conversion: It can attempt to convert character columns to numeric format.

Value Replacement: It can replace specific values in a chosen column with values from a provided dictionary.

Handling High Cardinality Categoricals: It can remove non-numeric columns with a high number of unique values.

Encoding Categorical Data: It can encode categorical columns into integer values, suitable for machine learning.

Label Encoding: It can encode a specified label column, often used in classification tasks.

Feature Scaling: It standardizes numeric columns by centering them at zero and scaling to a standard deviation of 1.

Column Dropping: It can drop specified columns from the dataset.

Renaming Label Column: It renames the label column to 'label' for consistency.

Feature Selection: It selects features based on their correlation with the label column.
### DataPreprocessor.R：
Data Splitting: The split_data method splits a dataset into training, validation, and testing sets. It uses stratified sampling to ensure class balance.

Principal Component Analysis (PCA): The apply_pca method performs PCA on the data to reduce its dimensionality while retaining a specified level of explained variance.

Undersampling: The undersample method balances the class distribution in the dataset by randomly undersampling the majority class.

K-Nearest Neighbors (KNN) Model: The train_initial_model method trains a KNN classifier on the undersampled training data and predicts labels for the validation set.

Similarity Matrix Calculation: The compute_similarity_matrix method computes a similarity matrix based on the confusion matrix of predicted versus actual labels.

Visualization: The visualize_similarity_matrix method creates a heatmap to visualize the similarity matrix.

Label Merging: The merge_classes_based_on_similarity method merges classes based on their similarity in prediction behavior.

Label Reencoding: The reencode_labels method reencodes labels as numeric values using factors.

Saving Datasets: The save_datasets_to_csv method saves the training and testing datasets to CSV files.

Label Counts: The label_counts method provides a count of unique labels in the dataset.

Overall Data Preprocessing: The perform_initial_steps method orchestrates a series of preprocessing steps, including data splitting, undersampling, model training, similarity analysis, label merging, reencoding, and optional PCA. It returns the final preprocessed datasets and related information.

**Algorithms and Techniques in DataPreprocessor.R:**

PCA (Principal Component Analysis): Used for dimensionality reduction.

K-Nearest Neighbors (KNN): Utilized as an initial classifier.

Confusion Matrix and Similarity Matrix: Employed to analyze the similarity between class predictions.

Stratified Sampling: Ensures balanced splits in the data.
## 3.Methods
### 3.1 SVM
