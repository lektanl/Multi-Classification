# Overall
This is a multi-classification question. We use four methods(SVM, naive bayes, Random Forest and XGBoost) in R language to solve a simple dataset.
## 1.Dataset
Source: https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre/data

Licence: https://creativecommons.org/publicdomain/zero/1.0/
## 2.Data pre-processing(Processor.R and DataPreprocessor.R)
### Process.R:

1. **initialize**: Initializes the object, setting the data frame (`df`) as a class property.

2. **drop_single_unique_value_columns**: Removes columns from the data frame that have only one unique value, as they do not provide informative variance for analysis or modeling.

3. **replace_missing_values**: Replaces specified representations of missing values in the data frame with `NA`, and then drops rows with `NA` values. 

4. **drop_missing_values**: Removes rows from the data frame that contain any `NA` values.

5. **remove_duplicates**: Eliminates duplicate rows from the data frame.

6. **convert_strings_to_numeric**: Attempts to convert character columns to numeric where possible. If a column can't be converted, it's left as is.

7. **drop_high_dim_non_numericals**: Removes character columns from the data frame if the number of unique non-NA values in a column exceeds a specified threshold ratio of the number of rows.

8. **encode_categoricals**: Converts character columns (except specified exceptions) to numeric by transforming them into factors and then converting to integer codes.

9. **encode_labels**: Specifically converts a specified label column to numeric using factor encoding.

10. **scale_features**: Scales numeric columns in the data frame, excluding specified columns.

11. **drop_columns**: Removes specified columns from the data frame.

12. **rename_label_column**: Renames a specified label column to 'label'.

13. **select_features_based_on_correlation**: Keeps features in the data frame that have a correlation with the label column above a certain threshold.

14. **get_processed_dataframe**: Returns the processed data frame.

15. **replace_values_in_column**: Replaces specific values in a specified column based on a provided map.

16. **process_dataframe**: A comprehensive function that applies several preprocessing steps in sequence, including missing value replacement, value replacement in a key column, high dimensionality reduction, string to numeric conversion, categorical encoding, column dropping, feature selection based on correlation, label column renaming, label encoding, and feature scaling.

Each function is designed to modify the data frame stored in the class, providing a fluent interface that allows chaining multiple preprocessing steps together. The `process_dataframe` function serves as a higher-level interface to apply a series of preprocessing steps in a standardized way.

### DataPreprocessor.R：
summary for each function:

1. **initialize**: 
   - Initializes the class instance.
   
2. **split_data**: 
   - Splits data into train, test, and optionally validation sets using stratified sampling.

3. **apply_pca**: 
   - Reduces dataset dimensionality by applying PCA, using the training data to determine components.

4. **undersample**: 
   - Balances classes by limiting each to the size of the smallest class.

5. **train_initial_model**: 
   - Trains a k-NN model on undersampled data and predicts on a validation set.

6. **compute_similarity_matrix**: 
   - Generates a matrix indicating label similarity based on prediction overlaps.

7. **visualize_similarity_matrix**: 
   - Displays a heatmap showing how similar different labels are based on their predictions.

8. **merge_classes_based_on_similarity**: 
   - Combines labels with similarity beyond a given threshold to address overlaps.

9. **reencode_labels**: 
   - Assigns new numeric codes to labels for consistency after potential merges.

10. **save_datasets_to_csv**: 
   - Exports the processed datasets as CSV files.

11. **label_counts**: 
   - Provides a count of instances for each label in the dataset.

12. **perform_initial_steps**: 
   - Executes a series of preprocessing steps, including data splitting, undersampling, initial modeling, label merging, and PCA.

This class structures and simplifies the data preprocessing workflow, especially for imbalanced classification tasks.

**Algorithms and Techniques in DataPreprocessor.R:**

PCA (Principal Component Analysis): Used for dimensionality reduction.

K-Nearest Neighbors (KNN): Utilized as an initial classifier.

Confusion Matrix and Similarity Matrix: Employed to analyze the similarity between class predictions.

Stratified Sampling: Ensures balanced splits in the data.
## 3.Methods
### 3.1 SVM
