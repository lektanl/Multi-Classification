library(R6)
library(caret)
library(e1071)
library(keras)
library(FNN)
library(dplyr)
library(xgboost)
library(class)
library(ggplot2)

#DataPreprocessor:
# - initialize(self): Initializes the DataPreprocessor object. No specific parameters or flags set at initialization.
# 
# - split_data(self, df, num_splits = 3): Splits the input dataframe into training, validation (if `num_splits` is set to 3), and test datasets.
# 
# - apply_pca(self, X_train, X_test, X_valid = NULL, n_components = 0.95): Applies Principal Component Analysis (PCA) on the dataset, reducing its dimensionality, conditioned on a specified variance retention threshold.
# 
# - undersample(self, X, y): Conducts undersampling to balance class distribution by ensuring each class in the dataset has equal representation.
# 
# - train_initial_model(self, X_train, y_train, X_valid, y_valid, k): Utilizes the k-nearest neighbors (kNN) algorithm to train an initial model with the undersampled training data and predicts on the validation data.
# 
# - compute_similarity_matrix(self, y_pred, y_true): Calculates a similarity matrix based on the predictions of the initial model, comparing it with actual validation labels.
# 
# - visualize_similarity_matrix(self, similarity_mtx): Renders the similarity matrix in the form of a heatmap for visual interpretation.
# 
# - merge_classes_based_on_similarity(self, df, similarity_mtx, similarity_threshold): Merges certain classes within the original dataframe, based on their inferred similarities from the similarity matrix.
# 
# - reencode_labels(self, df): Re-encodes class labels after potential class merging, ensuring labels are contiguous integers.
# 
# - save_datasets_to_csv(self, train_set, test_set): Exports processed training and test datasets to specified CSV files.
# 
# - label_counts(self, df): Counts and returns the number of occurrences for each unique label in the dataframe.
# 
# - perform_initial_steps(self, df_scaled, use_pca = FALSE, n_components = 0.95, k = 25, similarity_threshold = 0.65): Orchestrates and executes the preprocessing steps in sequence, from data splitting to PCA application, class merging, and final data preparation.

DataPreprocessor <- R6Class("DataPreprocessor",
                            public = list(
                              
                              initialize = function() {
                              },
                              
                              split_data = function(df, num_splits = 3) {
                                set.seed(0)
                                trainIndex <- createDataPartition(df$label, p = .8, list = FALSE, times = 1)
                                
                                X_train_full <- df[trainIndex, -ncol(df)]
                                y_train_full <- df$label[trainIndex]
                                X_test <- df[-trainIndex, -ncol(df)]
                                y_test <- df$label[-trainIndex]
                                
                                if (num_splits == 2) {
                                  return(list(X_train = X_train_full, X_test = X_test, y_train = y_train_full, y_test = y_test))
                                } else if (num_splits == 3) {
                                  trainIndex <- createDataPartition(y_train_full, p = .9, list = FALSE, times = 1)
                                  
                                  X_train <- X_train_full[trainIndex,]
                                  y_train <- y_train_full[trainIndex]
                                  X_valid <- X_train_full[-trainIndex,]
                                  y_valid <- y_train_full[-trainIndex]
                                  
                                  return(list(X_train = X_train, X_valid = X_valid, X_test = X_test, y_train = y_train, y_valid = y_valid, y_test = y_test))
                                } else {
                                  stop("Invalid number of splits. Choose either 2 or 3.")
                                }
                              },
                              
                              apply_pca = function(X_train, X_test, X_valid = NULL, n_components = 0.95) {
                                pca_model <- prcomp(X_train, center = TRUE, scale. = TRUE)
                                
                                X_train_pca <- as.data.frame(predict(pca_model, X_train))
                                X_test_pca <- as.data.frame(predict(pca_model, X_test))
                                
                                if (!is.null(X_valid)) {
                                  X_valid_pca <- as.data.frame(predict(pca_model, X_valid))
                                  return(list(X_train = X_train_pca, X_valid = X_valid_pca, X_test = X_test_pca))
                                } else {
                                  return(list(X_train = X_train_pca, X_test = X_test_pca))
                                }
                              },
                              
                              undersample = function(X, y) {
                                df <- data.frame(X, label = y)
                                min_class_size <- min(table(df$label))
                                if(min_class_size < 1) stop("One or more classes have insufficient samples for undersampling.")
                                
                                df_balanced <- df %>%
                                  group_by(label) %>%
                                  slice_sample(n = min_class_size) %>%
                                  ungroup()
                                
                                list(X = as.matrix(df_balanced[,-ncol(df_balanced)]), y = df_balanced$label)
                              },
                              
                              train_initial_model = function(X_train, y_train, X_valid, y_valid, k) {
                                y_pred <- knn(train = X_train, test = X_valid, cl = y_train, k = k)
                                return(y_pred)
                              },
                              
                              compute_similarity_matrix = function(y_pred, y_true) {
                                confusion_mtx <- as.matrix(table(Predicted = y_pred, Actual = y_true))
                                row_sums <- rowSums(confusion_mtx)
                                
                                normalized_confusion_mtx <- confusion_mtx / matrix(row_sums, nrow=nrow(confusion_mtx), ncol=ncol(confusion_mtx), byrow = TRUE)
                                
                                n <- nrow(normalized_confusion_mtx)
                                m <- ncol(normalized_confusion_mtx)
                                
                                overlap_mtx <- matrix(0, nrow = n, ncol = m)
                                
                                for (i in seq_len(n)) {
                                  for (j in seq_len(m)) {
                                    if (i == j) {
                                      overlap_mtx[i, j] <- 1
                                    } else {
                                      overlap_mtx[i, j] <- (normalized_confusion_mtx[i, j] + normalized_confusion_mtx[j, i]) / 2
                                      overlap_mtx[j, i] <- overlap_mtx[i, j]
                                    }
                                  }
                                }
                                
                                similarity_mtx <- 1 - overlap_mtx
                                similarity_mtx
                              },
                              
                              visualize_similarity_matrix = function(similarity_mtx) {
                                similarity_df <- as.data.frame(as.table(similarity_mtx))
                                colnames(similarity_df) <- c("Var1", "Var2", "Freq")
                                
                                ggplot(similarity_df, aes(x = Var1, y = Var2, fill = Freq)) + 
                                  geom_tile() + 
                                  geom_text(aes(label = sprintf("%.2f", Freq)), vjust = 1) + 
                                  scale_fill_gradient(low = "white", high = "blue") + 
                                  theme_minimal() +
                                  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
                                  labs(x = "Label", y = "Label", fill = "Similarity") +
                                  ggtitle("Similarity Matrix Heatmap")
                              },
                              
                              merge_classes_based_on_similarity = function(df, similarity_mtx, similarity_threshold) {
                                labels_to_merge <- which(similarity_mtx < similarity_threshold, arr.ind = TRUE)
                                already_merged <- integer(0)
                                label_mapping <- list()
                                
                                for(i in seq_along(labels_to_merge[,1])) {
                                  label1 <- labels_to_merge[i, 1]
                                  label2 <- labels_to_merge[i, 2]
                                  
                                  if(!(label2 %in% already_merged)) {
                                    df$label[df$label == label2] <- label1
                                    label_mapping[[as.character(label2)]] <- label1
                                    already_merged <- c(already_merged, label2)
                                  }
                                }
                                list(df = df, label_mapping = label_mapping)
                              },
                              
                              reencode_labels = function(df) {
                                df$label <- as.numeric(as.factor(df$label))
                                df
                              },
                              
                              save_datasets_to_csv = function(train_set, test_set) {
                                write.csv(train_set, "train_set.csv", row.names = FALSE)
                                write.csv(test_set, "test_set.csv", row.names = FALSE)
                              },
                              
                              label_counts = function(df) {
                                if (!"label" %in% colnames(df)) stop("The dataframe must contain a 'label' column.")
                                table(df$label)
                              },
                              
                              perform_initial_steps = function(df_scaled, use_pca = FALSE, n_components = 0.95, k = 25, similarity_threshold = 0.65) {
                                
                                # 1. Split the data into train, validation, and test sets
                                split_data_result <- self$split_data(df_scaled, num_splits = 3)
                                
                                # 2. Undersample the training set
                                undersampled_data <- self$undersample(split_data_result$X_train, split_data_result$y_train)
                                
                                # 3. Train the initial model
                                y_pred <- self$train_initial_model(undersampled_data$X, undersampled_data$y, split_data_result$X_valid, split_data_result$y_valid, k)
                                
                                # 4. Compute the similarity matrix
                                similarity_mtx <- self$compute_similarity_matrix(y_pred, split_data_result$y_valid)
                                
                                # 5. Merge classes based on the similarity matrix
                                merge_result <- self$merge_classes_based_on_similarity(df_scaled, similarity_mtx, similarity_threshold)
                                df_re_class <- merge_result$df
                                
                                # 6. Reencode labels
                                df_re_class <- self$reencode_labels(df_re_class)
                                
                                # 7. Split the data again for final undersampling
                                split_data_final_result <- self$split_data(df_re_class, num_splits = 2)
                                
                                # 8. Undersample the training set for the final time
                                undersampled_final_data <- self$undersample(split_data_final_result$X_train, split_data_final_result$y_train)
                                split_data_final_result$X_train <- undersampled_final_data$X
                                split_data_final_result$y_train <- undersampled_final_data$y
                                
                                # 9. Apply PCA if necessary
                                if (use_pca) {
                                  pca_result <- self$apply_pca(split_data_final_result$X_train, split_data_final_result$X_test)
                                  split_data_final_result$X_train <- pca_result$X_train
                                  split_data_final_result$X_test <- pca_result$X_test
                                }
                                
                                # 10. Merge features and labels into training and testing data frames
                                train_set <- data.frame(split_data_final_result$X_train, label = split_data_final_result$y_train)
                                test_set <- data.frame(split_data_final_result$X_test, label = split_data_final_result$y_test)
                                
                                return(list(df_re_class = df_re_class, train_set = train_set, test_set = test_set, similarity_mtx = similarity_mtx, label_changes = merge_result$label_mapping))
                              }
                              
                              
                              
                              
                            )
)