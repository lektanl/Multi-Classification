Processor <- R6Class("Processor",
                     
                     public = list(
                       df = NULL,
                       
                       initialize = function(df) {
                         self$df <- df
                       },
                       
                       drop_single_unique_value_columns = function() {
                         unique_counts <- sapply(self$df, function(col) length(unique(na.omit(col))))
                         self$df <- self$df[, unique_counts != 1]
                         return(self)
                       },
                       
                       replace_missing_values = function(missing_value_representations = c("?", "NaN")) {
                         self$df <- self$df %>%
                           mutate_all(~ replace(., . %in% missing_value_representations, NA))
                         self$drop_missing_values() # 删除替换后的NA值
                         return(self)
                       },
                       
                       drop_missing_values = function() {
                         self$df <- na.omit(self$df)
                         return(self)
                       },
                       
                       remove_duplicates = function() {
                         self$df <- unique(self$df)
                         return(self)
                       },
                       
                       convert_strings_to_numeric = function() {
                         try_convert_numeric <- function(column) {
                           result <- suppressWarnings(as.numeric(column))
                           if (anyNA(result)) return(column)
                           return(result)
                         }
                         
                         self$df <- self$df %>% mutate(across(where(is.character), ~ try_convert_numeric(.)))
                         return(self)
                       },
                       
                       drop_high_dim_non_numericals = function(threshold_ratio) {
                         for (col in names(self$df)) {
                           if (is.character(self$df[[col]])) {
                             unique_count <- length(unique(na.omit(self$df[[col]])))
                             if (unique_count > threshold_ratio * nrow(self$df)) {
                               self$df[[col]] <- NULL
                             }
                           }
                         }
                         return(self)
                       },
                       
                       encode_categoricals = function(exceptions = c()) {
                         for (col in names(self$df)) {
                           if (is.character(self$df[[col]]) && !(col %in% exceptions)) {
                             self$df[[col]] <- as.integer(as.factor(self$df[[col]]))
                           }
                         }
                         return(self)
                       },
                       
                       encode_labels = function(label_column = "label") {
                         if (label_column %in% names(self$df)) {
                           self$df[[label_column]] <- as.integer(as.factor(self$df[[label_column]]))
                         }
                         return(self)
                       },
                       
                       scale_features = function(exclude_columns = c()) {
                         columns_to_scale <- setdiff(names(self$df), exclude_columns)
                         
                         non_numeric_cols <- names(self$df)[sapply(self$df, function(col) !is.numeric(col))]
                         non_numeric_cols <- setdiff(non_numeric_cols, exclude_columns)
                         
                         if(length(non_numeric_cols) > 0) {
                           # Do nothing or handle non-numeric columns if necessary
                         }
                         
                         self$df[columns_to_scale] <- lapply(self$df[columns_to_scale], scale)
                         return(self)
                       },
                       
                       drop_columns = function(columns_to_drop) {
                         self$df[columns_to_drop] <- NULL
                         return(self)
                       },
                       
                       rename_label_column = function(label_column) {
                         if (label_column %in% names(self$df)) {
                           colnames(self$df)[which(names(self$df) == label_column)] <- 'label'
                         }
                         return(self)
                       },
                       
                       select_features_based_on_correlation = function(label_column='label', threshold=0.1) {
                         numeric_cols = which(sapply(self$df, is.numeric))
                         if (label_column %in% colnames(self$df[numeric_cols])) {
                           corr_matrix = cor(self$df[numeric_cols], use="complete.obs")
                           if (label_column %in% rownames(corr_matrix)) {
                             labels_corr = corr_matrix[label_column, ]
                             discard_cols = names(labels_corr[abs(labels_corr) < threshold & !is.na(labels_corr)])
                             self$df = self$df %>% select(-all_of(discard_cols))
                           }
                         }
                         return(self)
                       },
                       
                       get_processed_dataframe = function() {
                         return(self$df)
                       },
                       
                       replace_values_in_column = function(column_name, replacements) {
                         self$df[column_name] <- ifelse(!is.na(self$df[[column_name]]) & (self$df[[column_name]] %in% names(replacements)), 
                                                        replacements[self$df[[column_name]]], 
                                                        self$df[[column_name]])
                         return(self)
                       },
                       
                       process_dataframe = function(key_column, label_column, frequency_map, columns_to_drop, threshold_ratio = 0.1) {
                         self$replace_missing_values()
                         self$replace_values_in_column(key_column, frequency_map)
                         self$drop_high_dim_non_numericals(threshold_ratio)
                         self$convert_strings_to_numeric()
                         self$encode_categoricals(c(label_column))
                         self$drop_columns(columns_to_drop)
                         self$select_features_based_on_correlation(label_column, 0.9)
                         self$rename_label_column(label_column)
                         self$encode_labels('label')
                         self$scale_features(c('label', key_column))
                         return(self$get_processed_dataframe())
                       }
                     )
)
