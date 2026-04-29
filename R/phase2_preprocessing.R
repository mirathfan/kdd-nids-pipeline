# =============================================================================
# KDD Cup 1999 - Network Intrusion Detection
# Phase 2: Preprocessing, Feature Selection & SMOTE
# Primary language: R
# Input : data/kdd_labelled.csv       (from Phase 1)
# Outputs: data/train_balanced.csv    (SMOTE-balanced training set)
#          data/test.csv              (held-out test set, no SMOTE)
#          data/feature_names.rds     (selected feature list for Python)
#          outputs/p6_feature_importance.png
#          outputs/p7_smote_class_dist.png
# =============================================================================

# --- 0. Packages -------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,    # data wrangling
  caret,        # createDataPartition, nearZeroVar, preProcess
  randomForest, # feature importance
  smotefamily,  # SMOTE implementation
  scales,       # plot formatting
  knitr         # summary tables
)

dir.create("outputs", showWarnings = FALSE)

# --- 1. Load Phase 1 output --------------------------------------------------
message("Loading data/kdd_labelled.csv ...")
df <- read_csv("data/kdd_labelled.csv",
               col_types = cols(label = col_character(),
                                .default = col_double(),
                                protocol_type = col_character(),
                                service = col_character(),
                                flag = col_character()),
               show_col_types = FALSE) %>%
  mutate(label = factor(label,
                        levels = c("Normal","DoS","Probe","R2L","U2R")))

message(sprintf("Loaded: %s rows x %s cols", 
                format(nrow(df), big.mark=","), ncol(df)))

# --- 2. One-hot encode categorical features ----------------------------------
# 3 categorical columns: protocol_type (3 levels), service (66), flag (11)
message("\nOne-hot encoding categorical features...")

cat_cols <- c("protocol_type", "service", "flag")

# Build dummy matrix (drop first level to avoid perfect multicollinearity)
dummy_model <- dummyVars(~ protocol_type + service + flag,
                         data = df, fullRank = TRUE)
cat_encoded  <- predict(dummy_model, newdata = df) %>% as_tibble()

# Drop original categoricals, bind encoded columns
df_encoded <- df %>%
  select(-all_of(cat_cols)) %>%
  bind_cols(cat_encoded)

message(sprintf("After encoding: %s cols (was %s)",
                ncol(df_encoded), ncol(df)))

# --- 3. Remove near-zero variance features -----------------------------------
message("\nRemoving near-zero variance features...")

feature_cols <- setdiff(names(df_encoded), "label")

nzv_idx  <- nearZeroVar(df_encoded %>% select(all_of(feature_cols)),
                        saveMetrics = FALSE)
nzv_names <- feature_cols[nzv_idx]

message(sprintf("Removed %d near-zero variance features: %s",
                length(nzv_names),
                paste(head(nzv_names, 5), collapse = ", "),
                if (length(nzv_names) > 5) "..." else ""))

df_filtered <- df_encoded %>% select(-all_of(nzv_names))
feature_cols <- setdiff(names(df_filtered), "label")
message(sprintf("Features remaining: %d", length(feature_cols)))

# --- 4. Train / test split (stratified 80/20) --------------------------------
message("\nSplitting data 80/20 stratified by class...")
set.seed(42)

train_idx <- createDataPartition(df_filtered$label, p = 0.80, list = FALSE)
train_raw <- df_filtered[ train_idx, ]
test      <- df_filtered[-train_idx, ]

message(sprintf("Train: %s rows | Test: %s rows",
                format(nrow(train_raw), big.mark=","),
                format(nrow(test),      big.mark=",")))

# --- 5. Min-max scale numeric features ---------------------------------------
# Fit scaler on TRAIN only, apply to both train and test (no data leakage)
message("\nApplying min-max scaling (fit on train only)...")

numeric_features <- feature_cols   # all remaining cols are numeric after OHE

preproc <- preProcess(train_raw %>% select(all_of(numeric_features)),
                      method = c("range"))   # range = min-max [0,1]

train_scaled <- predict(preproc, train_raw)
test_scaled  <- predict(preproc, test)

message("Scaling complete. All features now in [0, 1].")

# --- 6. Feature importance via Random Forest (on 20% sample for speed) ------
message("\nRunning Random Forest for feature importance (sampled 20% of train)...")
set.seed(42)

sample_idx <- sample(nrow(train_scaled), size = floor(0.20 * nrow(train_scaled)))
rf_sample  <- train_scaled[sample_idx, ]

rf_model <- randomForest(
  label ~ .,
  data       = rf_sample,
  ntree      = 100,
  importance = TRUE,
  do.trace   = FALSE
)

# Extract and rank mean decrease in accuracy
importance_df <- importance(rf_model, type = 1) %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  rename(importance = MeanDecreaseAccuracy) %>%
  arrange(desc(importance))

message("\nTop 20 features by importance:")
print(head(importance_df, 20))

# Keep top 20 features for modelling
top_features <- importance_df %>%
  slice_head(n = 20) %>%
  pull(feature)

message(sprintf("\nSelected top %d features for modelling.", length(top_features)))

# --- 7. Plot: Feature importance ---------------------------------------------
p6 <- importance_df %>%
  slice_head(n = 20) %>%
  ggplot(aes(x = reorder(feature, importance), y = importance, fill = importance)) +
  geom_col(show.legend = FALSE, width = 0.7) +
  coord_flip() +
  scale_fill_gradient(low = "#B5D4F4", high = "#0C447C") +
  labs(
    title    = "Top 20 features — Random Forest importance",
    subtitle = "Mean decrease in accuracy (Phase 2 feature selection)",
    x        = NULL,
    y        = "Mean decrease in accuracy"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray50"),
    panel.grid.major.y = element_blank()
  )

ggsave("outputs/p6_feature_importance.png", p6, width = 9, height = 7, dpi = 150)
message("Saved: outputs/p6_feature_importance.png")

# --- 8. Subset to top features -----------------------------------------------
train_selected <- train_scaled %>% select(all_of(top_features), label)
test_selected  <- test_scaled  %>% select(all_of(top_features), label)

# --- 9. SMOTE on minority classes --------------------------------------------
# SMOTE works on numeric matrix — we apply it to Probe, R2L, U2R separately
# Strategy: oversample each minority class to reach ~5,000 samples minimum
message("\nApplying SMOTE to minority classes (Probe, R2L, U2R)...")

apply_smote <- function(data, target_class, target_n, k = 5) {
  # Separate target class vs rest
  minority <- data %>% filter(label == target_class)
  majority <- data %>% filter(label != target_class)
  
  current_n <- nrow(minority)
  if (current_n >= target_n) {
    message(sprintf("  %s already has %d samples, skipping.", target_class, current_n))
    return(minority)
  }
  
  dup_size <- ceiling(target_n / current_n) - 1
  message(sprintf("  %s: %d -> target %d (dup_size = %d)",
                  target_class, current_n, target_n, dup_size))
  
  # SMOTE requires numeric X and factor y
  X <- minority %>% select(-label) %>% as.data.frame()
  y <- minority$label
  
  # Adjust k if minority class is very small
  k_use <- min(k, nrow(minority) - 1)
  
  smote_result <- SMOTE(X, as.numeric(y), K = k_use, dup_size = dup_size)
  
  synthetic <- smote_result$data %>%
    as_tibble() %>%
    select(-class) %>%
    mutate(label = factor(target_class,
                          levels = c("Normal","DoS","Probe","R2L","U2R")))
  
  # Return only the synthetic rows (original minority already in train)
  synthetic_only <- synthetic %>% slice((current_n + 1):n())
  return(synthetic_only)
}

set.seed(42)

# Target: bring Probe to 8000, R2L to 5000, U2R to 2000
synthetic_probe <- apply_smote(train_selected, "Probe", target_n = 8000)
synthetic_r2l   <- apply_smote(train_selected, "R2L",   target_n = 5000)
synthetic_u2r   <- apply_smote(train_selected, "U2R",   target_n = 2000)

# Combine original train with synthetic samples
train_balanced <- bind_rows(
  train_selected,
  synthetic_probe,
  synthetic_r2l,
  synthetic_u2r
) %>%
  # Shuffle rows so classes are not ordered in blocks
  slice_sample(prop = 1, replace = FALSE)

message("\nClass distribution after SMOTE:")
class_after <- train_balanced %>%
  count(label) %>%
  mutate(pct = sprintf("%.2f%%", n / sum(n) * 100))
print(class_after)

# --- 10. Plot: Before vs After SMOTE ----------------------------------------
before <- train_selected %>%
  count(label) %>%
  mutate(stage = "Before SMOTE")

after <- train_balanced %>%
  count(label) %>%
  mutate(stage = "After SMOTE")

p7 <- bind_rows(before, after) %>%
  mutate(stage = factor(stage, levels = c("Before SMOTE", "After SMOTE"))) %>%
  ggplot(aes(x = label, y = n, fill = stage)) +
  geom_col(position = "dodge", width = 0.65) +
  scale_y_log10(labels = label_comma()) +
  scale_fill_manual(values = c("Before SMOTE" = "#B5D4F4",
                               "After SMOTE"  = "#0C447C")) +
  labs(
    title    = "Class distribution before vs after SMOTE",
    subtitle = "Log scale — minority classes (Probe, R2L, U2R) boosted",
    x        = "Attack class", y = "Count (log scale)", fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray50"),
    panel.grid.major.x = element_blank(),
    legend.position = "top"
  )

ggsave("outputs/p7_smote_class_dist.png", p7, width = 8, height = 5, dpi = 150)
message("Saved: outputs/p7_smote_class_dist.png")

# --- 11. Save outputs --------------------------------------------------------
message("\nSaving final datasets...")

write_csv(train_balanced, "data/train_balanced.csv")
write_csv(test_selected,  "data/test.csv")
saveRDS(top_features,     "data/feature_names.rds")
saveRDS(preproc,          "data/scaler.rds")   # save scaler for Python reference

message(sprintf("data/train_balanced.csv : %s rows x %s cols",
                format(nrow(train_balanced), big.mark=","), ncol(train_balanced)))
message(sprintf("data/test.csv           : %s rows x %s cols",
                format(nrow(test_selected),  big.mark=","), ncol(test_selected)))
message(sprintf("data/feature_names.rds  : %d feature names saved", length(top_features)))

# --- 12. Final summary table -------------------------------------------------
cat("\n===== PHASE 2 COMPLETE =====\n")
cat(sprintf("Original train size  : %s\n", format(nrow(train_selected),  big.mark=",")))
cat(sprintf("Balanced train size  : %s\n", format(nrow(train_balanced),  big.mark=",")))
cat(sprintf("Test size (untouched): %s\n", format(nrow(test_selected),   big.mark=",")))
cat(sprintf("Features selected    : %d (from %d after encoding)\n",
            length(top_features), length(feature_cols)))
cat("\nNext step -> Phase 3: R/phase3_baseline_models.R\n")
