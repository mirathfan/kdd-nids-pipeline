# =============================================================================
# KDD Cup 1999 - Network Intrusion Detection
# Phase 3: Baseline Models — Decision Tree & Random Forest
# Primary language: R
# Input : data/train_balanced.csv   (from Phase 2)
#         data/test.csv             (from Phase 2)
# Outputs: outputs/p8_dt_confusion.png
#          outputs/p9_rf_confusion.png
#          outputs/p10_model_comparison.png
#          data/baseline_results.rds   (metrics for Phase 5 comparison)
# =============================================================================

# --- 0. Packages -------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,  # data wrangling + ggplot2
  caret,      # confusionMatrix, train
  rpart,      # decision tree
  rpart.plot, # tree visualisation
  randomForest, # random forest
  scales,     # plot formatting
  knitr       # summary tables
)

dir.create("outputs", showWarnings = FALSE)

# --- 1. Load Phase 2 outputs -------------------------------------------------
message("Loading preprocessed data...")
train <- read_csv("data/train_balanced.csv", show_col_types = FALSE) %>%
  mutate(label = factor(label, levels = c("Normal","DoS","Probe","R2L","U2R")))

test  <- read_csv("data/test.csv", show_col_types = FALSE) %>%
  mutate(label = factor(label, levels = c("Normal","DoS","Probe","R2L","U2R")))

message(sprintf("Train: %s rows | Test: %s rows | Features: %d",
                format(nrow(train), big.mark=","),
                format(nrow(test),  big.mark=","),
                ncol(train) - 1))

# Helper: extract per-class and macro metrics from caret confusionMatrix
extract_metrics <- function(cm, model_name) {
  per_class <- cm$byClass %>%
    as.data.frame() %>%
    rownames_to_column("class") %>%
    mutate(class = str_remove(class, "Class: "),
           model = model_name) %>%
    select(model, class, Precision, Recall, F1)

  macro_f1  <- mean(per_class$F1,        na.rm = TRUE)
  macro_pre <- mean(per_class$Precision, na.rm = TRUE)
  macro_rec <- mean(per_class$Recall,    na.rm = TRUE)
  overall_acc <- cm$overall["Accuracy"]

  list(
    per_class  = per_class,
    macro_f1   = macro_f1,
    macro_pre  = macro_pre,
    macro_rec  = macro_rec,
    accuracy   = overall_acc,
    model_name = model_name
  )
}

# Helper: plot confusion matrix as heatmap
plot_confusion <- function(cm, title) {
  cm_tbl <- cm$table %>%
    as.data.frame() %>%
    rename(Predicted = Prediction, Actual = Reference)

  ggplot(cm_tbl, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = format(Freq, big.mark=",")),
              size = 3.5, fontface = "bold",
              color = ifelse(cm_tbl$Freq > max(cm_tbl$Freq) * 0.5,
                             "white", "gray20")) +
    scale_fill_gradient(low = "#E6F1FB", high = "#0C447C",
                        labels = label_comma()) +
    labs(title = title,
         subtitle = sprintf("Accuracy: %.2f%%",
                            cm$overall["Accuracy"] * 100),
         x = "Predicted class", y = "Actual class", fill = "Count") +
    theme_minimal(base_size = 12) +
    theme(
      plot.title    = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(color = "gray50"),
      axis.text.x   = element_text(angle = 30, hjust = 1),
      panel.grid    = element_blank()
    )
}

# =============================================================================
# MODEL 1: Decision Tree (rpart)
# =============================================================================
message("\n===== MODEL 1: Decision Tree =====")

set.seed(42)
dt_model <- rpart(
  label ~ .,
  data    = train,
  method  = "class",
  control = rpart.control(
    cp       = 0.0001,   # complexity parameter — lower = deeper tree
    minsplit = 20,       # min samples to attempt a split
    maxdepth = 15        # cap depth to prevent overfitting
  )
)

# Prune to best CP (lowest xerror)
best_cp  <- dt_model$cptable[which.min(dt_model$cptable[,"xerror"]), "CP"]
dt_pruned <- prune(dt_model, cp = best_cp)
message(sprintf("Decision Tree pruned. Best CP: %.6f | Tree size: %d leaves",
                best_cp, sum(dt_pruned$frame$var == "<leaf>")))

# Predict on test
dt_preds <- predict(dt_pruned, newdata = test, type = "class")
dt_cm    <- confusionMatrix(dt_preds, test$label)

message("\nDecision Tree — Test Results:")
print(dt_cm$overall[c("Accuracy","Kappa")])

dt_metrics <- extract_metrics(dt_cm, "Decision Tree")
message(sprintf("Macro F1: %.4f", dt_metrics$macro_f1))

message("\nPer-class metrics:")
print(dt_metrics$per_class)

# Save confusion matrix plot
p8 <- plot_confusion(dt_cm, "Decision Tree — confusion matrix")
ggsave("outputs/p8_dt_confusion.png", p8, width = 7, height = 5.5, dpi = 150)
message("Saved: outputs/p8_dt_confusion.png")

# Save tree diagram (top 4 levels for readability)
png("outputs/p8b_dt_tree.png", width = 1400, height = 900, res = 120)
rpart.plot(dt_pruned, type = 4, extra = 104,
           main = "Decision Tree structure (top levels)",
           cex  = 0.65, fallen.leaves = TRUE,
           box.palette = list("#E6F1FB","#E24B4A","#EF9F27","#1D9E75","#7F77DD"))
dev.off()
message("Saved: outputs/p8b_dt_tree.png")


# =============================================================================
# MODEL 2: Random Forest
# =============================================================================
message("\n===== MODEL 2: Random Forest =====")

# Use a stratified 50% sample to stay within 16GB RAM on Mac
# 204K rows is more than sufficient for a strong RF — OOB converges by tree 200
set.seed(42)
train_rf_idx <- createDataPartition(train$label, p = 0.50, list = FALSE)
train_rf     <- train[train_rf_idx, ]
message(sprintf("Training on stratified 50%% sample: %s rows (memory-safe for 16GB RAM)",
                format(nrow(train_rf), big.mark=",")))
message("Training Random Forest (300 trees, mtry = sqrt(p))...")
message("This may take 2-3 minutes...")

set.seed(42)
rf_model <- randomForest(
  label ~ .,
  data      = train_rf,
  ntree     = 300,
  mtry      = floor(sqrt(ncol(train_rf) - 1)),  # sqrt(20) ~ 4
  importance= TRUE,
  do.trace  = 100   # print OOB error every 100 trees
)

message("\nRandom Forest training complete.")
print(rf_model)

print(rf_model)

# Predict on test
rf_preds <- predict(rf_model, newdata = test, type = "class")
rf_cm    <- confusionMatrix(rf_preds, test$label)

message("\nRandom Forest — Test Results:")
print(rf_cm$overall[c("Accuracy","Kappa")])

rf_metrics <- extract_metrics(rf_cm, "Random Forest")
message(sprintf("Macro F1: %.4f", rf_metrics$macro_f1))

message("\nPer-class metrics:")
print(rf_metrics$per_class)

# Save confusion matrix plot
p9 <- plot_confusion(rf_cm, "Random Forest — confusion matrix")
ggsave("outputs/p9_rf_confusion.png", p9, width = 7, height = 5.5, dpi = 150)
message("Saved: outputs/p9_rf_confusion.png")

# =============================================================================
# MODEL COMPARISON PLOT
# =============================================================================
message("\n===== GENERATING COMPARISON PLOTS =====")

# Combine per-class F1 for both models
all_perclass <- bind_rows(dt_metrics$per_class, rf_metrics$per_class) %>%
  mutate(model = factor(model, levels = c("Decision Tree", "Random Forest")))

p10 <- ggplot(all_perclass, aes(x = class, y = F1, fill = model)) +
  geom_col(position = "dodge", width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", F1)),
            position = position_dodge(width = 0.65),
            vjust = -0.4, size = 3, color = "gray30") +
  scale_fill_manual(values = c("Decision Tree" = "#B5D4F4",
                               "Random Forest" = "#0C447C")) +
  scale_y_continuous(limits = c(0, 1.05), labels = label_number(accuracy=0.01)) +
  labs(
    title    = "Per-class F1 score: Decision Tree vs Random Forest",
    subtitle = "Evaluated on held-out test set (98,802 rows, no SMOTE)",
    x        = "Attack class", y = "F1 score", fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray50"),
    panel.grid.major.x = element_blank(),
    legend.position = "top"
  )

ggsave("outputs/p10_model_comparison.png", p10, width = 9, height = 5.5, dpi = 150)
message("Saved: outputs/p10_model_comparison.png")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
summary_tbl <- tibble(
  Model    = c("Decision Tree", "Random Forest"),
  Accuracy = c(dt_metrics$accuracy, rf_metrics$accuracy),
  `Macro F1`  = c(dt_metrics$macro_f1,  rf_metrics$macro_f1),
  `Macro Precision` = c(dt_metrics$macro_pre, rf_metrics$macro_pre),
  `Macro Recall`    = c(dt_metrics$macro_rec, rf_metrics$macro_rec)
) %>%
  mutate(across(where(is.numeric), ~round(.x, 4)))

cat("\n===== PHASE 3 RESULTS SUMMARY =====\n")
knitr::kable(summary_tbl, format = "simple") %>% print()

# =============================================================================
# SAVE FOR PHASE 5 COMPARISON
# =============================================================================
baseline_results <- list(
  dt = list(
    model      = dt_pruned,
    cm         = dt_cm,
    metrics    = dt_metrics,
    preds      = dt_preds
  ),
  rf = list(
    model      = rf_model,
    cm         = rf_cm,
    metrics    = rf_metrics,
    preds      = rf_preds
  ),
  summary    = summary_tbl,
  test_labels= test$label
)

saveRDS(baseline_results, "data/baseline_results.rds")
message("\nSaved: data/baseline_results.rds")

cat("\n===== PHASE 3 COMPLETE =====\n")
cat("Next step -> Phase 4: python/phase4_xgboost.py\n")
cat(sprintf("\nBaseline to beat in Phase 4:\n"))
cat(sprintf("  Random Forest macro F1 : %.4f\n", rf_metrics$macro_f1))
cat(sprintf("  Random Forest accuracy : %.4f\n", rf_metrics$accuracy))
