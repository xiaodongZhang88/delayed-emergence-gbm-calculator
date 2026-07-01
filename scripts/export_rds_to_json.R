args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript scripts/export_rds_to_json.R /path/to/model.rds /path/to/output_dir", call. = FALSE)
}

rds_path <- normalizePath(args[[1]], mustWork = TRUE)
out_dir <- normalizePath(args[[2]], mustWork = TRUE)

suppressPackageStartupMessages({
  library(gbm)
  library(jsonlite)
})

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

obj <- readRDS(rds_path)

get_private <- function(x) {
  get("private", get(".__enclos_env__", x))
}

task_backend <- obj$task$backend
backend_private <- get_private(task_backend)
source_data <- as.data.frame(backend_private$.data)

feature_names <- c("ALT", "BE", "GNRI", "RBC", "Urea", "WBC", "age")
required_columns <- c("DL1", feature_names)
missing_columns <- setdiff(required_columns, names(source_data))
if (length(missing_columns) > 0) {
  stop(sprintf("RDS is missing required columns: %s", paste(missing_columns, collapse = ", ")), call. = FALSE)
}

model_result_data <- get_private(obj$model)$.data$data
param_vals <- model_result_data$learner_components$learner_param_vals[[1]]

model_data <- source_data[, required_columns]
model_data$DL1 <- as.numeric(as.character(model_data$DL1))

set.seed(20260701)
final_model <- gbm(
  formula = DL1 ~ ALT + BE + GNRI + RBC + Urea + WBC + age,
  distribution = param_vals$distribution %||% "bernoulli",
  data = model_data,
  n.trees = param_vals$n.trees,
  interaction.depth = param_vals$interaction.depth,
  n.minobsinnode = param_vals$n.minobsinnode,
  shrinkage = param_vals$shrinkage,
  bag.fraction = 0.5,
  train.fraction = 1,
  keep.data = FALSE,
  n.cores = min(4, parallel::detectCores()),
  verbose = FALSE
)

tree_to_json <- function(index) {
  tree <- pretty.gbm.tree(final_model, i.tree = index)
  list(
    split_var = as.integer(tree$SplitVar),
    split_code = as.numeric(tree$SplitCodePred),
    left = as.integer(tree$LeftNode),
    right = as.integer(tree$RightNode),
    missing = as.integer(tree$MissingNode),
    prediction = as.numeric(tree$Prediction)
  )
}

median_row <- as.data.frame(as.list(vapply(model_data[, feature_names], median, numeric(1))))
names(median_row) <- feature_names

default_row <- data.frame(
  ALT = 14,
  BE = 0.81,
  GNRI = 102.90,
  RBC = 4.46,
  Urea = 5.15,
  WBC = 5.50,
  age = 62
)

validation_rows <- rbind(default_row, median_row)
validation_link <- as.numeric(predict(final_model, validation_rows, n.trees = final_model$n.trees, type = "link"))
validation_probability <- as.numeric(predict(final_model, validation_rows, n.trees = final_model$n.trees, type = "response"))

payload <- list(
  format = "r-gbm-json-v1",
  source = "Reconstructed full-data GBM from mlr3 model.rds",
  seed = 20260701,
  initF = as.numeric(final_model$initF),
  var_names = final_model$var.names,
  distribution = "bernoulli",
  class_names = c("0", "1"),
  positive_class = "1",
  n_trees = as.integer(final_model$n.trees),
  interaction_depth = as.integer(final_model$interaction.depth),
  n_minobsinnode = as.integer(final_model$n.minobsinnode),
  shrinkage = as.numeric(final_model$shrinkage),
  bag_fraction = as.numeric(final_model$bag.fraction),
  training_summary = list(
    n = nrow(model_data),
    outcome_negative = sum(model_data$DL1 == 0),
    outcome_positive = sum(model_data$DL1 == 1)
  ),
  validation = list(
    columns = names(validation_rows),
    rows = unname(split(validation_rows, seq_len(nrow(validation_rows)))),
    link = validation_link,
    probability = validation_probability
  ),
  trees = lapply(seq_len(final_model$n.trees), tree_to_json)
)

model_dir <- file.path(out_dir, "models")
data_dir <- file.path(out_dir, "data")
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

model_path <- file.path(model_dir, "final_gbm_model.json.gz")
con <- gzfile(model_path, open = "wt")
on.exit(close(con), add = TRUE)
write_json(payload, con, auto_unbox = TRUE, digits = 16)
close(con)

background_path <- file.path(data_dir, "background.csv")
write.csv(median_row, background_path, row.names = FALSE)

cat("Wrote model:", model_path, "\n")
cat("Wrote background:", background_path, "\n")
cat("Default probability:", validation_probability[[1]], "\n")
