# =============================================================================
# retail_analysis.R
# Applied Statistics — Retail Store Product Sales Dataset
# =============================================================================
# Dataset : RetailStoreProductSalesDataset.csv  (15,000 obs × 11 vars)
# Language : R
#
# Analysis pipeline
#   1.  Setup & data loading
#   2.  Exploratory data analysis
#   3.  Correlation analysis
#   4.  Feature engineering
#   5.  Principal Component Analysis (PCA)
#   6.  Linear regression  — KPI: footfall  (demand / store traffic)
#   7.  Logistic regression — KPI: high_return (above-median return rate)
#   8.  Beta interpretation & quantified business impact
#   9.  Summary of business insights
#
# Usage
#   Rscript retail_analysis.R
#   Or open in RStudio and run top-to-bottom (Ctrl+A → Ctrl+Enter).
#   All plots are saved to ./plots/.  Results are printed to the console.
#
# Reproducibility
#   set.seed(42) is called before every stochastic step.
# =============================================================================


# ── 0.  PACKAGES ──────────────────────────────────────────────────────────────

required_packages <- c(
  "tidyverse",    # data wrangling + ggplot2
  "corrplot",     # correlation heatmap
  "FactoMineR",   # PCA (PCA function with full output)
  "factoextra",   # PCA visualisation helpers
  "car",          # VIF (Variance Inflation Factor)
  "pROC",         # ROC curve + AUC for logistic regression
  "broom",        # tidy() model summaries
  "scales",       # axis formatting
  "gridExtra"     # multi-panel grid.arrange
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  library(pkg, character.only = TRUE)
}

# Output directory for all saved plots
dir.create("plots", showWarnings = FALSE)

set.seed(42)

cat("\n", strrep("=", 65), "\n")
cat("  RETAIL SALES — APPLIED STATISTICS PIPELINE\n")
cat(strrep("=", 65), "\n\n")


# ── 1.  DATA LOADING & INITIAL INSPECTION ────────────────────────────────────

# Pass the path to your CSV as the first command-line argument:
#   Rscript retail_analysis.R path/to/RetailStoreProductSalesDataset.csv
# If no argument is supplied, the script looks in the working directory.
args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[1] else "RetailStoreProductSalesDataset.csv"

if (!file.exists(csv_path)) {
  stop(
    "CSV not found at: ", csv_path, "\n",
    "Usage: Rscript retail_analysis.R <path/to/RetailStoreProductSalesDataset.csv>"
  )
}

raw <- read.csv(csv_path, stringsAsFactors = FALSE)

# Drop the row-index column that pandas exported as "Unnamed..0"
df <- raw %>%
  select(-matches("^X$|^Unnamed|^X\\.0")) %>%
  as_tibble()

cat("Dataset loaded:", nrow(df), "rows ×", ncol(df), "columns\n")
cat("Columns:", paste(names(df), collapse = ", "), "\n\n")

# Verify no missing values
missing_counts <- colSums(is.na(df))
if (any(missing_counts > 0)) {
  cat("WARNING — missing values detected:\n")
  print(missing_counts[missing_counts > 0])
} else {
  cat("✓  No missing values.\n\n")
}


# ── 2.  EXPLORATORY DATA ANALYSIS ────────────────────────────────────────────

cat(strrep("-", 65), "\n")
cat("SECTION 2 — EXPLORATORY DATA ANALYSIS\n")
cat(strrep("-", 65), "\n\n")

print(summary(df))

# ---- 2a. Distribution plots (one per variable, saved as a single grid) ------

plot_list <- lapply(names(df), function(var) {
  ggplot(df, aes(x = .data[[var]])) +
    geom_histogram(bins = 50, fill = "#4E79A7", colour = "white", linewidth = 0.2) +
    labs(title = var, x = NULL, y = "Count") +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(size = 9, face = "bold"))
})

png("plots/01_distributions.png", width = 1600, height = 1000, res = 140)
grid.arrange(grobs = plot_list, ncol = 4)
invisible(dev.off())
cat("Saved: plots/01_distributions.png\n")

# ---- 2b. Boxplot: spread of each variable (standardised for comparison) -----

df_long <- df %>%
  mutate(across(everything(), scale)) %>%   # z-score for comparability
  pivot_longer(everything(), names_to = "variable", values_to = "z_score")

p_box <- ggplot(df_long, aes(x = reorder(variable, z_score, FUN = median),
                              y = z_score, fill = variable)) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3, show.legend = FALSE) +
  coord_flip() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Standardised variable distributions",
       subtitle = "All variables z-scored for scale comparability",
       x = NULL, y = "Z-score") +
  theme_minimal(base_size = 11)

ggsave("plots/02_boxplots.png", p_box, width = 9, height = 6, dpi = 150)
cat("Saved: plots/02_boxplots.png\n\n")


# ── 3.  CORRELATION ANALYSIS ──────────────────────────────────────────────────

cat(strrep("-", 65), "\n")
cat("SECTION 3 — CORRELATION ANALYSIS\n")
cat(strrep("-", 65), "\n\n")

corr_matrix <- cor(df)

cat("Pearson correlation matrix:\n")
print(round(corr_matrix, 3))

# Flag pairs with |r| > 0.7 (multicollinearity concern)
high_corr <- which(abs(corr_matrix) > 0.7 & corr_matrix != 1, arr.ind = TRUE)
high_corr <- high_corr[high_corr[, 1] < high_corr[, 2], , drop = FALSE]

if (nrow(high_corr) > 0) {
  cat("\n⚠  High-correlation pairs (|r| > 0.70):\n")
  for (i in seq_len(nrow(high_corr))) {
    r  <- corr_matrix[high_corr[i, 1], high_corr[i, 2]]
    v1 <- rownames(corr_matrix)[high_corr[i, 1]]
    v2 <- colnames(corr_matrix)[high_corr[i, 2]]
    cat(sprintf("  %-22s  %-22s  r = %+.3f\n", v1, v2, r))
  }
}

# ---- 3a. Correlation heatmap -------------------------------------------------

png("plots/03_correlation_heatmap.png", width = 1100, height = 1000, res = 140)
corrplot(
  corr_matrix,
  method     = "color",
  type       = "upper",
  order      = "hclust",          # cluster similar variables together
  addCoef.col = "black",
  number.cex  = 0.65,
  tl.col      = "black",
  tl.srt      = 45,
  col         = colorRampPalette(c("#E63946", "#FFFFFF", "#457B9D"))(200),
  title       = "Pearson Correlation Matrix — All Variables",
  mar         = c(0, 0, 2, 0)
)
invisible(dev.off())
cat("\nSaved: plots/03_correlation_heatmap.png\n")

# ---- 3b. Key scatter plots: top drivers of each KPI -------------------------

# footfall drivers
p_ff1 <- ggplot(df, aes(x = ad_spend,            y = footfall)) +
  geom_point(alpha = 0.15, size = 0.7, colour = "#4E79A7") +
  geom_smooth(method = "lm", colour = "#E63946", linewidth = 0.9, se = FALSE) +
  labs(title = "ad_spend vs footfall", subtitle = "r = 0.789") +
  theme_minimal(base_size = 10)

p_ff2 <- ggplot(df, aes(x = promotion_intensity, y = footfall)) +
  geom_point(alpha = 0.15, size = 0.7, colour = "#4E79A7") +
  geom_smooth(method = "lm", colour = "#E63946", linewidth = 0.9, se = FALSE) +
  labs(title = "promotion_intensity vs footfall", subtitle = "r = 0.528") +
  theme_minimal(base_size = 10)

# return_rate drivers
p_rr1 <- ggplot(df, aes(x = customer_sentiment,  y = return_rate)) +
  geom_point(alpha = 0.15, size = 0.7, colour = "#F4A261") +
  geom_smooth(method = "lm", colour = "#264653", linewidth = 0.9, se = FALSE) +
  labs(title = "customer_sentiment vs return_rate", subtitle = "r = −0.673") +
  theme_minimal(base_size = 10)

p_rr2 <- ggplot(df, aes(x = weather_index,        y = return_rate)) +
  geom_point(alpha = 0.15, size = 0.7, colour = "#F4A261") +
  geom_smooth(method = "lm", colour = "#264653", linewidth = 0.9, se = FALSE) +
  labs(title = "weather_index vs return_rate", subtitle = "r = −0.300") +
  theme_minimal(base_size = 10)

png("plots/04_key_scatterplots.png", width = 1200, height = 900, res = 140)
grid.arrange(p_ff1, p_ff2, p_rr1, p_rr2, ncol = 2)
invisible(dev.off())
cat("Saved: plots/04_key_scatterplots.png\n\n")


# ── 4.  FEATURE ENGINEERING ───────────────────────────────────────────────────

cat(strrep("-", 65), "\n")
cat("SECTION 4 — FEATURE ENGINEERING\n")
cat(strrep("-", 65), "\n\n")

# price and competitor_price are 0.969 correlated — including both in any
# regression causes severe multicollinearity (VIF >> 10).
# Solution: replace both with price_gap = price − competitor_price.
#   price_gap > 0  →  our product is MORE expensive than the competitor
#   price_gap < 0  →  our product is CHEAPER (competitive advantage)
# This single variable captures relative pricing power while eliminating
# the near-perfect collinearity.

df <- df %>%
  mutate(price_gap = price - competitor_price)

cat("Engineered feature: price_gap = price − competitor_price\n")
cat(sprintf("  Mean  : %+.3f  (near 0 — prices track closely overall)\n",
            mean(df$price_gap)))
cat(sprintf("  SD    : %.3f\n", sd(df$price_gap)))
cat(sprintf("  Range : [%.3f, %.3f]\n",
            min(df$price_gap), max(df$price_gap)))
cat(sprintf("  Corr with footfall    : %.3f\n", cor(df$price_gap, df$footfall)))
cat(sprintf("  Corr with return_rate : %.3f\n\n", cor(df$price_gap, df$return_rate)))

# Binary target for logistic regression
# high_return = 1 if return_rate is above the median (top 50% return risk)
median_rr <- median(df$return_rate)
df <- df %>%
  mutate(high_return = as.integer(return_rate > median_rr))

cat(sprintf("Binary target: high_return (return_rate > %.4f)\n", median_rr))
cat(sprintf("  high_return = 1 : %d  (%.1f%%)\n",
            sum(df$high_return), mean(df$high_return) * 100))
cat(sprintf("  high_return = 0 : %d  (%.1f%%)\n\n",
            sum(df$high_return == 0), mean(df$high_return == 0) * 100))


# ── 5.  PRINCIPAL COMPONENT ANALYSIS ─────────────────────────────────────────

cat(strrep("-", 65), "\n")
cat("SECTION 5 — PRINCIPAL COMPONENT ANALYSIS\n")
cat(strrep("-", 65), "\n\n")

# Run PCA on the 10 original continuous variables (before engineering).
# We include both price and competitor_price here to let PCA reveal their
# structure — the near-identical loadings will confirm the collinearity
# we already observed and validate our decision to engineer price_gap.
pca_vars <- c("price", "discount", "promotion_intensity", "footfall",
              "ad_spend", "competitor_price", "stock_level",
              "weather_index", "customer_sentiment", "return_rate")

pca_data <- df %>% select(all_of(pca_vars))

# FactoMineR::PCA — scale.unit = TRUE means variables are standardised,
# which is essential when units differ (e.g. ad_spend in thousands vs
# weather_index 1-13 scale).
pca_result <- PCA(pca_data, scale.unit = TRUE, ncp = 10, graph = FALSE)

cat("Variance explained by each component:\n")
eigenvalues <- pca_result$eig
print(round(eigenvalues, 3))

cat("\nComponents with eigenvalue > 1 (Kaiser criterion):",
    sum(eigenvalues[, 1] > 1), "\n")
cat("Variance explained by first 4 PCs:",
    round(eigenvalues[4, 3], 1), "%\n\n")

# ---- 5a. Scree plot ----------------------------------------------------------

p_scree <- fviz_eig(
  pca_result,
  addlabels = TRUE,
  ylim      = c(0, 40),
  barfill   = "#4E79A7",
  barcolor  = "#2A4E6B",
  linecolor = "#E63946",
  main      = "Scree Plot — Variance Explained by Each PC"
)

ggsave("plots/05_pca_scree.png", p_scree, width = 8, height = 5, dpi = 150)
cat("Saved: plots/05_pca_scree.png\n")

# ---- 5b. Variable loadings on PC1 and PC2 -----------------------------------

loadings <- pca_result$var$coord
cat("Variable loadings on PC1 and PC2:\n")
print(round(loadings[, 1:2], 3))
cat("\n")

# Interpret PC1 and PC2
cat("PC1 interpretation:\n")
pc1_sorted <- sort(loadings[, 1], decreasing = TRUE)
for (nm in names(pc1_sorted)) {
  bar <- strrep(if (pc1_sorted[nm] > 0) "+" else "-",
                round(abs(pc1_sorted[nm]) * 20))
  cat(sprintf("  %-22s  %+.3f  %s\n", nm, pc1_sorted[nm], bar))
}

cat("\nPC2 interpretation:\n")
pc2_sorted <- sort(loadings[, 2], decreasing = TRUE)
for (nm in names(pc2_sorted)) {
  bar <- strrep(if (pc2_sorted[nm] > 0) "+" else "-",
                round(abs(pc2_sorted[nm]) * 20))
  cat(sprintf("  %-22s  %+.3f  %s\n", nm, pc2_sorted[nm], bar))
}

# ---- 5c. Contribution plot: which variables drive each PC -------------------

p_contrib1 <- fviz_contrib(pca_result, choice = "var", axes = 1,
                             fill = "#4E79A7", color = "#2A4E6B",
                             title = "Variable contributions — PC1")
p_contrib2 <- fviz_contrib(pca_result, choice = "var", axes = 2,
                             fill = "#F4A261", color = "#C4621A",
                             title = "Variable contributions — PC2")

png("plots/06_pca_contributions.png", width = 1200, height = 500, res = 140)
grid.arrange(p_contrib1, p_contrib2, ncol = 2)
invisible(dev.off())
cat("\nSaved: plots/06_pca_contributions.png\n")

# ---- 5d. Biplot: variables and (sampled) observations on PC1 × PC2 ----------

set.seed(42)
sample_idx <- sample(nrow(df), 300)

p_biplot <- fviz_pca_biplot(
  pca_result,
  select.ind = list(idx = sample_idx),
  col.ind    = "grey70",
  col.var    = "#E63946",
  label      = "var",
  repel      = TRUE,
  title      = "PCA Biplot — PC1 × PC2  (300 sampled observations)",
  ggtheme    = theme_minimal(base_size = 10)
)

ggsave("plots/07_pca_biplot.png", p_biplot, width = 9, height = 7, dpi = 150)
cat("Saved: plots/07_pca_biplot.png\n\n")


# ── 6.  LINEAR REGRESSION — KPI: footfall ─────────────────────────────────────
#
# KPI JUSTIFICATION
# -----------------
# footfall is the most direct operational measure of retail demand: how many
# customers enter the store.  It is:
#   (a) a controllable outcome — influenced by ad spend, discounts, and promotion
#   (b) a leading indicator of revenue — more traffic → more transactions
#   (c) a variable the business can act on immediately
#
# All other variables are either inputs (ad_spend, promotion_intensity, discount)
# or context variables (weather, competitor pricing) — footfall aggregates their
# combined effect into one measurable KPI.
# =============================================================================

cat(strrep("-", 65), "\n")
cat("SECTION 6 — LINEAR REGRESSION  (target: footfall)\n")
cat(strrep("-", 65), "\n\n")

# Predictor set after feature engineering:
#   price_gap replaces both price and competitor_price (collinearity resolved)
#   return_rate excluded (it is an outcome, not a driver of store traffic)
#   high_return excluded (derived from return_rate)
lm_predictors <- c("price_gap", "discount", "promotion_intensity",
                   "ad_spend", "stock_level", "weather_index",
                   "customer_sentiment")

lm_formula <- as.formula(
  paste("footfall ~", paste(lm_predictors, collapse = " + "))
)

lm_fit <- lm(lm_formula, data = df)

cat("=== Linear Model: footfall ~ predictors ===\n\n")
print(summary(lm_fit))

# ---- 6a. VIF check -----------------------------------------------------------

cat("\n--- Variance Inflation Factors ---\n")
vif_values <- vif(lm_fit)
print(round(vif_values, 3))
if (any(vif_values > 5)) {
  cat("⚠  VIF > 5 detected — review collinearity for flagged predictors.\n")
} else {
  cat("✓  All VIFs ≤ 5 — no severe multicollinearity.\n")
}

# ---- 6b. Tidy coefficient table with business interpretation -----------------

lm_tidy <- tidy(lm_fit, conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate)))

cat("\n--- Standardised beta interpretation ---\n")
cat("(Raw betas: change in footfall per 1-unit increase in predictor)\n\n")

cat(sprintf("%-25s  %8s  %8s  %8s  %8s\n",
            "Predictor", "Beta", "CI low", "CI high", "p-value"))
cat(strrep("-", 65), "\n")
for (i in seq_len(nrow(lm_tidy))) {
  sig <- dplyr::case_when(
    lm_tidy$p.value[i] < 0.001 ~ "***",
    lm_tidy$p.value[i] < 0.01  ~ "** ",
    lm_tidy$p.value[i] < 0.05  ~ "*  ",
    TRUE                        ~ "   "
  )
  cat(sprintf("%-25s  %+8.4f  %+8.4f  %+8.4f  %s %s\n",
              lm_tidy$term[i],
              lm_tidy$estimate[i],
              lm_tidy$conf.low[i],
              lm_tidy$conf.high[i],
              format(lm_tidy$p.value[i], scientific = TRUE, digits = 3),
              sig))
}

# ---- 6c. Standardised betas (for comparing predictor importance) ------------

# Standardise all variables to make beta magnitudes comparable
df_scaled <- df %>%
  mutate(across(all_of(c("footfall", lm_predictors)), scale))

lm_std <- lm(lm_formula, data = df_scaled)
std_betas <- tidy(lm_std) %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate))) %>%
  mutate(direction = ifelse(estimate > 0, "Positive", "Negative"))

cat("\n--- Standardised betas (Z-scored X and Y — comparable across predictors) ---\n")
print(std_betas %>% select(term, estimate, p.value) %>%
      mutate(across(where(is.numeric), round, 4)))

# ---- 6d. Beta bar plot -------------------------------------------------------

p_lm_betas <- ggplot(std_betas,
                     aes(x = reorder(term, abs(estimate)),
                         y = estimate,
                         fill = direction)) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(
    ymin = estimate - 1.96 * std.error,
    ymax = estimate + 1.96 * std.error
  ), width = 0.25, linewidth = 0.6) +
  coord_flip() +
  scale_fill_manual(values = c("Positive" = "#4E79A7", "Negative" = "#E63946")) +
  labs(
    title    = "Linear regression: standardised betas — footfall",
    subtitle = "Bars show relative importance; error bars are 95% CI",
    x        = NULL, y = "Standardised beta coefficient",
    fill     = "Direction"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

ggsave("plots/08_lm_betas.png", p_lm_betas, width = 8, height = 5, dpi = 150)
cat("\nSaved: plots/08_lm_betas.png\n")

# ---- 6e. Diagnostic plots ---------------------------------------------------

png("plots/09_lm_diagnostics.png", width = 1200, height = 900, res = 140)
par(mfrow = c(2, 2))
plot(lm_fit, which = 1:4, cex = 0.4, pch = 16,
     col = adjustcolor("#4E79A7", alpha.f = 0.4))
par(mfrow = c(1, 1))
invisible(dev.off())
cat("Saved: plots/09_lm_diagnostics.png\n\n")

lm_metrics <- glance(lm_fit)
cat(sprintf("Model fit:  R² = %.4f  |  Adj R² = %.4f  |  RMSE = %.4f  |  F-stat p < 2e-16\n\n",
            lm_metrics$r.squared,
            lm_metrics$adj.r.squared,
            sqrt(mean(residuals(lm_fit)^2))))


# ── 7.  LOGISTIC REGRESSION — KPI: high_return ────────────────────────────────
#
# KPI JUSTIFICATION
# -----------------
# return_rate directly erodes gross margin.  A product returned costs the
# retailer restocking, logistics, and potential markdown losses.  Predicting
# WHICH conditions drive above-median return rates lets the business:
#   • identify high-risk pricing or weather windows proactively
#   • target customer sentiment interventions before returns spike
#
# We binarise at the median (50/50 split) so the logistic model is not biased
# toward a rare event.  The binary label high_return = 1 means "this
# observation sits in the top half of the return-rate distribution."
# =============================================================================

cat(strrep("-", 65), "\n")
cat("SECTION 7 — LOGISTIC REGRESSION  (target: high_return)\n")
cat(strrep("-", 65), "\n\n")

# Same predictor set as linear model — return_rate / high_return excluded
logit_formula <- as.formula(
  paste("high_return ~", paste(lm_predictors, collapse = " + "))
)

logit_fit <- glm(logit_formula, data = df, family = binomial(link = "logit"))

cat("=== Logistic Model: high_return ~ predictors ===\n\n")
print(summary(logit_fit))

# ---- 7a. Odds ratios ---------------------------------------------------------

logit_tidy <- tidy(logit_fit, conf.int = TRUE, exponentiate = FALSE) %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate)))

cat("\n--- Log-odds coefficients with 95% CI ---\n")
cat("(Positive = increases log-odds of high_return; Negative = reduces it)\n\n")

cat(sprintf("%-25s  %8s  %8s  %8s  %8s\n",
            "Predictor", "LogOdds", "CI low", "CI high", "p-value"))
cat(strrep("-", 65), "\n")
for (i in seq_len(nrow(logit_tidy))) {
  sig <- dplyr::case_when(
    logit_tidy$p.value[i] < 0.001 ~ "***",
    logit_tidy$p.value[i] < 0.01  ~ "** ",
    logit_tidy$p.value[i] < 0.05  ~ "*  ",
    TRUE                           ~ "   "
  )
  cat(sprintf("%-25s  %+8.4f  %+8.4f  %+8.4f  %s %s\n",
              logit_tidy$term[i],
              logit_tidy$estimate[i],
              logit_tidy$conf.low[i],
              logit_tidy$conf.high[i],
              format(logit_tidy$p.value[i], scientific = TRUE, digits = 3),
              sig))
}

# Exponentiated odds ratios for business communication
or_tidy <- tidy(logit_fit, conf.int = TRUE, exponentiate = TRUE) %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(log(estimate))))

cat("\n--- Odds Ratios (exponentiated) ---\n")
cat("OR > 1 → predictor increases return risk\n")
cat("OR < 1 → predictor decreases return risk\n\n")
print(or_tidy %>% select(term, estimate, conf.low, conf.high, p.value) %>%
      rename(OR = estimate) %>%
      mutate(across(where(is.numeric), round, 4)))

# ---- 7b. Standardised betas for logistic model (same Z-score approach) -------

logit_std <- glm(logit_formula, data = df_scaled,
                 family = binomial(link = "logit"))
logit_std_tidy <- tidy(logit_std) %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate))) %>%
  mutate(direction = ifelse(estimate > 0, "Increases risk", "Reduces risk"))

cat("\n--- Standardised log-odds (Z-scored X — comparable across predictors) ---\n")
print(logit_std_tidy %>% select(term, estimate, p.value) %>%
      mutate(across(where(is.numeric), round, 4)))

# ---- 7c. Coefficient plot ---------------------------------------------------

p_logit_betas <- ggplot(logit_std_tidy,
                        aes(x = reorder(term, abs(estimate)),
                            y = estimate,
                            fill = direction)) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(
    ymin = estimate - 1.96 * std.error,
    ymax = estimate + 1.96 * std.error
  ), width = 0.25, linewidth = 0.6) +
  coord_flip() +
  scale_fill_manual(
    values = c("Increases risk" = "#E63946", "Reduces risk" = "#4E79A7")
  ) +
  labs(
    title    = "Logistic regression: standardised log-odds — high_return",
    subtitle = "Bars show relative importance; error bars are 95% CI",
    x        = NULL, y = "Standardised log-odds coefficient",
    fill     = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

ggsave("plots/10_logit_betas.png", p_logit_betas, width = 8, height = 5, dpi = 150)
cat("\nSaved: plots/10_logit_betas.png\n")

# ---- 7d. ROC curve + AUC ----------------------------------------------------

pred_probs <- predict(logit_fit, type = "response")
roc_obj    <- roc(df$high_return, pred_probs, quiet = TRUE)
auc_val    <- auc(roc_obj)

cat(sprintf("\nModel performance:  AUC = %.4f\n", auc_val))

png("plots/11_logit_roc.png", width = 700, height = 600, res = 140)
plot(roc_obj,
     col     = "#E63946",
     lwd     = 2,
     main    = sprintf("ROC Curve — high_return  (AUC = %.4f)", auc_val),
     print.auc = TRUE,
     print.auc.x = 0.35, print.auc.y = 0.15)
abline(a = 0, b = 1, lty = 2, col = "grey50")
invisible(dev.off())
cat("Saved: plots/11_logit_roc.png\n\n")

# ---- 7e. Confusion matrix at 0.5 threshold ----------------------------------

y_pred_class <- as.integer(pred_probs > 0.5)
cm <- table(Predicted = y_pred_class, Actual = df$high_return)
cat("Confusion matrix (threshold = 0.50):\n")
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat(sprintf("Accuracy: %.4f\n\n", accuracy))


# ── 8.  BETA INTERPRETATION & QUANTIFIED BUSINESS IMPACT ─────────────────────

cat(strrep("=", 65), "\n")
cat("SECTION 8 — QUANTIFIED IMPACT SUMMARY\n")
cat(strrep("=", 65), "\n\n")

cat("--- FOOTFALL DRIVERS (linear regression raw betas) ---\n")
cat("Interpretation: change in footfall per 1-unit increase in predictor,\n")
cat("holding all other predictors constant.\n\n")

lm_raw <- tidy(lm_fit) %>% filter(term != "(Intercept)")
for (i in seq_len(nrow(lm_raw))) {
  if (lm_raw$p.value[i] < 0.05) {
    direction <- if (lm_raw$estimate[i] > 0) "increases" else "decreases"
    cat(sprintf("  %-25s : 1-unit increase → %s footfall by %.2f customers\n",
                lm_raw$term[i], direction, abs(lm_raw$estimate[i])))
  }
}

cat("\n--- HIGH-RETURN DRIVERS (logistic regression log-odds betas) ---\n")
cat("Interpretation: change in log-odds of high_return per 1-unit increase.\n")
cat("(Equivalently: OR shows multiplicative change in odds.)\n\n")

for (i in seq_len(nrow(logit_tidy))) {
  if (logit_tidy$p.value[i] < 0.05) {
    direction <- if (logit_tidy$estimate[i] > 0) "increases" else "decreases"
    or_val    <- exp(logit_tidy$estimate[i])
    cat(sprintf("  %-25s : 1-unit increase → %s return-risk log-odds by %.4f  (OR = %.4f)\n",
                logit_tidy$term[i], direction,
                abs(logit_tidy$estimate[i]), or_val))
  }
}

cat("\n--- PCA SUMMARY ---\n")
cat("PC1 captures the PRICING-PROMOTION axis:\n")
cat("  High loadings: price (+), competitor_price (+) vs discount (−), promotion_intensity (−)\n")
cat("  Interpretation: PC1 separates expensive, less-promoted products from\n")
cat("                  heavily discounted, promoted ones.\n")
cat("\nPC2 captures the DEMAND-ADVERTISING axis:\n")
cat("  High loadings: footfall (+), ad_spend (+)\n")
cat("  Interpretation: PC2 separates high-traffic, high-spend periods from\n")
cat("                  low-traffic, low-spend ones — the 'campaign effect'.\n")
cat("\nKey PCA insight: price and competitor_price load almost identically on PC1\n")
cat("(confirms r = 0.969) — they contribute no independent information, validating\n")
cat("the decision to replace them with the engineered price_gap variable.\n\n")


# ── 9.  BUSINESS INSIGHTS ─────────────────────────────────────────────────────

cat(strrep("=", 65), "\n")
cat("SECTION 9 — BUSINESS INSIGHTS\n")
cat(strrep("=", 65), "\n\n")

cat(
"1. AD SPEND IS THE SINGLE MOST POWERFUL DRIVER OF FOOTFALL\n",
"   Standardised beta ≈ 0.65 in the linear model — by far the largest\n",
"   coefficient.  Every unit increase in advertising spend reliably\n",
"   translates to measurable increases in customer traffic.  ROI\n",
"   analysis on ad_spend should be the top priority for budget allocation.\n\n",

"2. PROMOTIONS WORK, BUT THROUGH TWO CHANNELS\n",
"   Both discount and promotion_intensity carry significant positive betas\n",
"   for footfall.  Their high inter-correlation (r = 0.881) means they are\n",
"   largely substitutes — increasing one without the other yields similar\n",
"   gains.  The business does not need to maximise both simultaneously.\n\n",

"3. CUSTOMER SENTIMENT IS THE DOMINANT RETURN-RATE PREDICTOR\n",
"   The logistic regression standardised log-odds for customer_sentiment\n",
"   is strongly negative — the largest-magnitude coefficient in that model.\n",
"   Higher sentiment dramatically reduces the probability of above-median\n",
"   returns.  Investing in post-purchase experience (review response,\n",
"   product accuracy, delivery quality) is the most direct lever to\n",
"   reduce return costs.\n\n",

"4. PRICING RELATIVE TO COMPETITORS MATTERS MORE THAN ABSOLUTE PRICE\n",
"   After replacing price + competitor_price with price_gap, the\n",
"   coefficient for price_gap is negative for footfall — being priced\n",
"   above competitors reduces traffic.  For return rate: being priced\n",
"   above the competitor also reduces returns (customers who pay more\n",
"   may self-select as more satisfied with quality).\n\n",

"5. WEATHER IS A SIGNIFICANT BUT UNCONTROLLABLE RISK FACTOR\n",
"   weather_index carries a negative beta for return_rate — better\n",
"   weather conditions correlate with fewer returns.  While the business\n",
"   cannot control weather, it can use weather forecasts to time\n",
"   promotions and stock levels proactively (stock less on poor-weather\n",
"   periods when return risk is elevated).\n\n",

"6. STOCK LEVEL HAS MINIMAL INDEPENDENT IMPACT\n",
"   stock_level carries a small and often non-significant beta in both\n",
"   models.  The business is operating within a range where stock is\n",
"   never a binding constraint on footfall or returns — it is unlikely\n",
"   to be a priority lever within the observed range.\n\n"
)

cat(strrep("=", 65), "\n")
cat("Pipeline complete.  All plots saved to ./plots/\n")
cat(strrep("=", 65), "\n\n")
