# Load libraries
library(dplyr)
library(corrplot)
library(car)

set.seed(42)

# 1. Load Data and Clean
df <- read.csv("RetailStoreProductSalesDataset.csv")
df <- df %>% select(-matches("^X|^Unnamed"))

# 2. Feature Engineering & KPI Setup
# price and competitor_price are nearly identical (r = 0.969), so combining
# them into a single "price gap" variable removes the collinearity
df$price_gap <- df$price - df$competitor_price

# Binary target for logistic regression: 1 = above median return rate
median_rr      <- median(df$return_rate)
df$high_return <- ifelse(df$return_rate > median_rr, 1, 0)

# 3. Correlation Analysis
num_vars <- df %>% select(price_gap, discount, promotion_intensity, footfall,
                          ad_spend, stock_level, weather_index,
                          customer_sentiment, return_rate)

cor_matrix <- cor(num_vars)
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.7,
         title = "Correlation Matrix", mar = c(0, 0, 1.5, 0))

# 4. Principal Component Analysis
# Keeping raw price and competitor_price in the PCA to show how they
# load onto the same component -- this is what justifies replacing them
# with price_gap in the regression models
pca_data <- df %>% select(price, competitor_price, discount, promotion_intensity,
                          ad_spend, footfall, stock_level,
                          weather_index, customer_sentiment, return_rate)

pca_result <- prcomp(pca_data, scale. = TRUE)

summary(pca_result)
print(pca_result$rotation[, 1:2])

# "." as point labels so 15,000 observations don't make the plot unreadable
biplot(pca_result, xlabs = rep(".", nrow(pca_data)), cex = 0.7)

# 5. Linear Regression (KPI: Footfall)
# Standardising all variables so the betas are directly comparable to each
# other -- a larger beta means a more important predictor regardless of units
df_scaled <- df %>%
  mutate(across(c(footfall, price_gap, discount, promotion_intensity,
                  ad_spend, stock_level, weather_index, customer_sentiment),
                ~as.numeric(scale(.))))

lm_model <- lm(footfall ~ price_gap + discount + promotion_intensity +
                 ad_spend + stock_level + weather_index + customer_sentiment,
               data = df_scaled)

print("Linear Regression Summary (Standardised Betas):")
summary(lm_model)

# VIF check -- values below 5 confirm the collinearity was resolved by price_gap
print("Variance Inflation Factors:")
print(vif(lm_model))

# Bar chart of the betas to make the ranking easier to see
lm_coefs <- coef(lm_model)[-1]
barplot(sort(lm_coefs),
        horiz     = TRUE,
        las       = 1,
        col       = ifelse(sort(lm_coefs) > 0, "#4E79A7", "#E63946"),
        main      = "Standardised Betas -- Footfall Drivers",
        xlab      = "Standardised Beta Coefficient",
        cex.names = 0.8)
abline(v = 0, lwd = 1.5)

# 6. Logistic Regression (KPI: High Return)
log_model <- glm(high_return ~ price_gap + discount + promotion_intensity +
                   ad_spend + stock_level + weather_index + customer_sentiment,
                 data = df_scaled, family = binomial)

print("Logistic Regression Summary:")
summary(log_model)

# Odds ratios are easier to interpret than raw log-odds:
# OR > 1 = increases return risk, OR < 1 = decreases return risk
print("Odds Ratios:")
exp(coef(log_model))