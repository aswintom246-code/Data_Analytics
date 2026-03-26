# ==============================
# Diabetes Prediction using KNN
# ==============================

# Install packages (run once only)
install.packages("class")
install.packages("caret")
install.packages("dplyr")
install.packages("ggplot2")

# Load libraries
library(class)
library(caret)
library(dplyr)
library(ggplot2)

# ==============================
# 1. Load Dataset
# ==============================
df <- read.csv("diabetes.csv")

cat("Shape:", nrow(df), "x", ncol(df), "\n")
str(df)

# ==============================
# 2. Remove Duplicate Rows
# ==============================
df <- distinct(df)

# ==============================
# 3. Replace Invalid Zero Values
# ==============================
zero_cols <- c("Glucose","BloodPressure","SkinThickness","Insulin","BMI")

for(col in zero_cols){
  df[[col]][df[[col]] == 0] <- median(df[[col]][df[[col]] > 0], na.rm = TRUE)
}

# ==============================
# 4. Handle Missing Values
# ==============================
for(col in names(df)){
  if(any(is.na(df[[col]]))){
    df[[col]][is.na(df[[col]])] <- mean(df[[col]], na.rm = TRUE)
  }
}

cat("Missing values:", sum(is.na(df)), "\n")

# ==============================
# 5. Outlier Capping (99%)
# ==============================
numeric_cols <- c("Pregnancies","Glucose","BloodPressure","SkinThickness",
                  "Insulin","BMI","DiabetesPedigreeFunction","Age")

for(col in numeric_cols){
  cap <- quantile(df[[col]], 0.99, na.rm = TRUE)
  df[[col]] <- pmin(df[[col]], cap)
}

# ==============================
# 6. Normalization (Min-Max)
# ==============================
minmax <- function(x){
  (x - min(x)) / (max(x) - min(x))
}

df_scaled <- df
df_scaled[numeric_cols] <- lapply(df[numeric_cols], minmax)

# Convert target variable
df_scaled$Outcome <- as.factor(df_scaled$Outcome)

# ==============================
# 7. Train-Test Split (70-30)
# ==============================
set.seed(123)

train_idx <- sample(1:nrow(df_scaled), 0.7 * nrow(df_scaled))

train_data <- df_scaled[train_idx, ]
test_data  <- df_scaled[-train_idx, ]

cat("Train size:", nrow(train_data), "\n")
cat("Test size:", nrow(test_data), "\n")

# ==============================
# 8. Prepare Data
# ==============================
X_train <- as.matrix(train_data[, numeric_cols])
X_test  <- as.matrix(test_data[, numeric_cols])

y_train <- train_data$Outcome
y_test  <- test_data$Outcome

# ==============================
# 9. Apply KNN (K = 3)
# ==============================
set.seed(123)

pred <- knn(
  train = X_train,
  test  = X_test,
  cl    = y_train,
  k     = 3
)

# ==============================
# 10. Evaluation
# ==============================
conf_matrix <- confusionMatrix(pred, y_test)
print(conf_matrix)

accuracy <- sum(pred == y_test) / length(pred)
print(paste("Accuracy:", accuracy))

# ==============================
# 11. Visualization
# ==============================

# Plot 1: Diabetes Distribution
barplot(table(df$Outcome),
        col = c("blue","red"),
        main = "Diabetes Distribution",
        xlab = "Outcome (0=No, 1=Yes)",
        ylab = "Count")

# Plot 2: Model Accuracy
barplot(accuracy,
        col = "green",
        main = "KNN Model Accuracy",
        ylim = c(0,1))

# Plot 3: Age vs Glucose
plot(df$Age, df$Glucose,
     col = ifelse(df$Outcome == 1, "red", "black"),
     pch = 16,
     main = "Age vs Glucose",
     xlab = "Age",
     ylab = "Glucose")

plot(1:10, 1:10)

legend("topright",
       legend = c("No Diabetes","Diabetes"),
       col = c("black","red"),
       pch = 16)
