# Install required packages (run once)
install.packages("class")
install.packages("caret")

# Load libraries
library(class)
library(caret)

# Load dataset (CSV file)
data <- read.csv("diabetes.csv")

# View dataset
head(data)

# Convert Outcome to factor
data$Outcome <- as.factor(data$Outcome)

# Check missing values
colSums(is.na(data))

# Normalize data (important for KNN)
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

data_norm <- as.data.frame(lapply(data[ ,1:8], normalize))
data_norm$Outcome <- data$Outcome

# Split data into training and testing (70% train, 30% test)
set.seed(123)
train_index <- sample(1:nrow(data_norm), 0.7 * nrow(data_norm))

train_data <- data_norm[train_index, ]
test_data <- data_norm[-train_index, ]

# Apply KNN (k = 5)
pred <- knn(
  train = train_data[ ,1:8],
  test = test_data[ ,1:8],
  cl = train_data$Outcome,
  k = 5
)

# Confusion Matrix
confusionMatrix(pred, test_data$Outcome)

# Accuracy
accuracy <- sum(pred == test_data$Outcome) / length(pred)
print(paste("Accuracy:", accuracy))