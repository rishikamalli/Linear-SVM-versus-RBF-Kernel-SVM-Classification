# Kernel SVM Classification

#Importing the data 
dataset <- read.csv("Social_Network_Ads.csv")
dataset = dataset[3:5]

#splitting the dataset into training set and Test set 
#install.packages("caTools")
library(caTools) 
set.seed(123) 
split = sample.split(dataset$Purchased, SplitRatio = 0.75) 
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling preferred on Classification Problems
#Scaling only the Independent Variable
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

#Fitting Linear SVM Classifier to the Training set
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set, 
                 type = 'C-classification',
                 kernel = 'linear')
#Predict the test set results 
y_pred = predict(classifier, newdata = test_set[-3]) 

#Making the Confusion Matrix - to evaluate how many of our pred is true 
cm = table(test_set[,3], y_pred)

#Visualising the Test set 
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[,1]) - 1, max(set[,1]) + 1, by = 0.01) 
X2 = seq(min(set[,2]) - 1, max(set[,2]) + 1, by = 0.01) 
grid_set = expand.grid(X1, X2) 
colnames(grid_set) = c("Age", "EstimatedSalary")
y_grid = predict(classifier, newdata = grid_set) 
plot(set[, -3],
     main = "Linear SVM Classifier (Test Set)",
     xlab = "Age", ylab = "Estimated Salary", 
     xlim = range(X1), ylim = range(X2)) 
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = ".", col = ifelse(y_grid == 1, "yellow2", "royalblue4")) 
points(set, pch = 21, bg = ifelse(set[,3] == 1, "orange", "navyblue")) 


#Fitting Kernel SVM Classifier to the Training Set 
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set, 
                 type = 'C-classification', 
                 kernel = 'radial')

#Predict the test set results 
y_pred = predict(classifier, newdata = test_set[-3]) 

#Making the Confusion Matrix - to evaluate how many of our pred is true 
cm = table(test_set[,3], y_pred)
#cm in console 

#Installed the ElemStatLearn package if encountered with any issue
#Visualising the Training set 
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
#Building the grid using imaginary pixels
X1 = seq(min(set[,1]) - 1, max(set[,1]) + 1, by = 0.01) 
X2 = seq(min(set[,2]) - 1, max(set[,2]) + 1, by = 0.01) 
grid_set = expand.grid(X1, X2) 
colnames(grid_set) = c("Age", "EstimatedSalary")
y_grid = predict(classifier, newdata = grid_set) 
plot(set[, -3],
     main = "RBF Kernel SVM Classifier (Training Set)",
     xlab = "Age", ylab = "Estimated Salary", 
     xlim = range(X1), ylim = range(X2)) 
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = ".", col = ifelse(y_grid == 1, "yellow2", "royalblue4")) 
points(set, pch = 21, bg = ifelse(set[,3] == 1, "orange", "navyblue")) 

#Visualising the Test set 
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[,1]) - 1, max(set[,1]) + 1, by = 0.01) 
X2 = seq(min(set[,2]) - 1, max(set[,2]) + 1, by = 0.01) 
grid_set = expand.grid(X1, X2) 
colnames(grid_set) = c("Age", "EstimatedSalary")
y_grid = predict(classifier, newdata = grid_set) 
plot(set[, -3],
     main = "RBF Kernel SVM Classifier (Test Set)",
     xlab = "Age", ylab = "Estimated Salary", 
     xlim = range(X1), ylim = range(X2)) 
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = ".", col = ifelse(y_grid == 1, "yellow2", "royalblue4")) 
points(set, pch = 21, bg = ifelse(set[,3] == 1, "orange", "navyblue")) 

