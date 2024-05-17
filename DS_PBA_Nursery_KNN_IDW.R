#title: DS PBA3 K-NN Classifier for Original train data set and test data set, mode 10,40,70 and nb 10, 40, and 70 

#Output: HTML 

#Date: 2024-05-14

#Original training and testing data set

#Installing Packages 

# install.packages("e1071")
# install.packages("caTools")
# install.packages("class")
# install.packages("tidyverse")
# install.packages("tidymodels")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("dials")


#Loading packages
library(e1071)
library(caTools)
library(class)
library(tidyverse)
library(tidymodels)
library(rpart)
library(rpart.plot)
library(dials)

#Importing and or Loading the data sets base training and base testing set across all datasets created for mode and nb
#First step we start with the baseline which is the original training data set
getwd()
Nursery_data <- read_csv("original_training_set_no_na.csv")
Nursery_test <- read_csv("original_testing_set_no_na.csv")
summary(Nursery_data)
summary(Nursery_test)

str(Nursery_data)
str(Nursery_test)

# Remove the x column for the Nursery data and Nursery test

Nursery_data.subset <- Nursery_data[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                      'class')]
head(Nursery_data.subset)

Nursery_test.subset <- Nursery_test[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                      'class')]
head(Nursery_test.subset)

#Data cleaning process- Converting the character data as factor

Nursery_data.subset$parents=factor(Nursery_data.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
Nursery_data.subset$has_nurs=factor(Nursery_data.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
Nursery_data.subset$form=factor(Nursery_data.subset$form,levels = c("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
Nursery_data.subset$children<- as.numeric(sub('more',4,Nursery_data.subset$children))
Nursery_data.subset$housing=factor(Nursery_data.subset$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
Nursery_data.subset$finance=factor(Nursery_data.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
Nursery_data.subset$social=factor(Nursery_data.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
Nursery_data.subset$health=factor(Nursery_data.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
Nursery_data.subset$class=factor(Nursery_data.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"),labels=c(1,2,3,4,5))

Nursery_test.subset$parents=factor(Nursery_test.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
Nursery_test.subset$has_nurs=factor(Nursery_test.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
Nursery_test.subset$form=factor(Nursery_test.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
Nursery_test.subset$children<- as.numeric(sub('more',4,Nursery_test.subset$children))
Nursery_test.subset$housing=factor(Nursery_test.subset$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
Nursery_test.subset$finance=factor(Nursery_test.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
Nursery_test.subset$social=factor(Nursery_test.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
Nursery_test.subset$health=factor(Nursery_test.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
Nursery_test.subset$class=factor(Nursery_test.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))

summary(Nursery_data.subset)
summary(Nursery_test.subset)

#converting the above factor data into numerical except the target "class" for Nursery data and testing data 

# Nursery_data.subset$parents<- as.numeric(factor (Nursery_data.subset$parents))
# Nursery_data.subset$has_nurs <- as.numeric(factor(Nursery_data.subset$has_nurs))
# Nursery_data.subset$form <- as.numeric(factor (Nursery_data.subset$form))
# Nursery_data.subset$children <- as.numeric(factor (Nursery_data.subset$children))
# Nursery_data.subset$housing <- as.numeric(factor(Nursery_data.subset$housing))
# Nursery_data.subset$finance <- as.numeric(factor(Nursery_data.subset$finance))
# Nursery_data.subset$social <- as.numeric(factor(Nursery_data.subset$social))
# Nursery_data.subset$health <- as.numeric(factor(Nursery_data.subset$health))
# 
# Nursery_test.subset$parents<- as.numeric(factor (Nursery_test.subset$parents))
# Nursery_test.subset$has_nurs <- as.numeric(factor(Nursery_test.subset$has_nurs))
# Nursery_test.subset$form <- as.numeric(factor (Nursery_test.subset$form))
# Nursery_test.subset$children <- as.numeric(factor (Nursery_test.subset$children))
# Nursery_test.subset$housing <- as.numeric(factor(Nursery_test.subset$housing))
# Nursery_test.subset$finance <- as.numeric(factor(Nursery_test.subset$finance))
# Nursery_test.subset$social <- as.numeric(factor(Nursery_test.subset$social))
# Nursery_test.subset$health <- as.numeric(factor(Nursery_test.subset$health))

# str(Nursery_data.subset)
# summary(Nursery_data.subset)
# 
# View(Nursery_data.subset)
# dim(Nursery_data.subset)
# 
# View(Nursery_test.subset)
# dim(Nursery_test.subset)

# #Handling of missing values, which have been identified in the training target variable
# 
# Nursery_data.subset$class[is.na(Nursery_data.subset$class)] <- "3"
# 
# summary(Nursery_data.subset)
# summary(Nursery_test.subset)

#Our data set has already been Split as Nursery data and Nursery test 

Nursery_data.subset_train <- Nursery_data.subset[1:9072, ]
Nursery_test.subset_test <- Nursery_test.subset[1:3888, ]

# Nursery_data.subset_train_target <- Nursery_data.subset[1:9072, 8]
# Nursery_test.subset_test_target <- Nursery_test.subset[1:3888, 8]
# require(class)

#Determine the most appropriate K value by square root of all all the observations
# library(caret)
# opt_k = sqrt(9072)

library(caret)

print('starting')
print(Sys.time())
knnModel <- train(
  class ~ ., 
  data = Nursery_data.subset_train, 
  method = "knn", 
  trControl = trainControl(method = "cv"), 
  tuneGrid = data.frame(k = c(2:95))
)
print('end')
print(Sys.time())

opt_k = knnModel$bestTune$k


#Algorithm as m1 for knn

m1 <- knn(train= Nursery_data.subset_train, test= Nursery_test.subset_test, cl= Nursery_data.subset_train$class, k= opt_k)
m1

#Make predictions on Nursery training and Nursery testing data 

train_predictions <- knn(train= Nursery_data.subset_train, test= Nursery_data.subset_train, cl= Nursery_data.subset_train$class, k=opt_k)
test_predictions <- knn(train= Nursery_data.subset_train, test= Nursery_test.subset_test, cl= Nursery_data.subset_train$class, k=opt_k)

#Calculate evaluation metrics for Nursery training data  

train_confusion <-confusionMatrix(train_predictions, Nursery_data.subset_train$class)
train_base_accuracy <-train_confusion$overall["Accuracy"]
train_base_sensitivity <-train_confusion$byClass[, "Sensitivity"]
train_base_recall <- train_confusion$byClass[, "Recall"]
train_base_f1 <- train_confusion$byClass[, "F1"]

#Calculate evaluation metrics for testing data

test_confusion <-confusionMatrix(test_predictions, Nursery_test.subset_test$class)
test_base_accuracy <- test_confusion$overall["Accuracy"]
test_base_sensitivity <- test_confusion$byClass[, "Sensitivity"]
test_base_recall <- test_confusion$byClass[, "Recall"]
test_base_f1 <- test_confusion$byClass[, "F1"]

# Confusion Matrix 
table(Nursery_test.subset_test$class, m1)

#End for Original training and testing data set


#Start Mode 10, 40, 70

#Importing and or loading the data set mode at 10

mode_data <- read_csv("mode_imputation_10.csv")
summary(mode_data)
summary(Nursery_test)

# Remove the x column for the mode data and Nursery test

mode_data.subset <- mode_data[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                      'class')]
head(mode_data.subset)

Nursery_test.subset <- Nursery_test[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                      'class')]
head(Nursery_test.subset)


#Data cleaning process- Converting the character data as factor

mode_data.subset$parents=factor(mode_data.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
mode_data.subset$has_nurs=factor(mode_data.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
mode_data.subset$form=factor(mode_data.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
mode_data.subset$children<- as.numeric(sub('more',4,mode_data.subset$children))
mode_data.subset$housing=factor(mode_data$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
mode_data.subset$finance=factor(mode_data.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
mode_data.subset$social=factor(mode_data.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
mode_data.subset$health=factor(mode_data.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
mode_data.subset$class=factor(mode_data.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))

Nursery_test.subset$parents=factor(Nursery_test.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
Nursery_test.subset$has_nurs=factor(Nursery_test.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
Nursery_test.subset$form=factor(Nursery_test.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
Nursery_test.subset$children<- as.numeric(sub('more',4,Nursery_test.subset$children))
Nursery_test.subset$housing=factor(Nursery_test.subset$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
Nursery_test.subset$finance=factor(Nursery_test.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
Nursery_test.subset$social=factor(Nursery_test.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
Nursery_test.subset$health=factor(Nursery_test.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
Nursery_test.subset$class=factor(Nursery_test.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))


#converting the above factor data into numerical except the target "class" for mode 10 data and testing data 

#mode_data.subset$parents<- as.numeric(factor (mode_data.subset$parents))
#mode_data.subset.subset$has_nurs <- as.numeric(factor(mode_data.subset$has_nurs))
#mode_data.subset$form <- as.numeric(factor (mode_data.subset$form))
#mode_data.subset$children <- as.numeric(factor (mode_data.subset$children))
#mode_data.subset$housing <- as.numeric(factor(mode_data.subset$housing))
#mode_data.subset$finance <- as.numeric(factor(mode_data.subset$finance))
#mode_data.subset$social <- as.numeric(factor(mode_data.subset$social))
#mode_data.subset$health <- as.numeric(factor(mode_data.subset$health))

#Nursery_test.subset$parents<- as.numeric(factor (Nursery_test.subset$parents))
#Nursery_test.subset$has_nurs <- as.numeric(factor(Nursery_test.subset$has_nurs))
#Nursery_test.subset$form <- as.numeric(factor (Nursery_test.subset$form))
#Nursery_test.subset$children <- as.numeric(factor (Nursery_test.subset$children))
#Nursery_test.subset$housing <- as.numeric(factor(Nursery_test.subset$housing))
#Nursery_test.subset$finance <- as.numeric(factor(Nursery_test.subset$finance))
#Nursery_test.subset$social <- as.numeric(factor(Nursery_test.subset$social))
#Nursery_test.subset$health <- as.numeric(factor(Nursery_test.subset$health))

#str(mode_data.subset)
#summary(mode_data.subset)

#View(mode_data.subset)
#dim(mode_data.subset)

#View(Nursery_test.subset)
#dim(Nursery_test.subset)

#Handling of missing values, which have been identified in the training target variable

#mode_data.subset$class[is.na(mode_data.subset$class)] <- "3"

#summary(mode_data.subset)
#summary(Nursery_test.subset)

#Our data set has already been Split as mode 10 data and Nursery test 

mode_data.subset_train <- mode_data.subset[1:9072, ]
Nursery_test.subset_test <- Nursery_test.subset[1:3888, ]

#mode_data.subset_train_target <- mode_data.subset[1:9072, 9]
#Nursery_test.subset_test_target <- Nursery_test.subset[1:3888, 9]
#require(class)

#Determine the most appropriate K value by square root of all all the observations
#sqrt(9072)

#Algorithm as m2 for knn mode 10

m2 <- knn(train= mode_data.subset_train, test= Nursery_test.subset_test, cl= mode_data.subset_train$class, k= opt_k)
m2

#Make predictions on mode 10 and Nursery testing data

train_predictions_mode <- knn(train= mode_data.subset_train, test= Nursery_test.subset_test, cl= mode_data.subset_train$class, k= opt_k)
test_predictions_mode <- knn(train= Nursery_data.subset_train, test= Nursery_test.subset_test, cl= Nursery_data.subset_train$class, k=opt_k)

#Calculate evaluation metrics for mode 10 training data  

train_confusion_mode <-confusionMatrix(train_predictions_mode, mode_data.subset_train$class)
train_base_accuracy_mode <-train_confusion$overall["Accuracy"]
train_base_sensitivity_mode <-train_confusion$byClass[, "Sensitivity"]
train_base_recall_mode <- train_confusion$byClass[, "Recall"]
train_base_f1_mode <- train_confusion$byClass[, "F1"]

#Calculate evaluation metrics for testing data

test_confusion_mode <-confusionMatrix(test_predictions_mode, Nursery_test.subset_test$class)
test_base_accuracy_mode <- test_confusion$overall["Accuracy"]
test_base_sensitivity_mode <- test_confusion$byClass[, "Sensitivity"]
test_base_recall_mode <- test_confusion$byClass[, "Recall"]
test_base_f1_mode <- test_confusion$byClass[, "F1"]

#Making the Confusion Matrix 

table(Nursery_test.subset_test$class, m2)

#End for mode 10


#Start for mode 40 

#Importing and or loading the data set mode at 40

mode2_data <- read_csv("mode_imputation_40.csv")
summary(mode2_data)
summary(Nursery_test)

# Remove the x column for the mode data and Nursery test

mode2_data.subset <- mode2_data[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                'class')]
head(mode2_data.subset)


Nursery_test.subset <- Nursery_test[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                      'class')]
head(Nursery_test.subset)


#Data cleaning process- Converting the character data as factor

mode2_data.subset$parents=factor(mode2_data.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
mode2_data.subset$has_nurs=factor(mode2_data.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
mode2_data.subset$form=factor(mode2_data.subset$form, levels = c("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
mode2_data.subset$children<- as.numeric(sub('more',4,mode2_data.subset$children))
mode2_data.subset$housing=factor(mode2_data$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
mode2_data.subset$finance=factor(mode2_data.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
mode2_data.subset$social=factor(mode2_data.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
mode2_data.subset$health=factor(mode2_data.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
mode2_data.subset$class=factor(mode2_data.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))

Nursery_test.subset$parents=factor(Nursery_test.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
Nursery_test.subset$has_nurs=factor(Nursery_test.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
Nursery_test.subset$form=factor(Nursery_test.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
Nursery_test.subset$children<- as.numeric(sub('more',4,Nursery_test.subset$children))
Nursery_test.subset$housing=factor(Nursery_test.subset$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
Nursery_test.subset$finance=factor(Nursery_test.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
Nursery_test.subset$social=factor(Nursery_test.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
Nursery_test.subset$health=factor(Nursery_test.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
Nursery_test.subset$class=factor(Nursery_test.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))


#Converting data from factor to numeric

#mode2_data.subset$parents<- as.numeric(factor (mode2_data.subset$parents))
#mode2_data.subset.subset$has_nurs <- as.numeric(factor(mode2_data.subset$has_nurs))
#mode2_data.subset$form <- as.numeric(factor (mode2_data.subset$form))
#mode2_data.subset$children <- as.numeric(factor (mode2_data.subset$children))
#mode2_data.subset$housing <- as.numeric(factor(mode2_data.subset$housing))
#mode2_data.subset$finance <- as.numeric(factor(mode2_data.subset$finance))
#mode2_data.subset$social <- as.numeric(factor(mode2_data.subset$social))
#mode2_data.subset$health <- as.numeric(factor(mode2_data.subset$health))

#Nursery_test.subset$parents<- as.numeric(factor (Nursery_test.subset$parents))
#Nursery_test.subset$has_nurs <- as.numeric(factor(Nursery_test.subset$has_nurs))
#Nursery_test.subset$form <- as.numeric(factor (Nursery_test.subset$form))
#Nursery_test.subset$children <- as.numeric(factor (Nursery_test.subset$children))
#Nursery_test.subset$housing <- as.numeric(factor(Nursery_test.subset$housing))
#Nursery_test.subset$finance <- as.numeric(factor(Nursery_test.subset$finance))
#Nursery_test.subset$social <- as.numeric(factor(Nursery_test.subset$social))
#Nursery_test.subset$health <- as.numeric(factor(Nursery_test.subset$health))

#str(mode2_data.subset)
#summary(mode2_data.subset)

#View(mode2_data.subset)
#dim(mode2_data.subset)

#View(Nursery_test.subset)
#dim(Nursery_test.subset)

#summary(mode2_data.subset)
#summary(Nursery_test.subset)

#Handling of missing values, which have been identified in the training target variable

#mode2_data.subset$class[is.na(mode2_data.subset$class)] <- "3"
#summary(mode2_data.subset)
#summary(Nursery_test.subset)

#Our data set has already been Split as mode 10 data and Nursery test 

mode2_data.subset_train <- mode2_data.subset[1:9072, ]
Nursery_test.subset_test <- Nursery_test.subset[1:3888, ]

#mode2_data.subset_train_target <- mode2_data.subset[1:9072, 9]
#Nursery_test.subset_test_target <- Nursery_test.subset[1:3888, 9]
#require(class)

#Determine the most appropriate K value by square root of all all the observations
#sqrt(9072)

#Algorithm as m3 for knn

m3 <- knn(train= mode2_data.subset_train, test= Nursery_test.subset_test, cl= mode2_data.subset_train$class, k= opt_k)
m3

#Make predictions on mode 40 and Nursery testing data

train_predictions_mode40 <- knn(train= mode2_data.subset_train, test= Nursery_data.subset_train, cl= mode2_data.subset_train$class, k=opt_k)
test_predictions_mode40 <- knn(train= Nursery_data.subset_train, test= Nursery_test.subset_test, cl= Nursery_data.subset_train$class, k=opt_k)


#Calculate evaluation metrics for mode 40 training data  

train_confusion_mode40 <-confusionMatrix(train_predictions_mode40, mode2_data.subset_train$class)
train_base_accuracy_mode40 <-train_confusion$overall["Accuracy"]
train_base_sensitivity_mode40 <-train_confusion$byClass[, "Sensitivity"]
train_base_recall_mode40 <- train_confusion$byClass[, "Recall"]
train_base_f1_mode40 <- train_confusion$byClass[, "F1"]

#Calculate evaluation metrics for testing data

test_confusion_mode40 <-confusionMatrix(test_predictions_mode40, Nursery_test.subset_test$class)
test_base_accuracy_mode40 <- test_confusion$overall["Accuracy"]
test_base_sensitivity_mode40 <- test_confusion$byClass[, "Sensitivity"]
test_base_recall_mode40 <- test_confusion$byClass[, "Recall"]
test_base_f1_mode40 <- test_confusion$byClass[, "F1"]

#Making the Confusion Matrix 

table(Nursery_test.subset_test$class, m3)

#End for mode 40 


#Start for mode 70 

#Importing and or loading the data set mode at 70

mode3_data <- read_csv("mode_imputation_70.csv")
summary(mode3_data)
summary(Nursery_test)

# Remove the x column for the mode data and Nursery test

mode3_data.subset <- mode3_data[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                  'class')]
head(mode3_data.subset)


Nursery_test.subset <- Nursery_test[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                      'class')]
head(Nursery_test.subset)


#Data cleaning process- Converting the character data as factor

mode3_data.subset$parents=factor(mode3_data.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
mode3_data.subset$has_nurs=factor(mode3_data.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
mode3_data.subset$form=factor(mode3_data.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
mode3_data.subset$children<- as.numeric(sub('more',4,mode3_data.subset$children))
mode3_data.subset$housing=factor(mode3_data$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
mode3_data.subset$finance=factor(mode3_data.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
mode3_data.subset$social=factor(mode3_data.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
mode3_data.subset$health=factor(mode3_data.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
mode3_data.subset$class=factor(mode3_data.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))

Nursery_test.subset$parents=factor(Nursery_test.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
Nursery_test.subset$has_nurs=factor(Nursery_test.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
Nursery_test.subset$form=factor(Nursery_test.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
Nursery_test.subset$children<- as.numeric(sub('more',4,Nursery_test.subset$children))
Nursery_test.subset$housing=factor(Nursery_test.subset$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
Nursery_test.subset$finance=factor(Nursery_test.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
Nursery_test.subset$social=factor(Nursery_test.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
Nursery_test.subset$health=factor(Nursery_test.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
Nursery_test.subset$class=factor(Nursery_test.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))


#converting data into numeric

#mode3_data.subset$parents<- as.numeric(factor (mode3_data.subset$parents))
#mode3_data.subset.subset$has_nurs <- as.numeric(factor(mode3_data.subset$has_nurs))
#mode3_data.subset$form <- as.numeric(factor (mode3_data.subset$form))
#mode3_data.subset$children <- as.numeric(factor (mode3_data.subset$children))
#mode3_data.subset$housing <- as.numeric(factor(mode3_data.subset$housing))
#mode3_data.subset$finance <- as.numeric(factor(mode3_data.subset$finance))
#mode3_data.subset$social <- as.numeric(factor(mode3_data.subset$social))
#mode3_data.subset$health <- as.numeric(factor(mode3_data.subset$health))

#Nursery_test.subset$parents<- as.numeric(factor (Nursery_test.subset$parents))
#Nursery_test.subset$has_nurs <- as.numeric(factor(Nursery_test.subset$has_nurs))
#Nursery_test.subset$form <- as.numeric(factor (Nursery_test.subset$form))
#Nursery_test.subset$children <- as.numeric(factor (Nursery_test.subset$children))
#Nursery_test.subset$housing <- as.numeric(factor(Nursery_test.subset$housing))
#Nursery_test.subset$finance <- as.numeric(factor(Nursery_test.subset$finance))
#Nursery_test.subset$social <- as.numeric(factor(Nursery_test.subset$social))
#Nursery_test.subset$health <- as.numeric(factor(Nursery_test.subset$health))

#str(mode3_data.subset)
#summary(mode3_data.subset)

#View(mode3_data.subset)
#dim(mode3_data.subset)

#View(Nursery_test.subset)
#dim(Nursery_test.subset)

#summary(mode3_data.subset)
#summary(Nursery_test.subset)

#There are no missing values identified for Mode 3

#Our data set has already been Split as mode 10 data and Nursery test 

mode3_data.subset_train <- mode3_data.subset[1:9072, ]
Nursery_test.subset_test <- Nursery_test.subset[1:3888, ]

#mode3_data.subset_train_target <- mode3_data.subset_train[1:9072, 9]
#Nursery_test.subset_test_target <- Nursery_test.subset[1:3888, 9]
#require(class)

#Determine the most appropriate K value by square root of all all the observations
#sqrt(9072)

#Algorithm as m4 for knn

m4 <- knn(train= mode3_data.subset_train, test= Nursery_test.subset_test, cl= mode3_data.subset_train$class, k= opt_k)
m4

#Make predictions on mode 70 and Nursery testing data

train_predictions_mode70 <- knn(train= mode3_data.subset_train, test= Nursery_data.subset_train, cl= mode3_data.subset_train$class, k=opt_k)
test_predictions_mode70 <- knn(train= Nursery_data.subset_train, test= Nursery_test.subset_test, cl= Nursery_data.subset_train$class, k=opt_k)

#Calculate evaluation metrics for mode 70 training data  

train_confusion_mode70 <-confusionMatrix(train_predictions, mode3_data.subset_train$class)
train_base_accuracy_mode70 <-train_confusion$overall["Accuracy"]
train_base_sensitivity_mode70 <-train_confusion$byClass[, "Sensitivity"]
train_base_recall_mode70 <- train_confusion$byClass[, "Recall"]
train_base_f1_mode70 <- train_confusion$byClass[, "F1"]

#Calculate evaluation metrics for testing data

test_confusion_mode70 <-confusionMatrix(test_predictions_mode70, Nursery_test.subset_test$class)
test_base_accuracy_mode70 <- test_confusion$overall["Accuracy"]
test_base_sensitivity_mode70 <- test_confusion$byClass[, "Sensitivity"]
test_base_recall_mode70 <- test_confusion$byClass[, "Recall"]
test_base_f1_mode70 <- test_confusion$byClass[, "F1"]

#Making the Confusion Matrix 

table(Nursery_test.subset_test$class, m4)

#End for mode 70 


#Start nb 10, 40, 70

#Importing and or loading the data set nb at 10

nb_data <- read_csv("nb_imputation_10.csv")
summary(nb_data)
summary(Nursery_test)

# Remove the x column for the mode data and Nursery test

nb_data.subset <- nb_data[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                'class')]
head(nb_data.subset)

Nursery_test.subset <- Nursery_test[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                      'class')]
head(Nursery_test.subset)


#Data cleaning process- Converting the character data as factor

nb_data.subset$parents=factor(nb_data.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
nb_data.subset$has_nurs=factor(nb_data.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
nb_data.subset$form=factor(nb_data.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
nb_data.subset$children<- as.numeric(sub('more',4,nb_data.subset$children))
nb_data.subset$housing=factor(nb_data.subset$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
nb_data.subset$finance=factor(nb_data.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
nb_data.subset$social=factor(nb_data.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
nb_data.subset$health=factor(nb_data.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
nb_data.subset$class=factor(nb_data.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))

Nursery_test.subset$parents=factor(Nursery_test.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
Nursery_test.subset$has_nurs=factor(Nursery_test.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
Nursery_test.subset$form=factor(Nursery_test.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
Nursery_test.subset$children<- as.numeric(sub('more',4,Nursery_test.subset$children))
Nursery_test.subset$housing=factor(Nursery_test.subset$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
Nursery_test.subset$finance=factor(Nursery_test.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
Nursery_test.subset$social=factor(Nursery_test.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
Nursery_test.subset$health=factor(Nursery_test.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
Nursery_test.subset$class=factor(Nursery_test.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))


#Converting data to numeric for nb 10 

#nb_data.subset$parents<- as.numeric(factor (nb_data.subset$parents))
#nb_data.subset.subset$has_nurs <- as.numeric(factor(nb_data.subset$has_nurs))
#nb_data.subset$form <- as.numeric(factor (nb_data.subset$form))
#nb_data.subset$children <- as.numeric(factor (nb_data.subset$children))
#nb_data.subset$housing <- as.numeric(factor(nb_data.subset$housing))
#nb_data.subset$finance <- as.numeric(factor(nb_data.subset$finance))
#nb_data.subset$social <- as.numeric(factor(nb_data.subset$social))
#nb_data.subset$health <- as.numeric(factor(nb_data.subset$health))

#Nursery_test.subset$parents<- as.numeric(factor (Nursery_test.subset$parents))
#Nursery_test.subset$has_nurs <- as.numeric(factor(Nursery_test.subset$has_nurs))
#Nursery_test.subset$form <- as.numeric(factor (Nursery_test.subset$form))
#Nursery_test.subset$children <- as.numeric(factor (Nursery_test.subset$children))
#Nursery_test.subset$housing <- as.numeric(factor(Nursery_test.subset$housing))
#Nursery_test.subset$finance <- as.numeric(factor(Nursery_test.subset$finance))
#Nursery_test.subset$social <- as.numeric(factor(Nursery_test.subset$social))
#Nursery_test.subset$health <- as.numeric(factor(Nursery_test.subset$health))

#str(nb_data.subset)
#summary(nb_data.subset)

#View(nb_data.subset)
#dim(nb_data.subset)

#View(Nursery_test.subset)
#dim(Nursery_test.subset)

#summary(nb_data.subset)
#summary(Nursery_test.subset)

#Handling of missing values, which have been identified in the training target variable

#nb_data.subset$class[is.na(nb_data.subset$class)]<- "3"
#summary(nb_data.subset)
#summary(Nursery_test.subset)

#Our data set has already been Split as nb 10 data and Nursery test 

nb_data.subset_train <- nb_data.subset[1:9072, ]
Nursery_test.subset_test <- Nursery_test.subset[1:3888, ]

#nb_data.subset_train_target <- nb_data.subset_train[1:9072, 9]
#Nursery_test.subset_test_target <- Nursery_test.subset[1:3888, 9]
#require(class)

#Determine the most appropriate K value by square root of all all the observations
#sqrt(9072)

#Algorithm as m5 for knn

m5 <- knn(train= nb_data.subset_train, test= Nursery_test.subset_test, cl= nb_data.subset_train$class, k= opt_k)
m5

#Make predictions on nb 10 and Nursery testing data

train_predictions_nb <- knn(train= nb_data.subset_train, test= Nursery_data.subset_train, cl= nb_data.subset_train$class, k=opt_k)
test_predictions_nb <- knn(train= Nursery_data.subset_train, test= Nursery_test.subset_test, cl= Nursery_data.subset_train$class, k=opt_k)

#Calculate evaluation metrics for mode 70 training data  

train_confusion_nb <-confusionMatrix(train_predictions_nb, nb_data.subset_train$class)
train_base_accuracy_nb <-train_confusion$overall["Accuracy"]
train_base_sensitivity_nb <-train_confusion$byClass[, "Sensitivity"]
train_base_recall_nb <- train_confusion$byClass[, "Recall"]
train_base_f1_nb <- train_confusion$byClass[, "F1"]

#Calculate evaluation metrics for testing data

test_confusion_nb <-confusionMatrix(test_predictions_nb, Nursery_test.subset_test$class)
test_base_accuracy_nb <- test_confusion$overall["Accuracy"]
test_base_sensitivity_nb <- test_confusion$byClass[, "Sensitivity"]
test_base_recall_nb <- test_confusion$byClass[, "Recall"]
test_base_f1_nb <- test_confusion$byClass[, "F1"]

#Making the Confusion Matrix 

table(Nursery_test.subset_test$class, m5)

#End for nb 10 


#Importing and or loading the data set nb 2 at 40

nb2_data <- read_csv("nb_imputation_40.csv")

summary(nb2_data)
summary(Nursery_test)

# Remove the x column for the mode data and Nursery test

nb2_data.subset <- nb2_data[c('parents','has_nurs','form','children','housing','finance','social','health',  
                            'class')]
head(nb2_data.subset)


Nursery_test.subset <- Nursery_test[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                      'class')]
head(Nursery_test.subset)


#Data cleaning process- Converting the character data as factor

nb2_data.subset$parents=factor(nb2_data.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
nb2_data.subset$has_nurs=factor(nb2_data.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
nb2_data.subset$form=factor(nb2_data.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
nb2_data.subset$children<- as.numeric(sub('more',4,nb2_data.subset$children))
nb2_data.subset$housing=factor(nb2_data.subset$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
nb2_data.subset$finance=factor(nb2_data.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
nb2_data.subset$social=factor(nb2_data.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
nb2_data.subset$health=factor(nb2_data.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
nb2_data.subset$class=factor(nb2_data.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))

Nursery_test.subset$parents=factor(Nursery_test.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
Nursery_test.subset$has_nurs=factor(Nursery_test.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
Nursery_test.subset$form=factor(Nursery_test.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
Nursery_test.subset$children<- as.numeric(sub('more',4,Nursery_test.subset$children))
Nursery_test.subset$housing=factor(Nursery_test.subset$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
Nursery_test.subset$finance=factor(Nursery_test.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
Nursery_test.subset$social=factor(Nursery_test.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
Nursery_test.subset$health=factor(Nursery_test.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
Nursery_test.subset$class=factor(Nursery_test.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))

#Converting the data to numeric for nb 40

#nb2_data.subset$parents<- as.numeric(factor (nb2_data.subset$parents))
#nb2_data.subset.subset$has_nurs <- as.numeric(factor(nb2_data.subset$has_nurs))
#nb2_data.subset$form <- as.numeric(factor (nb2_data.subset$form))
#nb2_data.subset$children <- as.numeric(factor (nb2_data.subset$children))
#nb2_data.subset$housing <- as.numeric(factor(nb2_data.subset$housing))
#nb2_data.subset$finance <- as.numeric(factor(nb2_data.subset$finance))
#nb2_data.subset$social <- as.numeric(factor(nb2_data.subset$social))
#nb2_data.subset$health <- as.numeric(factor(nb2_data.subset$health))

#Nursery_test.subset$parents<- as.numeric(factor (Nursery_test.subset$parents))
#Nursery_test.subset$has_nurs <- as.numeric(factor(Nursery_test.subset$has_nurs))
#Nursery_test.subset$form <- as.numeric(factor (Nursery_test.subset$form))
#Nursery_test.subset$children <- as.numeric(factor (Nursery_test.subset$children))
#Nursery_test.subset$housing <- as.numeric(factor(Nursery_test.subset$housing))
#Nursery_test.subset$finance <- as.numeric(factor(Nursery_test.subset$finance))
#Nursery_test.subset$social <- as.numeric(factor(Nursery_test.subset$social))
#Nursery_test.subset$health <- as.numeric(factor(Nursery_test.subset$health))

#str(nb2_data.subset)
#summary(nb2_data.subset)

#View(nb2_data.subset)
#dim(nb2_data.subset)

#View(Nursery_test.subset)
#dim(Nursery_test.subset)

#summary(nb2_data.subset)
#summary(Nursery_test.subset)

#summary(nb2_data.subset)
#Handling missing values identified in the target variable 

#nb2_data.subset$class[is.na(nb2_data.subset$class)] <- "1"
#nb2_data.subset$parents[is.na(nb2_data.subset$parents)] <- "2"
#nb2_data.subset$has_nurs[is.na(nb2_data.subset$has_nurs)] <- "3"
#nb2_data.subset$form[is.na(nb2_data.subset$form)] <- "2"
#nb2_data.subset$housing[is.na(nb2_data.subset$housing)] <- "1"
#nb2_data.subset$finance[is.na(nb2_data.subset$finance)] <- "1"
#nb2_data.subset$social[is.na(nb2_data.subset$social)] <- "4"
#nb2_data.subset$health[is.na(nb2_data.subset$health)] <- "4"
#summary(nb2_data.subset)

#Our data set has already been Split as nb 10 data and Nursery test

nb2_data.subset_train <- nb2_data.subset[1:9072, ]
Nursery_test.subset_test <- Nursery_test.subset[1:3888, ]

#nb2_data.subset_train_target <- nb2_data.subset[1:9072, 9]
#Nursery_test.subset_test_target <- Nursery_test.subset[1:3888, 9]
#require(class)

#Determine the most appropriate K value by square root of all all the observations
#sqrt(9072)

#Algorithm as m6 for knn

m6 <- knn(train= nb2_data.subset_train, test= Nursery_test.subset_test, cl= nb2_data.subset_train$class, k= opt_k)
m6

#Make predictions on nb 40  and Nursery testing data

train_predictions_nb2 <- knn(train= nb2_data.subset_train, test= Nursery_data.subset_train, cl= nb2_data.subset_train$class, k=opt_k)
test_predictions_nb2 <- knn(train= Nursery_data.subset_train, test= Nursery_test.subset_test, cl= Nursery_data.subset_train$class, k=opt_k)

#Calculate evaluation metrics for nb 40training data  

train_confusion_nb2 <-confusionMatrix(train_predictions, nb2_data.subset_train$class)
train_base_accuracy_nb2 <-train_confusion$overall["Accuracy"]
train_base_sensitivity_nb2 <-train_confusion$byClass[, "Sensitivity"]
train_base_recall_nb2 <- train_confusion$byClass[, "Recall"]
train_base_f1_nb2 <- train_confusion$byClass[, "F1"]

#Calculate evaluation metrics for testing data

test_confusion_nb2 <-confusionMatrix(test_predictions_nb2, Nursery_test.subset_test$class)
test_base_accuracy_nb2 <- test_confusion$overall["Accuracy"]
test_base_sensitivity_nb2 <- test_confusion$byClass[, "Sensitivity"]
test_base_recall_nb2 <- test_confusion$byClass[, "Recall"]
test_base_f1_nb2 <- test_confusion$byClass[, "F1"]

#Making the Confusion Matrix 

table(Nursery_test.subset_test$class, m6)

#End for nb 40 


#Start for nb 70 

#Importing and or loading the data set nb 3 at 70

nb3_data <- read_csv("nb_imputation_70.csv")
summary(nb3_data)
summary(Nursery_test)


#Handling the missing values

library(dplyr)
library(stringr)
nb3_data <-  nb3_data %>%
mutate (class = case_when(
  class == 1 ~ "not_recom",
  class == 2 ~ "priority",
  class == 3 ~ "recommend",
  class == 4 ~ "spec_prior",
  class == 5 ~ "very_recom", 
  TRUE ~ as.character(class)
))

nb3_data <- nb3_data %>%
  mutate(class = case_when(class == "recommended" ~ "recommend", TRUE ~ class))


# Remove the x column for the mode data and Nursery test

nb3_data.subset <- nb3_data[c('parents','has_nurs','form','children','housing','finance','social','health',  
                              'class')]


Nursery_test.subset <- Nursery_test[c('parents','has_nurs','form','children','housing','finance','social','health',  
                                      'class')]
head(Nursery_test.subset)


#Data cleaning process- Converting the character data as factor

nb3_data.subset$parents=factor(nb3_data.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
nb3_data.subset$has_nurs=factor(nb3_data.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
nb3_data.subset$form=factor(nb3_data.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
nb3_data.subset$children<- as.numeric(sub('more',4,nb3_data.subset$children))
nb3_data.subset$housing=factor(nb3_data.subset$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
nb3_data.subset$finance=factor(nb3_data.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
nb3_data.subset$social=factor(nb3_data.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
nb3_data.subset$health=factor(nb3_data.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
nb3_data.subset$class=factor(nb3_data.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))

Nursery_test.subset$parents=factor(Nursery_test.subset$parents,levels = c("great_pret", "pretentious", "usual"), labels=c(1,2,3))
Nursery_test.subset$has_nurs=factor(Nursery_test.subset$has_nurs,levels = c("critical", "improper", "less_proper", "proper", "very_crit"), labels=c(1,2,3,4,5))
Nursery_test.subset$form=factor(Nursery_test.subset$form, levels = c ("complete", "completed", "foster", "incomplete"), labels=c(1,2,3,4))
Nursery_test.subset$children<- as.numeric(sub('more',4,Nursery_test.subset$children))
Nursery_test.subset$housing=factor(Nursery_test.subset$housing, levels = c("convenient","critical","less_conv"),labels=c(1,2,3))
Nursery_test.subset$finance=factor(Nursery_test.subset$finance,levels = c("convenient","inconv"),labels=c(1,2))
Nursery_test.subset$social=factor(Nursery_test.subset$social,levels = c("nonprob","problematic","slightly_prob"),labels=c(1,2,3))
Nursery_test.subset$health=factor(Nursery_test.subset$health,levels = c("not_recom","priority","recommended"),labels=c(1,2,3))
Nursery_test.subset$class=factor(Nursery_test.subset$class,levels = c("not_recom", "priority", "recommend", "spec_prior", "very_recom"), labels=c(1,2,3,4,5))


#Converting the factor data to Numeric 

#nb3_data.subset$parents<- as.numeric(factor (nb3_data.subset$parents))
#nb3_data.subset.subset$has_nurs <- as.numeric(factor(nb3_data.subset$has_nurs))
#nb3_data.subset$form <- as.numeric(factor (nb3_data.subset$form))
#nb3_data.subset$children <- as.numeric(factor (nb3_data.subset$children))
#nb3_data.subset$housing <- as.numeric(factor(nb3_data.subset$housing))
#nb3_data.subset$finance <- as.numeric(factor(nb3_data.subset$finance))
#nb3_data.subset$social <- as.numeric(factor(nb_data.subset$social))
#nb3_data.subset$health <- as.numeric(factor(nb3_data.subset$health))

#Nursery_test.subset$parents<- as.numeric(factor (Nursery_test.subset$parents))
#Nursery_test.subset$has_nurs <- as.numeric(factor(Nursery_test.subset$has_nurs))
#Nursery_test.subset$form <- as.numeric(factor (Nursery_test.subset$form))
#Nursery_test.subset$children <- as.numeric(factor (Nursery_test.subset$children))
#Nursery_test.subset$housing <- as.numeric(factor(Nursery_test.subset$housing))
#Nursery_test.subset$finance <- as.numeric(factor(Nursery_test.subset$finance))
#Nursery_test.subset$social <- as.numeric(factor(Nursery_test.subset$social))
#Nursery_test.subset$health <- as.numeric(factor(Nursery_test.subset$health))

#str(nb3_data.subset)
#summary(nb3_data.subset)

#View(nb3_data.subset)
#dim(nb3_data.subset)

#View(Nursery_test.subset)
#dim(Nursery_test.subset)

#summary(nb3_data.subset)
#summary(Nursery_test.subset)


#Our data set has already been Split as nb 10 data and Nursery test

nb3_data.subset_train <- nb3_data.subset[1:9072, ]
Nursery_test.subset_test <- Nursery_test.subset[1:3888, ]

#nb3_data.subset_train_target <- nb3_data.subset[1:9072, 9]
#Nursery_test.subset_test_target <- Nursery_test.subset[1:3888, 9]
#require(class)

#Determine the most appropriate K value by square root of all all the observations
#sqrt(9072)

#Algorithm as m6 for knn

m7 <- knn(train= nb3_data.subset_train, test= Nursery_test.subset_test, cl=nb3_data.subset_train$class, k=opt_k)
m7

#Make predictions on nb 70  and Nursery testing data

train_predictions_nb3 <- knn(train= nb3_data.subset_train, test= Nursery_data.subset_train, cl= nb3_data.subset_train$class, k=opt_k)
test_predictions_nb3 <- knn(train= Nursery_data.subset_train, test= Nursery_test.subset_test, cl= Nursery_data.subset_train$class, k=opt_k)

#Calculate evaluation metrics for mode 70 training data  

train_confusion_nb3 <-confusionMatrix(train_predictions, nb3_data.subset_train$class)
train_base_accuracy_nb3 <-train_confusion$overall["Accuracy"]
train_base_sensitivity_nb3 <-train_confusion$byClass[, "Sensitivity"]
train_base_recall_nb3 <- train_confusion$byClass[, "Recall"]
train_base_f1_nb3 <- train_confusion$byClass[, "F1"]

#Calculate evaluation metrics for testing data

test_confusion_nb3 <-confusionMatrix(test_predictions_nb3, Nursery_test.subset_test$class)
test_base_accuracy_nb3 <- test_confusion$overall["Accuracy"]
test_base_sensitivity_nb3 <- train_confusion$byClass[, "Sensitivity"]
test_base_recall_nb3 <- test_confusion$byclass[, "Recall"]
test_base_f1_nb3 <- test_confusion$byClass[, "F1"]

#Making the Confusion Matrix 

table(Nursery_test.subset_test$class, m7)

#End for nb 70 

