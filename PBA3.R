install.packages("tidyverse")
install.packages("ggplot2")
install.packages("caret")
install.packages("caretEnsemble")

original_testing_set_no_na$Outcome <- factor (data$Outcome, levels = c(0,1), labels = c("False", "True"))

str(original_testing_set_no_na)

head(original_testing_set_no_na)

summary(original_testing_set_no_na)

#Building a model
#split data into training and test data sets

indxTrain <- createDataPartition(y=original_testing_set_no_na$finance, p=0.10, list = FALSE)
training <- original_testing_set_no_na[indxTrain,]
testing <- original_testing_set_no_na[-indxTrain,]

#Check dimensions of the split

prop.table(table(original_testing_set_no_na$finance)) * 100

prop.table(table(training$finance)) * 100

prop.table(table(testing$finance)) * 100

#create objects x which holds the predictor variables and y which holds the response variables

x= training [, -10]
y= training$finance

model= train(x,y,"nb", trControl=trainControl(method= "cv", number = 10))
model
warnings()

#Model Evaluation
#Predict testing set

Predict <- predict(model,newdata = testing )


#Get the confusion matrix to see accuracy value and other parameter values

confusionMatrix(Predict, testing$finance )
Reference






