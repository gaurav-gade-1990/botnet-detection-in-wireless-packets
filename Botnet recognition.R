###Support Vector Machines -------------------
## Botnet Recognition ----

setwd("C:/RAssignments/790")
## Step 2: Exploring and preparing the data ----
# read in data and examine structure
Botnet_TrainingData <- read.csv("training_2000flows.csv")
Botnet_ValidationData <- read.csv("testing_2000flows.csv")


install.packages("kernlab")
library(kernlab)
library(caret)
library(mlbench)
library(rpart)  
library(rpart.plot)
library(e1071)

model <- ksvm(BotOrNot ~ ., data = Botnet_TrainingData,
                          kernel = "rbfdot")
summary(model)

predictedY <- predict(model, Botnet_ValidationData)


library(caret)
library(mlbench)
library(rpart)  
library(rpart.plot)
library(e1071)
install.packages("mlbench")
dotplot(predictedY)

summary(predictedY)

table(predictedY,Botnet_ValidationData$BotOrNot)

length(predictedY)
length(Botnet_ValidationData$BotOrNot)
library(caret)

confusionMatrix(predictedY,Botnet_ValidationData$BotOrNot)



table(true=Botnet_ValidationData$BotorNot == 1, predictedY = predictedY)

print(predictedY)


#############################
  
library(rpart)  
library(rpart.plot)

fit <- rpart(BotOrNot ~ Duration + BPS + IOPR + APL,
             method = "class",
             data = Botnet_TrainingData,
             control= rpart.control(minsplit = 1),
             parms=list(split='information'))

summary(fit)

rpart.plot(fit, type=4, extra=1)

predict_DS <- predict(fit, data=Botnet_ValidationData, type="prob")
predict_DS <- predict(fit, data=Botnet_ValidationData, type="class")


confusionMatrix(predict_DS,Botnet_ValidationData$BotOrNot)


###############################
library(e1071)

model_NaiveBayes <- naiveBayes(BotOrNot ~ Duration + BPS + IOPR + APL,
                    Botnet_TrainingData)

results_NaiveBayes <- predict (model,Botnet_ValidationData)

confusionMatrix(results_NaiveBayes,Botnet_ValidationData$BotOrNot)

###################################

model_NaiveBayes_laplace = naiveBayes(BotOrNot ~., Botnet_TrainingData, laplace=.01)

results_NaiveBayes_laplace <- predict (model_NaiveBayes_laplace,Botnet_ValidationData)

confusionMatrix(results_NaiveBayes_laplace,Botnet_ValidationData$BotOrNot)

#################################

control <- trainControl(method="repeatedcv", number=10, repeats=3)

#svm (support vector machine) model
set.seed(7)
modelSVM <- train(BotOrNot~., data=Botnet_TrainingData, method="svmRadial", trControl = control)

predict_SVM <- predict(modelSVM, Botnet_ValidationData)
confusionMatrix(predict_SVM, Botnet_ValidationData$BotOrNot)

#gbm (gradient boosting algorithm) model
set.seed(7)
modelGbm <- train(BotOrNot~., data=Botnet_TrainingData, method="gbm", trControl=control, verbose=FALSE)

predict_Gbm <- predict(modelGbm, Botnet_ValidationData)
confusionMatrix(predict_Gbm, Botnet_ValidationData$BotOrNot)

#naiveBayes
set.seed(7)
modelNB <- train(BotOrNot~., data=Botnet_TrainingData, method="nb", trControl=control)

predict_NB <- predict(modelNB, Botnet_ValidationData)
confusionMatrix(predict_NB, Botnet_ValidationData$BotOrNot)

#decision tree
set.seed(7)
modelDT <- train(BotOrNot~., data=Botnet_TrainingData, method="rpart", trControl=control)

predict_DT <- predict(modelDT, Botnet_ValidationData)
confusionMatrix(predict_DT, Botnet_ValidationData$BotOrNot)

#random forest
set.seed(7)
modelRF <- train(BotOrNot~., data=Botnet_TrainingData, method="rf", trControl=control)

results <- resamples(list(NaiveBayes=modelNB, GradientBoosting=modelGbm, SVM=modelSVM, DecisionTree=modelDT, RandomForest=modelRF))

predict_results <- resamples(list(NaBa = predict_NB, GBMs=predict_Gbm, SVMs = predict_SVM, DTs=predict_DT))

summary(results)

bwplot(results)

dotplot(results)
