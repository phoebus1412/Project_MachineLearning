---
title: "Predictive Classifiers on Human's Activities"
author: "phoebus.ZOU"
date: "Wednesday, March 18, 2015"
output: html_document
---
Acknowledgement:Data for this report come from 'Human Activity Recognition Project' conducted by GroupWare@LES, Thanks for their permission to use the data.For more information, please visit http://groupware.les.inf.puc-rio.br/har. 

Overview:This report introduces an implementation to predict human's activities based on random forests algorithm. In this report, data were collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Besides, I also included cross validation and error analysis in this report. 

Data Processing

```r
download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
",destfile="training.csv")
download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
",destfile="testing.csv")
training<-read.csv("training.csv")
testing<-read.csv("testing.csv")
```
In order to measure what kind of a activity one is peroforming, we usually consider three dimensional positions and accelerators of some parts of bodies. so I consider implementing our algorithm through a subset of observed variables. Noticed that in some predictors, large observations are NULL and some predictors are distributed in a small range. Then,I will use nearZeroVar() function in caert package to delete such predictors as they are not affecting much.

```r
inTrain<-createDataPartition(y=training$classe,p=0.85,list=FALSE)
trainCV<-training[inTrain,]
testCV<-training[-inTrain,]
zero<-nearZeroVar(training)
trainCln <- trainCV[-zero]
testCln<- testCV[-zero]
testingCln<-testing[-zero]
```
In the remaining predictors, I find out that many observations are empty, this may lead to some problems in classification,therefore,I use preprocess() to impute these missing values.And do a deeper predictor selection.I removed non-numerical predictor as they more or less affect test results.

```r
numericalIndex = which(lapply(trainCln,class) %in% c('numeric') )
modelImpute <- preProcess(trainCln[,numericalIndex], method=c('knnImpute'))
trainAll <- cbind(predict(modelImpute, trainCln[,numericalIndex]),trainCln$classe)
testAll <- cbind(predict(modelImpute, testCln[,numericalIndex]),testCln$classe)
testingFinal <- predict(modelImpute, testingCln[,numericalIndex])
names(trainAll)[ncol(trainAll)] <- 'classe'
names(testAll)[ncol(testAll)] <- 'classe'
```

Modeling
Consider that I have a lot of observations,here I adopt random forest as my training model.

```r
modelrf<-randomForest(classe~.,data=trainAll,ntree=400,mtry=33)
```

Cross Validation
I have prepared a cross validation set when processing raw data.Now I can calculate my model's accuracy.
In-sample accuracy

```r
inAccuracy<-predict(modelrf,trainAll) 
print(confusionMatrix(inAccuracy,trainAll$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4743    0    0    0    0
##          B    0 3228    0    0    0
##          C    0    0 2909    0    0
##          D    0    0    0 2734    0
##          E    0    0    0    0 3066
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2844     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
The in sample accuracy is 100%, which means my model is bias free.
Out-of-sample accuracy

```r
outAccuracy<- predict(modelrf,testAll) 
print(confusionMatrix(outAccuracy,testAll$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 834   5   0   0   0
##          B   2 562   1   0   1
##          C   0   2 507   0   1
##          D   1   0   5 482   1
##          E   0   0   0   0 538
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9935          
##                  95% CI : (0.9899, 0.9961)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9918          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9877   0.9883   1.0000   0.9945
## Specificity            0.9976   0.9983   0.9988   0.9972   1.0000
## Pos Pred Value         0.9940   0.9929   0.9941   0.9857   1.0000
## Neg Pred Value         0.9986   0.9971   0.9975   1.0000   0.9988
## Prevalence             0.2845   0.1934   0.1744   0.1638   0.1839
## Detection Rate         0.2835   0.1910   0.1723   0.1638   0.1829
## Detection Prevalence   0.2852   0.1924   0.1734   0.1662   0.1829
## Balanced Accuracy      0.9970   0.9930   0.9935   0.9986   0.9972
```
The cross validation accuracy is almost 100%, which indicates that my model is generalized enough to handle new data sets.

Prediction on Given Test sets
Result after applied my model on given test data sets is below:

```r
result <- predict(modelrf, testingFinal) 
error<-sum(result!=testing$classe)/20
```

```
## Warning in is.na(e2): is.na() applied to non-(list or vector) of type
## 'NULL'
```

```r
error
```

```
## [1] 0
```
Conclusion
From the above result, error rate is 0 which means my model perfectly fits the test data sets.Above all, my model is able to distinguish what activity is performing based on observations from accelerometers on the belt, forearm, arm, and dumbell.










