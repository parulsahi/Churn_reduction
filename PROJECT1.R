rm(list=ls(all=T))
##set working directory
setwd("C:/Users/parul/Desktop/Data Science/PROJECT/project1")

##load libraries
#loading multiple packages at once
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest",'imbalance', "unbalanced", "C50", "dummies", 
      "e1071", "Information","MASS", "rpart", "gbm", "ROSE", 'sampling', 'class','e1071','Metrics',
      'DataCombine', 'gplots','inTrees','GGally','purrr','ROCR','tidyr','ggplot2','pROC')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)


## Read the data
train = read.csv("Train_data.csv", header = T, na.strings = c(" ", "", "NA"),stringsAsFactors = FALSE)
test = read.csv("Test_data.csv", header = T, na.strings = c(" ", "", "NA"),stringsAsFactors = FALSE)
train$isTrain=TRUE
test$isTrain=FALSE

##combine train and test data to preprocess data before feeding it to ML algorithms
data1=rbind(train,test)


##**************************DATA EXPLORATION******************************
dim(data1)
str(data1)
data1$international.plan=as.factor(data1$international.plan)
data1$voice.mail.plan=as.factor(data1$voice.mail.plan)
data1$area.code=as.factor(data1$area.code)
data1$Churn=as.factor(data1$Churn)
data1$state=as.factor(data1$state)


#***************************MISSING VALUE ANALYSIS********************************************
#create dataframe with missing percentage
missing_val = data.frame(apply(data1,2,function(x){sum(is.na(x))}))
#convert row names into columns
missing_val$Columns = row.names(missing_val)
row.names(missing_val) = NULL
#Rename the variable conating missing values
names(missing_val)[1] =  "Missing_percentage"
#calculate missing percentage
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(data1)) * 100
missing_val = missing_val[,c(2,1)]
##NO MISSING DATA##




#********************************DATA VISUALIZATION*************************
print("proportion of Churn in each class: 1: negative class,   2: positive class")
prop.table(table(data1$Churn))
#1. target variable: Churn
ggplot(data1,aes(factor(Churn))) +geom_bar(fill = "coral",alpha = 0.7)+labs(y="count",x="Churn") + theme_classic()+ggtitle("Customer Churn")

#2.#effect of area code on churn
ggplot(data1, aes(area.code, Churn))  + geom_bar(stat = "identity", aes(fill = factor(Churn)))

#3.#effect of state on churn
ggplot(data1, aes(state, Churn))  + geom_bar(stat = "identity", aes(fill = factor(Churn)))

#4.#effect of voice mail plan on churn
ggplot(data1, aes(voice.mail.plan, Churn))  + geom_bar(stat = "identity", aes(fill = factor(Churn)))

#5.#effect of international plan on churn
ggplot(data1, aes(international.plan, Churn))  + geom_bar(stat = "identity", aes(fill = factor(Churn)))

#6.#effect of number of service calls on churn
ggplot(data1, aes(number.customer.service.calls, Churn))  + geom_bar(stat = "identity", aes(fill = factor(Churn)))




##convert factor strings to numeric factor##
##Data Manupulation; convert string categories into factor numeric
for(i in 1:ncol(data1)){
  
  if(class(data1[,i]) == 'factor'){
    
    data1[,i] = factor(data1[,i], labels=(1:length(levels(factor(data1[,i])))))
    
  }
}
#**************************************FEATURE SELECTION***********************************
# ## Find correlated independent variables
numeric_index = sapply(data1,is.numeric) #selecting only numeric

numeric_data = data1[,numeric_index]
cnames=colnames(numeric_data)
#visual plot of correlation matrix
ggcorr(data1[cnames],label=TRUE,label_alpha = TRUE)

cormatrix=cor(data1[cnames])
cormatrix[!lower.tri(cormatrix)]=0
#abc.new <- data[,!apply(cormatrix,2,function(x) any(abs(x) > 0.95))]
cor_var=c()
for(i in cnames){
for(j in cnames){
if(abs(cormatrix[j,i])>0.95){
cor_var=append(cor_var,j)
}}}
#remove correlated variables from data
data1=data1[, !colnames(data1) %in% cor_var]

##chi-square test
cat_var=list("state","area.code","internatiional.plan","voice.mail.plan")
factor_index = sapply(data1,is.factor)
factor_data = data1[,factor_index]

for (i in 1:dim(factor_data)[2])
{
print(names(factor_data)[i])
print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}
#drop the categorical variable for which p-value> 0.05
#Null hypo, H0: predictor and target variable are independent
#Reject H0 when p-value <0.05 (alpha value), hence select (drop) those variables for which p-value<0.05 
#Drop phone number as it is an irrelevant variable for churn prediction
drop_var=c("phone.number","area.code")
data1=data1[, !colnames(data1) %in% drop_var]
#drop 'state' as it has too many levels
data1=subset(data1,select=-c(state))

datacopy=data1
data1=datacopy
#******************SOLVING TARGET CLASS IMBALANCE PROBLEM*******************************
##divide data into train and test sets and perform Resampling
#load original data
data1=datacopy


#1. Random Over Sampling
#applied only on train data
library(ROSE)
data1=datacopy
train=subset(data1,isTrain==TRUE)
test=subset(data1,isTrain==FALSE)
table(train$Churn)
train_over=ovun.sample(Churn~. , data=train, method = "over" , N=2850*2)$data
table(train_over$Churn)
#combine to generate complete data
data1=rbind(train_over,test)

#2. Random under Sampling
#applied on whole data
data1=datacopy
table(data1$Churn)
data1=ovun.sample(Churn~. , data=data1, method = "under" , N=707*2)$data
table(data1$Churn)


# 3. Combining under and over sampling
#applied on train data
data1=datacopy
train=subset(data1,isTrain==TRUE)
test=subset(data1,isTrain==FALSE)
table(train$Churn)
train_both=ovun.sample(Churn~. , data=train, method = "over" , p=0.5)$data
data1=rbind(train_both,test)

# 4. Generate synthetic data using SMOTE oversampling
library(unbalanced)
data1=datacopy
train=subset(data1,isTrain==TRUE)
test=subset(data1,isTrain==FALSE)
table(train$Churn)
train_smote=ubBalance(X=train[,!colnames(train)=="Churn"],Y=train$Churn, positive=2, type = "ubSMOTE", verbose=TRUE)
train_smote_balanced=cbind(train_smote$X,train_smote$Y)
colnames(train_smote_balanced)[which(names(train_smote_balanced) == "train_smote$Y")] <- "Churn"
train_smote_balanced$isTrain=TRUE
table(train_smote_balanced$Churn)
data1=rbind(train_smote_balanced,test)
#or use SmoteClassif







#5. Under sampling using TOMEK links
#applied on whole data
data1=datacopy
table(data1$Churn)
#data_tomek=ubBalance(X=data1[,!colnames(data1)=="Churn"], Y=data1$Churn, positive = 2, type="ubTomek", verbose = TRUE)
library(UBL)
tomek=TomekClassif(Churn~., data1, dist = "HEOM", rem = "maj")
class(tomek)
tomek1=as.data.frame(tomek[[1]])
data1=tomek1
table(data1$Churn)








#************************check numeric variable normality******************
#a.account.length
hist(data1$account.length)
#b.number.vmail.messages
hist(data1$number.vmail.messages)
#c.total.day.minutes
hist(data1$total.day.minutes)
#d.total.day.calls
hist(data1$total.day.calls)
#e.total.eve.minutes
hist(data1$total.eve.minutes)
#f.total.eve.calls
hist(data1$total.eve.calls)
#g.total.night.minutes
hist(data1$total.night.minutes)
#h.total.night.calls
hist(data1$total.night.calls)
#i.total.intl.minutes
hist(data1$total.intl.minutes)
#j.total.intl.calls
hist(data1$total.intl.calls)
#k.number.customer.service.calls
hist(data1$number.customer.service.calls)

##################### OR VIEW HISTOGRAMS IN SINGLE PANE#############
data1 %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

#***********************FEATURE SCALING***********************
#apply normalization on data
numeric_index = sapply(data1,is.numeric) #selecting only numeric
numeric_data = data1[,numeric_index]
cnames=colnames(numeric_data)
for(i in cnames){
  print(i)
  data1[,i] = (data1[,i] - min(data1[,i]))/
    (max(data1[,i] - min(data1[,i])))
}

##Apply Classification algorithms 

errorfunction <- function(cm){
  TN=cm$table[1,1]
  FN=cm$table[1,2]
  FP=cm$table[2,1]
  TP=cm$table[2,2]
  FNR=((FN*100)/(FN+TP))
  acc=(((TP+TN)*100)/(TP+TN+FP+FN))
  sens=(TP*100/(TP+FN))
  spec=(TN*100/(TN+FP))
  prec=(TP*100/(TP+FP))
  cat(sprintf("FALSE NEGATIVE RATE  :%.2f %%\nACCURACY             :%.2f %%\nSENSTIVITY           :%.2f %%\nSPECIFICITY          :%.2f %%\nPRECISION            :%.2f %%",FNR,acc,sens,spec,prec))
}

train=subset(data1,isTrain==TRUE)
train=subset(train,select=-(isTrain))
test=subset(data1,isTrain==FALSE)
test=subset(test,select=-(isTrain))

#1.DECISION TREE CLASSIFIER
#Develop Model on training data
DT_model = C5.0(Churn ~., train, trials = 100, rules = TRUE)
#Summary of DT model
summary(DT_model)
#write rules into disk
write(capture.output(summary(DT_model)), "DT_Rules.txt")
#Lets predict for test cases
DT_Predictions = predict(DT_model, test[,!colnames(test)=="Churn"], type = "class")
##Evaluate the performance of classification model
ConfMatrix_DT = table(predictions=DT_Predictions,actual=test$Churn)
cm1=confusionMatrix(ConfMatrix_DT, positive='2')
print("DECISION TREE ERROR METRICS")
errorfunction(cm1)
roc.curve(test$Churn,DT_Predictions)

#2.RANDOM FOREST CLASSIFIER

RF_model = randomForest(Churn ~ ., train, importance = TRUE, ntree = 500)

#Extract rules fromn random forest
#transform rf object to an inTrees' format
treeList = RF2List(RF_model)  
# #Extract rules
exec = extractRules(treeList, train[,!colnames(test)=="Churn"])  # R-executable conditions
# #Visualize some rules
exec[1:2,]
# #Make rules more readable:
readableRules = presentRules(exec, colnames(train))
readableRules[1:2,]
#Predict test data using random forest model
RF_Predictions = predict(RF_model, test[,!colnames(test)=="Churn"])
##Evaluate the performance of classification model
ConfMatrix_RF = table(predictions=RF_Predictions,actual=test$Churn)
cm2=confusionMatrix(ConfMatrix_RF, positive='2')
print("RANDOM FOREST ERROR METRICS")
errorfunction(cm2)
#ROC-AUC
roc.curve(test$Churn,RF_Predictions)

#3.Logistic Regression
logit_model = glm(Churn ~ ., data = train, family = "binomial")
#summary of the model
summary(logit_model)
#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")
#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.3, 2, 1)
##Evaluate the performance of classification model
ConfMatrix_RF = table(predictions=logit_Predictions,actual=test$Churn)
cm3=confusionMatrix(ConfMatrix_RF, positive='2')
print("LOGISTIC REGRESSION ERROR METRICS")
errorfunction(cm3)

#ROC-AUC
roc.curve(test$Churn,logit_Predictions)


#4. k-nearest neighbors Classifier
library(class)
#Predict test data
#enter the number of neighbors
k=13
KNN_Predictions = knn(train[,!colnames(test)=="Churn"], test[,!colnames(test)=="Churn"], train$Churn, k = k)
#Confusion matrix
Conf_matrix = table(KNN_Predictions, test$Churn)
cm4=confusionMatrix(Conf_matrix, positive='2')
sprintf("KNN classifier ERROR METRICS for k= %d",k)
errorfunction(cm4)
roc.curve(test$Churn,KNN_Predictions)

#5. Naive Bayes
#Develop model
NB_model = naiveBayes(Churn ~ ., data = train)
#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,!colnames(test)=="Churn"], type = 'class')
#Look at confusion matrix
Conf_matrix = table(predicted = NB_Predictions, actual = test$Churn)
cm5=confusionMatrix(Conf_matrix, positive='2')
print("NAIVE BAYES ERROR METRICS")
errorfunction(cm5)
roc.curve(test$Churn,NB_Predictions)


