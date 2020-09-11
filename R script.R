library(tidyverse)#To create timble 
library(plotROC)#PLot ROC curve
library(devtools)#install CATBOOST package
library(caret)#TRain model and predict
library(Matrix)#Create matrix
library(randomcoloR)#Get different colours
library(ggthemes)#Use themes
library(xgboost)#For XGBoost model
library(ROSE)#Perform undersample using ROSE
library(mlbench)
library(magrittr)
library(ggplot2)#Plot graphs
library(reshape2)
library(pROC)#compute roc
library(catboost)#Catboost model 
library(yardstick)
library(ggcorrplot)#Plot correlation plot
library(corrplot)#PLot correlation plot
#Install catboost
devtools::install_url('https://github.com/catboost/catboost/releases/download/v0.23/catboost-R-Windows-0.23.tgz', INSTALL_opts = c("--no-multiarch"))

#Redaing test and train data
train <- read.csv(unz("santander-customer-transaction-prediction-dataset (1).zip", "train.csv"), header = TRUE,sep = ",") 
test <- read.csv(unz("santander-customer-transaction-prediction-dataset (1).zip", "test.csv"), header = TRUE,sep = ",") 

#Statistical descrptions of train and test sets
str(train)
str(test)

#Check data type of columns train
cat("Number of rows: ",nrow(train),"\n NUmber of Columns: ", ncol(train))
cat("Datatypes of each column in train seat:\n",sapply(train, typeof))

#Check data type of columns test
cat("Number of rows: ",nrow(test),"\n NUmber of Columns: ", ncol(test))
cat("Datatypes of each column in test seat:\n",sapply(test, typeof))

# Check the skewness for each column in the train dataset
skew_values <- apply(train[,3:ncol(train)], 2, skewness)
summary(skew_values)

# Check the kutrosis for each column in the train dataset
kur_values <- apply(train[,3:ncol(train)], 2, kurtosis)
kur_values
summary(kur_values)

# histogram of variable with the highest skew in train set
hist(train[[names(skew_values[skew_values == max(skew_values)])]], main = "Histogram of var_168",xlab = "skewness")
# histogram of variable with the highest kutrosis in train set
hist(train[[names(kur_values[kur_values == max(kur_values)])]], main = "Histogram of var_179",xlab = "kurtosis")

# Check the skewness for each column in the test dataset
skew_values_test <- apply(test[,2:ncol(test)], 2, skewness)
skew_values_test
summary(skew_values_test)

# Check the kurtosis for each column in the test dataset
kur_values_test <- apply(test[,2:ncol(test)], 2, kurtosis)
kur_values_test
summary(kur_values_test)

# histogram of variable with the highest skew in test set
hist(train[[names(skew_values_test[skew_values_test == max(skew_values_test)])]], main = "Histogram of var_168",xlab = "skewness")
# histogram of variable with the highest kutrosis in test set
hist(train[[names(kur_values_test[kur_values_test == max(kur_values_test)])]], main = "Histogram of var_179",xlab = "kurtosis")


#Distribution of the target column in train set
train$target<-as.factor(train$target)
table(train$target)/length(train$target)*100
  
ggplot(train,aes(x=target,fill=as.factor(target)))+theme_bw()+
  geom_bar(stat='count')+
  theme(legend.position='none')

#Distribution of columns wtr target
#PLot distribution of train
melt(train[,2:30]) %>%
  ggplot(aes(x = value, fill=target)) + 
  facet_wrap(~variable,scales = "free_y") + 
  geom_density(kernel='gaussian') +
  theme_few() + 
  labs(x = 'Frequency distribution', y = 'Number of customers') +
  scale_x_continuous(breaks=5) 

#PLot distribution of test
melt(test[,2:30]) %>%
  ggplot(aes(x = value)) + 
  facet_wrap(~variable,scales = "free_y") + 
  geom_density(kernel='gaussian') +
  theme_few() + 
  labs(x = 'Frequency distribution', y = 'Number of customers') +
  scale_x_continuous(breaks=5) 

#Finding the missing values in train data
missing_value_train<-data.frame(missing_value_train=apply(train,2,function(x){sum(is.na(x))}))
missing_value_train<-sum(missing_value_train)
missing_value_train

#Finding the missing values in test data
missing_val_test<-data.frame(missing_val_test=apply(test,2,function(x){sum(is.na(x))}))
missing_val_test<-sum(missing_val_test)
missing_val_test
train<-train[,-1]

#Z score standradizatio train
train[,2:201]<-(scale(train[,2:201]))
summary(train)

#Z score standradizatio test
test[,2:199]<-(scale(test[,2:199]))
summary(test)

#Distribution of columns wtr target after stardaisation
#PLot distribution of train
melt(train[,2:30]) %>%
  ggplot(aes(x = value, fill=target)) + 
  facet_wrap(~variable,scales = "free_y") + 
  geom_density(kernel='gaussian') +
  theme_few() + 
  labs(x = 'Frequency distribution', y = 'Number of customers') +
  scale_x_continuous(breaks=5) 
#PLot distribution of test
melt(test[,2:30]) %>%
  ggplot(aes(x = value)) + 
  facet_wrap(~variable,scales = "free_y") + 
  geom_density(kernel='gaussian') +
  theme_few() + 
  labs(x = 'Frequency distribution', y = 'Number of customers') +
  scale_x_continuous(breaks=5) 

#drop id columns from train and test
train<-train[,-1]
test<-test[,-1]
str(train)

#Correlations in train data
#Find correlation train
train_cor<-train
train_cor$target<-as.numeric(train_cor$target)
train_correlations<-cor(train_cor,use="pairwise.complete.obs")
train_correlations
ggcorrplot(train_correlations)

#ROSE to help balance target
set.seed(123)
rose_data <- ROSE(target~.,data = train,seed =1)$data                         
table(rose_data$target) 

#Plot after balancing
ggplot(rose_data,aes(x=target,fill=as.factor(target)))+theme_bw()+
  geom_bar(stat='count')+
  theme(legend.position='none')

#Build Models
#Partition train to valid from balanced dataset
partition_all <- createDataPartition(rose_data$target, p = 0.7, list = FALSE)
train_set <- rose_data[ partition_all,]
valid_set  <- rose_data[-partition_all,]
dim(train_set)
dim(valid_set)

#Partition train to valid from unbalanced dataset
table(train$target)
partition_unbalanced <- createDataPartition(train$target, p = 0.7, list = FALSE)
train_unbalanced <- train[ partition_unbalanced,]
valid_unbalanced  <- train[-partition_unbalanced,]
dim(train_unbalanced)
dim(valid_unbalanced)


#XBOOST with balanced data using Grid search
#Create matrix
#train set 
train_xgb_rose_values<-train_set[,2:201]
#Create sparse matrix
train_xgb_rose<-xgb.DMatrix(data = as.matrix(train_xgb_rose_values), label = as.matrix(train_set$target))
train_xgb_rose
#Create train labels
train_xgb_rose_lables<-train_set$target
#Model takes categorical and hence change the levels
levels(train_xgb_rose_lables)<-c("no","yes")
train_xgb_rose_lables

#Create matrix
#valid set 
valid_xgb_rose_values<-valid_set[,2:201]
#Create sparse matrix
valid_xgb_rose<-xgb.DMatrix(data = as.matrix(valid_xgb_rose_values), label = valid_set$target)
valid_xgb_rose
#Create valid labels
valid_xgb_rose_lables<-valid_set$target

#set hyper parameters
xgbGrid_rose <- expand.grid(nrounds = 100,  
                            max_depth = c(5, 10, 15, 20),
                            colsample_bytree = seq(0.5, 0.9, length.out = 5),
                            eta = 0.3,
                            gamma=10,
                            min_child_weight = 2,
                            subsample = 0.5)



#cross validation
xgb_trcontrol_rose = trainControl(
  method = "cv",
  number = 3,  
  verboseIter = TRUE,
  returnData = FALSE,
  classProbs = TRUE
)
#Train model
start_time <- Sys.time()
set.seed(100)
xgb_model_rose = train(train_xgb_rose, train_xgb_rose_lables,  
                  trControl = xgb_trcontrol_rose,
                  tuneGrid = xgbGrid_rose,
                  method = "xgbTree",metric = "ROC")
end_time <- Sys.time()
end_time-start_time
#Time taken was 3.169779 hours

#Model prediction 
xgb_model_rose$bestTune
xgbpred_rose <- predict(xgb_model_rose,valid_xgb_rose)
levels(xgbpred_rose)<-c(0,1)
#PLotting auc roc curve and got AUC of 0.729
par(pty='s')
roc_xgb_rose <- roc(valid_xgb_rose_lables, as.numeric(xgbpred_rose),plot = TRUE, 
                legacy = TRUE, col = "red", 
                lwd=5, print.auc=TRUE, 
                print.auc.x=0.3, print.auc.y=0.2)

#Confusion matrix for model with ROSE and got model accuracy of 73%
xgboost_confusion_rose<-confusionMatrix(xgbpred_rose, valid_xgb_rose_lables)

library(ggplot2)
library(scales)
#Plot confussion matrix
ggplot(data = as.data.frame(xgboost_confusion_rose$table),aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
  theme(legend.position = "none") +
  ggtitle(paste("Accuracy", percent_format()(xgboost_confusion_rose$overall[1])))

#Feature importance
# get number of features selected
nrow(varImp(xgb_model_rose, scale = TRUE)$importance)
#200 features selected

#Plot feature importance plot 
m_rose<-varImp(xgb_model_rose)$importance
m_rose<-as.data.frame(m_rose)
m_rose<-rownames_to_column(m_rose)
m_rose<-mutate(m_rose,rowname = forcats::fct_inorder(rowname ))
filtered<-top_n(m_rose,n=20)
bar_colour<-distinctColorPalette(20)
ggplot(data= filtered,aes(x = rowname, y = Overall)) +
  geom_bar(stat="identity", width=.5, fill=bar_colour)+ 
  coord_flip() + xlab(" Row Name")+
  ggtitle("Variable Importance") + 
  theme(plot.title = element_text(hjust = 0.5))

#XBOOST with unbalanced dataset and grid search
#Create matrix
#Create sparse matrix
train_xgb_ub_grid<-xgb.DMatrix(data = as.matrix(train_xgb_values), label = as.matrix(train_unbalanced$target))
train_xgb_ub_grid
#Create train labels
levels(train_xgb_lables)<-c("no","yes")
train_xgb_lables

#Create matrix
#valid set 
#Create sparse matrix
valid_xgb_ub_grid<-xgb.DMatrix(data = as.matrix(valid_xgb_values), label = valid_unbalanced$target)
valid_xgb_ub_grid
#Create valid labels
valid_xgb_lables<-valid_unbalanced$target

#set hyper parameters
xgbGrid_ub <- expand.grid(nrounds = 100,  # this is n_estimators in the python code above
                          max_depth = c(5, 10, 15, 20),
                          colsample_bytree = seq(0.5, 0.9, length.out = 5),
                          eta = 0.3,
                          gamma=10,
                          min_child_weight = 2,
                          subsample = 0.5)

#cross validation
xgb_trcontrol_ub_grid = trainControl(
  method = "cv",
  number = 3,  
  verboseIter = TRUE,
  returnData = FALSE,
  classProbs = TRUE
)
#Train model
start_time <- Sys.time()
set.seed(100)
xgb_model_ub_grid = train(train_xgb_ub_grid, train_xgb_lables,  
                          trControl = xgb_trcontrol_ub_grid,tuneGrid = xgbGrid_ub,
                          method = "xgbTree",metric = "ROC")
end_time <- Sys.time()
end_time-start_time
#Time taken was 2.922677 hours

#Model prediction 
xgb_model_ub_grid$bestTune
xgbpred_ub_grid<- predict(xgb_model_ub_grid,valid_xgb_ub)
levels(xgbpred_ub_grid)<-c(0,1)
#PLotting auc roc curve and got AUC of 0.631
par(pty='s')
roc_xgb_ub_grid<- roc(valid_xgb_lables, as.numeric(xgbpred_ub_grid),plot = TRUE, 
                      legacy = TRUE, col = "red", 
                      lwd=5, print.auc=TRUE, 
                      print.auc.x=0.3, print.auc.y=0.2)

#Confusion matrix for model with ROSE and got model accuracy of 91.4%
xgboost_confusion_ub_grid<-confusionMatrix(xgbpred_ub_grid, valid_xgb_lables)
xgboost_confusion_ub_grid
library(ggplot2)
library(scales)

ggplot(data = as.data.frame(xgboost_confusion_ub_grid$table),aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
  theme(legend.position = "none") +
  ggtitle(paste("XGBoost with imbalanced data and grid search: Accuracy", percent_format()(xgboost_confusion_ub_grid$overall[1])))

#Feature importance
# get number of features selected
nrow(varImp(xgb_model_ub_grid, scale = TRUE)$importance)
#198 features selected

#Plot feature importance plot 
m_ub_grid<-varImp(xgb_model_ub_grid)$importance
m_ub_grid<-as.data.frame(m_ub_grid)
m_ub_grid<-rownames_to_column(m_ub_grid)
m_ub_grid<-mutate(m_ub_grid,rowname = forcats::fct_inorder(rowname ))
filtered_ub_grid<-top_n(m_ub_grid,n=20)
bar_colour<-distinctColorPalette(20)
ggplot(data= filtered_ub_grid,aes(x = rowname, y = Overall)) +
  geom_bar(stat="identity", width=.5, fill=bar_colour)+ 
  coord_flip() + xlab(" Row Name")+
  ggtitle("XGBoost with imbalanced data and grid search Variable Importance") + 
  theme(plot.title = element_text(hjust = 0.5))


#XBOOST with balanced data using Random search
#Create matrix
#train set 
train_xgb_rose_values<-train_set[,2:201]
#Create sparse matrix
train_xgb_rose<-xgb.DMatrix(data = as.matrix(train_xgb_rose_values), label = as.matrix(train_set$target))
train_xgb_rose
#Create train labels
train_xgb_rose_lables<-train_set$target
#Model takes categorical and hence change the levels
levels(train_xgb_rose_lables)<-c("no","yes")
train_xgb_rose_lables

#Create matrix
#valid set 
valid_xgb_rose_values<-valid_set[,2:201]
#Create sparse matrix
valid_xgb_rose<-xgb.DMatrix(data = as.matrix(valid_xgb_rose_values), label = valid_set$target)
valid_xgb_rose
#Create valid labels
valid_xgb_rose_lables<-valid_set$target

#cross validation
xgb_trcontrol_rose_random = trainControl(
  method = "cv",
  number = 3,  
  verboseIter = TRUE,
  returnData = FALSE,
  classProbs = TRUE,
  search = "random"
)
#Train model
start_time <- Sys.time()
set.seed(100)
xgb_model_rose_random = train(train_xgb_rose, train_xgb_rose_lables,  
                              trControl = xgb_trcontrol_rose_random,
                              method = "xgbTree",metric = "ROC")
end_time <- Sys.time()
end_time-start_time
#Time taken was 1.375804 hours

#Model prediction 
xgb_model_rose_random$bestTune
xgbpred_rose_random <- predict(xgb_model_rose_random,valid_xgb_rose)
levels(xgbpred_rose_random)<-c(0,1)
#PLotting auc roc curve and got AUC of 0.641
par(pty='s')
roc_xgb_rose_random <- roc(valid_xgb_rose_lables, as.numeric(xgbpred_rose_random),plot = TRUE, 
                           legacy = TRUE, col = "red", 
                           lwd=5, print.auc=TRUE, 
                           print.auc.x=0.3, print.auc.y=0.2)

#Confusion matrix for model with ROSE and got model accuracy of 92%
xgboost_confusion_rose_random<-confusionMatrix(xgbpred_rose_random, valid_xgb_rose_lables)

library(ggplot2)
library(scales)

ggplot(data = as.data.frame(xgboost_confusion_rose_random$table),aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
  theme(legend.position = "none") +
  ggtitle(paste("XGBoost with balanced data and random search: Accuracy", percent_format()(xgboost_confusion_rose_random$overall[1])))

#Feature importance
# get number of features selected
nrow(varImp(roc_xgb_rose_random, scale = TRUE)$importance)
#200 features selected

#Plot feature importance plot 
m_rose_random<-varImp(xgb_model_rose_random)$importance
m_rose_random<-as.data.frame(m_rose_random)
m_rose_random<-rownames_to_column(m_rose_random)
m_rose_random<-mutate(m_rose_random,rowname = forcats::fct_inorder(rowname ))
filtered_rose_random<-top_n(m_rose_random,n=20)
bar_colour<-distinctColorPalette(20)
ggplot(data= filtered_rose_random,aes(x = rowname, y = Overall)) +
  geom_bar(stat="identity", width=.5, fill=bar_colour)+ 
  coord_flip() + xlab(" Row Name")+
  ggtitle("XGBoost with balanced data and random search Variable Importance") + 
  theme(plot.title = element_text(hjust = 0.5))



#XBOOST with unbalanced dataset and Random search
#Create matrix
#train set 
train_xgb_values<-train_unbalanced[,2:201]
#Create sparse matrix
train_xgb_ub<-xgb.DMatrix(data = as.matrix(train_xgb_values), label = as.matrix(train_unbalanced$target))
train_xgb_ub
#Create train labels
train_xgb_lables<-train_unbalanced$target
#Model takes categorical and hence change the levels
levels(train_xgb_lables)<-c("no","yes")
train_xgb_lables

#Create matrix
#valid set 
valid_xgb_values<-valid_unbalanced[,2:201]
#Create sparse matrix
valid_xgb_ub<-xgb.DMatrix(data = as.matrix(valid_xgb_values), label = valid_unbalanced$target)
valid_xgb_ub
#Create valid labels
valid_xgb_lables<-valid_unbalanced$target

#cross validation
xgb_trcontrol_ub_random = trainControl(
  method = "cv",
  number = 3,  
  verboseIter = TRUE,
  returnData = FALSE,
  classProbs = TRUE,
  search = "random"
)
#Train model
start_time <- Sys.time()
set.seed(100)
xgb_model_ub_random = train(train_xgb_ub, train_xgb_lables,  
                            trControl = xgb_trcontrol_ub_random,
                            method = "xgbTree",metric = "ROC")
end_time <- Sys.time()
end_time-start_time
#Time taken was 1.74246 hours

#Model prediction 
xgb_model_ub_random$bestTune
xgbpred_ub_random <- predict(xgb_model_ub_random,valid_xgb_ub)
levels(xgbpred_ub_random)<-c(0,1)
#PLotting auc roc curve and got AUC of 0.645
par(pty='s')
roc_xgb_ub_random <- roc(valid_xgb_lables, as.numeric(xgbpred_ub_random),plot = TRUE, 
                         legacy = TRUE, col = "red", 
                         lwd=5, print.auc=TRUE, 
                         print.auc.x=0.3, print.auc.y=0.2)

#Confusion matrix for model with ROSE and got model accuracy of 92%
xgboost_confusion_ub_random<-confusionMatrix(xgbpred_ub_random, valid_xgb_lables)

library(ggplot2)
library(scales)

ggplot(data = as.data.frame(xgboost_confusion_ub_random$table),aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
  theme(legend.position = "none") +
  ggtitle(paste("XGBoost with imbalanced data and random search: Accuracy", percent_format()(xgboost_confusion_rose_random$overall[1])))

#Feature importance
# get number of features selected
nrow(varImp(xgb_model_ub_random, scale = TRUE)$importance)
#200 features selected

#Plot feature importance plot 
m_ub_random<-varImp(xgb_model_ub_random)$importance
m_ub_random<-as.data.frame(m_ub_random)
m_ub_random<-rownames_to_column(m_ub_random)
m_ub_random<-mutate(m_ub_random,rowname = forcats::fct_inorder(rowname ))
filtered_ub_random<-top_n(m_ub_random,n=20)
bar_colour<-distinctColorPalette(20)
ggplot(data= filtered_ub_random,aes(x = rowname, y = Overall)) +
  geom_bar(stat="identity", width=.5, fill=bar_colour)+ 
  coord_flip() + xlab(" Row Name")+
  ggtitle("XGBoost with imbalanced data and random search Variable Importance") + 
  theme(plot.title = element_text(hjust = 0.5))


#Castboost with balanced data using grid search

#Seperate train taret and values
cat_train_lables_rose<-train_set[,'target']
cat_train_values_rose<-train_set[,-1]

#Seperate valid target and values
cat_valid_lables_rose<-valid_set[,'target']
cat_valid_values_rose<-valid_set[,-1]

# Create cat Dataset
train_pool_rose <- catboost.load_pool(data = cat_train_values_rose, label = as.numeric(cat_train_lables_rose))
test_pool_rose <- catboost.load_pool(data = cat_valid_values_rose, label = as.numeric(cat_valid_lables_rose))
#Changing lables to yes or no
levels(cat_train_lables_rose) <- c("no", "yes")
#Create coss validation parameters
#cross validation
cat_trcontrol_rose = trainControl(
  method = "cv",
  number = 3,  
  returnData = FALSE,
  verboseIter = TRUE,
  classProbs = TRUE
)

#Create grid search parameters
cat_grid_rose <- expand.grid(depth = c(6,7,8,9,10),
                        iterations = c(550,600),
                        rsm = 0.95,
                        learning_rate = c(0.3,0.4,0.5),
                        border_count = 100,
                        l2_leaf_reg = 0.004)
#Training model
start_time <- Sys.time()
cat_train <- train(cat_train_values_rose,cat_train_lables_rose,
                   method = catboost.caret,
                   logging_level = 'Verbose', preProc = NULL, metric = "ROC",
                   tuneGrid = cat_grid_rose, trControl = cat_trcontrol_rose)
end_time <- Sys.time()
end_time - start_time
#Model took 2.272147 hours to train
cat_train

#Model prediction
cat_train$bestTune
cat_valid_values_rose
catpred_rose <- predict(cat_train,cat_valid_values_rose)
levels(catpred_rose)<-c(0,1)

#PLotting auc roc curve
par(pty='s')
roc_cat_rose <- roc(cat_valid_lables_rose, as.numeric(catpred_rose),plot = TRUE, 
               legacy = TRUE, col = "red", 
               lwd=5, print.auc=TRUE, 
               print.auc.x=0.3, print.auc.y=0.2)
roc_test
levels(catpred) <- c(0, 1)
#AUC of 0.744
#Confusion matrix
catboost_confusion_rose<-confusionMatrix(catpred_rose, cat_valid_lables_rose)
catboost_confusion_rose
library(ggplot2)
library(scales)

ggplot(data = as.data.frame(catboost_confusion_rose$table),aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
  theme(legend.position = "none") +
  ggtitle(paste("Accuracy", percent_format()(catboost_confusion_rose$overall[1])))
#accuracy of 74%

#Feature importance
# get number of features selected
nrow(varImp(cat_train, scale = TRUE)$importance)
#all 200 features used

cat_features_rose<-varImp(cat_train)$importance
cat_features_rose<-as.data.frame(cat_features_rose)
cat_features_rose<-rownames_to_column(cat_features_rose)
cat_features_rose<-mutate(cat_features_rose,rowname = forcats::fct_inorder(rowname ))
filtered_cat<-top_n(cat_features_rose,n=20)
bar_colour<-distinctColorPalette(20)
ggplot(data= filtered_cat,aes(x = rowname, y = Overall)) +
  geom_bar(stat="identity", width=.5, fill=bar_colour)+ 
  coord_flip() + xlab(" Row Name")+
  ggtitle("Variable Importance") + 
  theme(plot.title = element_text(hjust = 0.5))




##CATBOOST with unbalanced data using grid search
#Castboost

train_unbalanced$target<-as.factor(train_unbalanced$target)
#Seperate train taret and values
cat_train_lables<-train_unbalanced[,'target']
cat_train_values<-train_unbalanced[,-1]

#Seperate valid target and values
cat_valid_lables<-valid_unbalanced[,'target']
cat_valid_values<-valid_unbalanced[,-1]

# Create cat Dataset
train_pool_ub <- catboost.load_pool(data = cat_train_values, label = as.numeric(cat_train_lables))
test_pool_ub <- catboost.load_pool(data = cat_valid_values, label = as.numeric(cat_valid_lables))
#Changing lables to yes or no
levels(cat_train_lables) <- c("no", "yes")
#Create coss validation parameters
#cross validation
cat_trcontrol_ub = trainControl(
  method = "cv",
  number = 3,  
  returnData = FALSE,
  verboseIter = TRUE,
  classProbs = TRUE
)

#Create grid search parameters
cat_grid_ud <- expand.grid(depth = c(6,7,8,9,10),
                        iterations = c(550,600),
                        rsm = 0.95,
                        learning_rate = c(0.3,0.4,0.5),
                        border_count = 100,
                        l2_leaf_reg = 0.004)
#Training model
start_time <- Sys.time()
cat_train_ub <- train(cat_train_values,cat_train_lables,
                   method = catboost.caret,
                   logging_level = 'Verbose', preProc = NULL, metric = "ROC",
                   tuneGrid = cat_grid_ud, trControl = cat_trcontrol_ub)
end_time <- Sys.time()
#Model took 1.93766 hours to train
end_time-start_time
cat_train

#Model prediction
cat_train_ub$bestTune

catpred_ub <- predict(cat_train_ub,cat_valid_values)
levels(catpred_ub)<-c(0,1)

#PLotting auc roc curve
par(pty='s')
roc_cat_ub <- roc(cat_valid_lables, as.numeric(catpred_ub),plot = TRUE, 
               legacy = TRUE, col = "red", 
               lwd=5, print.auc=TRUE, 
               print.auc.x=0.3, print.auc.y=0.2)
roc_cat_ub

#AUC of 0.658
#Confusion matrix accuracy 91.5%
catboost_confusion_ub<-confusionMatrix(catpred_ub, cat_valid_lables)
catboost_confusion_ub
library(ggplot2)
library(scales)

ggplot(data = as.data.frame(catboost_confusion_ub$table),aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
  theme(legend.position = "none") +
  ggtitle(paste("CatBoost with grid search: Accuracy", percent_format()(catboost_confusion_ub$overall[1])))

#Feature importance
# get number of features selected
nrow(varImp(cat_train_ub, scale = TRUE)$importance)
#all 200 features selected

cat_features_ub<-varImp(cat_train_ub)$importance
cat_features_ub<-as.data.frame(cat_features_ub)
cat_features_ub<-rownames_to_column(cat_features_ub)
cat_features_ub<-mutate(cat_features_ub,rowname = forcats::fct_inorder(rowname ))
filtered_cat_ub<-top_n(cat_features_ub,n=20)
bar_colour<-distinctColorPalette(20)
ggplot(data= filtered_cat_ub,aes(x = rowname, y = Overall)) +
  geom_bar(stat="identity", width=.5, fill=bar_colour)+ 
  coord_flip() + xlab(" Row Name")+
  theme(plot.title = element_text(hjust = 0.5))+
  ggtitle("CatBoost Variable Importance with Grid Search")

#Castboost with balanced data using random search

#Seperate train taret and values
cat_train_lables_rose<-train_set[,'target']
cat_train_values_rose<-train_set[,-1]

#Seperate valid target and values
cat_valid_lables_rose<-valid_set[,'target']
cat_valid_values_rose<-valid_set[,-1]

#Changing lables to yes or no
levels(cat_train_lables_rose) <- c("no", "yes")
#Create coss validation parameters
#cross validation
cat_trcontrol_rose_random = trainControl(
  method = "cv",
  number = 3,  
  returnData = FALSE,
  verboseIter = TRUE,
  classProbs = TRUE,
  search = "random"
)


#Training model
start_time <- Sys.time()
set.seed(123)
cat_train_rose_random <- train(cat_train_values_rose,cat_train_lables_rose,
                               method = catboost.caret,
                               logging_level = 'Verbose', preProc = NULL, metric = "ROC",
                               trControl = cat_trcontrol_rose_random)
end_time <- Sys.time()
end_time - start_time
#Model took 11.23678 mins to train

#Model prediction
cat_train_rose_random$bestTune
cat_valid_values_rose
catpred_rose_random <- predict(cat_train_rose_random,cat_valid_values_rose)
levels(catpred_rose_random)<-c(0,1)

#PLotting auc roc curve
par(pty='s')
roc_cat_rose_random <- roc(cat_valid_lables_rose, as.numeric(catpred_rose_random),plot = TRUE, 
                           legacy = TRUE, col = "red", 
                           lwd=5, print.auc=TRUE, 
                           print.auc.x=0.3, print.auc.y=0.2)
roc_test
levels(catpred_rose_random) <- c(0, 1)
#AUC of 0.671
#Confusion matrix
catboost_confusion_rose_random<-confusionMatrix(catpred_rose_random, cat_valid_lables_rose)
catboost_confusion_rose_random
library(ggplot2)
library(scales)

ggplot(data = as.data.frame(catboost_confusion_rose_random$table),aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
  theme(legend.position = "none") +
  ggtitle(paste("CatBoost with balanced random search :Accuracy", percent_format()(catboost_confusion_rose_random$overall[1])))
#accuracy of 91.82%

#Feature importance
# get number of features selected
nrow(varImp(cat_train_rose_random, scale = TRUE)$importance)
#all 200 features used

cat_features_rose_random<-varImp(cat_train_rose_random)$importance
cat_features_rose_random<-as.data.frame(cat_features_rose_random)
cat_features_rose_random<-rownames_to_column(cat_features_rose_random)
cat_features_rose_random<-mutate(cat_features_rose_random,rowname = forcats::fct_inorder(rowname ))
filtered_cat_rose_random<-top_n(cat_features_rose_random,n=20)
bar_colour<-distinctColorPalette(20)
ggplot(data= filtered_cat_rose_random,aes(x = rowname, y = Overall)) +
  geom_bar(stat="identity", width=.5, fill=bar_colour)+ 
  coord_flip() + xlab(" Row Name")+
  ggtitle("CatBoost with balanced random search Variable Importance") + 
  theme(plot.title = element_text(hjust = 0.5))


#Catboost using unbalanced data and random search
#cross validation
cat_trcontrol_random_ub = trainControl(
  method = "cv",
  number = 3,  
  returnData = FALSE,
  verboseIter = TRUE,
  classProbs = TRUE,
  search = "random"
)
#train model
start_time <- Sys.time()
set.seed(100)
cat_train_ub_random <- train(cat_train_values,cat_train_lables,
                      method = catboost.caret,
                      logging_level = 'Verbose', preProc = NULL, metric = "ROC",
                      trControl = cat_trcontrol_ub)
end_time <- Sys.time()
end_time-start_time
#time taken 15.98863 mins

#Model prediction
cat_train_ub_random$bestTune

catpred_ub_random <- predict(cat_train_ub_random,cat_valid_values)
levels(catpred_ub_random)<-c(0,1)

#PLotting auc roc curve
par(pty='s')
roc_cat_ub_random <- roc(cat_valid_lables, as.numeric(catpred_ub_random),plot = TRUE, 
                  legacy = TRUE, col = "red", 
                  lwd=5, print.auc=TRUE, 
                  print.auc.x=0.3, print.auc.y=0.2)
roc_cat_ub

#AUC of 0.658
#Confusion matrix accuracy 92
catboost_confusion_ub_random<-confusionMatrix(catpred_ub_random, cat_valid_lables)
catboost_confusion_ub_random
library(ggplot2)
library(scales)

ggplot(data = as.data.frame(catboost_confusion_ub_random$table),aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
  theme(legend.position = "none") +
  ggtitle(paste("CatBoost with random search: Accuracy", percent_format()(catboost_confusion_ub_random$overall[1])))

#Feature importance
# get number of features selected
nrow(varImp(cat_train_ub_random, scale = TRUE)$importance)
#all 200 features selected

cat_features_ub_random<-varImp(cat_train_ub_random)$importance
cat_features_ub_random<-as.data.frame(cat_features_ub_random)
cat_features_ub_random<-rownames_to_column(cat_features_ub_random)
cat_features_ub_random<-mutate(cat_features_ub_random,rowname = forcats::fct_inorder(rowname ))
filtered_cat_ub_random<-top_n(cat_features_ub_random,n=20)
bar_colour<-distinctColorPalette(20)
ggplot(data= filtered_cat_ub_random,aes(x = rowname, y = Overall)) +
  geom_bar(stat="identity", width=.5, fill=bar_colour)+ 
  coord_flip() + xlab(" Row Name")+
  ggtitle("CatBoost Variable Importance with Random Search") + 
  theme(plot.title = element_text(hjust = 0.5))


