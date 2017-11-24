rm(list=ls())
library(class)
library(unbalanced)
library(party)
library(caret)
library(e1071)
library(randomForest)
library(nnet)
library(FSelector)
library(deepnet)
# setwd("")

normalize = function(x){
  return((x-min(x)) / (max(x)-min(x)))
}

writeCsv = function(predi, f = "Prueba.csv"){
  if (is.numeric(predi)==FALSE) {predi = as.numeric(levels(predi))[predi]}
  write.csv(predi, file = f, quote = FALSE)
  ln = readLines(paste(f,sep = ""),-1)
  ln[1]="Id,Prediction"
  writeLines(ln,paste(f,sep = ""))
}

## load the database dataset
database = read.table("/home/manuelmontero/Escritorio/R Projects/KAGGLE, No balanceado/train.csv", sep=",", header = TRUE)
databaseTest = read.table("/home/manuelmontero/Escritorio/R Projects/KAGGLE, No balanceado/test.csv", sep=",", header = TRUE)
databaseTest[,1] = NULL

plot(database[,23], col=database[,23]+1, xlab = "X", ylab = "Y", pch = 16)

# d = rbind(database[1:dim(database)[2]-1],databaseTest)
# 
# for (i in 1:dim(d)[2]){
#   d[,i] = normalize(d[,i])
# }
# 
# database = cbind(d[1:dim(database)[1],],database[,dim(database)[2]])
# colnames(database)[23] = "PV1MATH"
# databaseTest = d[(dim(database)[1]+1):dim(d)[1],]
# rownames(databaseTest) = 1:3200

############################################### CV ###############################################

database[,23] = (database[,23]-1) * -1
unique(database$PV1MATH)
nClass0 = sum(database$PV1MATH == 0) # minority
nClass1 = sum(database$PV1MATH == 1) # mayority
IR = nClass1 / nClass0

# Set up the dataset for 5 fold cross validation.
# Make sure to respect the PV1MATH imbalance in the folds.
pos = (1:dim(database)[1])[database$PV1MATH==0] # minority
neg = (1:dim(database)[1])[database$PV1MATH==1] # mayority

CVperm_pos = matrix(sample(pos,length(pos)), ncol=10, byrow=T)
CVperm_neg = matrix(sample(neg,length(neg)), ncol=10, byrow=T)

CVperm = rbind(CVperm_pos, CVperm_neg)

###################################### Base performance of 40NN ####################################

knn.pred = NULL
for( i in 1:10){
  predictions = knn(database[-CVperm[,i], -23], database[CVperm[,i], -23], database[-CVperm[,i], 23], k = 40)
  knn.pred = c(knn.pred, predictions)
}
acc = sum((database$PV1MATH[as.vector(CVperm)] == 0 & knn.pred == 1) 
          | (database$PV1MATH[as.vector(CVperm)] == 1 & knn.pred == 2)) / (nClass0 + nClass1)
tpr = sum(database$PV1MATH[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr = sum(database$PV1MATH[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean = sqrt(tpr * tnr)
print("3nn database")
print(gmean)

predi = knn(database[-CVperm[,1], -23], databaseTest, database[-CVperm[,i], 23], k = 40)
writeCsv(predi, "Prueba.csv")

###################################### 40NN + ROS ####################################

knn.pred = NULL
for( i in 1:5){
  
  train = database[-CVperm[,i], -23]
  classes.train = database[-CVperm[,i], 23] 
  test  = database[CVperm[,i], -23]
  
  # randomly oversample the minority class (class 0)
  minority.indices = (1:dim(train)[1])[classes.train == 0]
  to.add = dim(train)[1] - 2 * length(minority.indices)
  duplicate = sample(minority.indices, to.add, replace = T)
  for( j in 1:length(duplicate)){
    train = rbind(train, train[duplicate[j],])
    classes.train = c(classes.train, 0)
  }  
  
  # use the modified training set to make predictions
  predictions =  knn(train, test, classes.train, k = 40)
  knn.pred = c(knn.pred, predictions)
}

tpr.ROS = sum(database$PV1MATH[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.ROS = sum(database$PV1MATH[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.ROS = sqrt(tpr.ROS * tnr.ROS)
print("ROS database")
print(gmean.ROS)

####

train = database[-CVperm[,1], -23]
classes.train = database[-CVperm[,i], 23] 
test  = database[CVperm[,1], -23]

# randomly oversample the minority class (class 0)
minority.indices = (1:dim(train)[1])[classes.train == 0]
to.add = dim(train)[1] - 2 * length(minority.indices)
duplicate = sample(minority.indices, to.add, replace = T)
for( j in 1:length(duplicate)){
  train = rbind(train, train[duplicate[j],])
  classes.train = c(classes.train, 0)
}  

predi = knn(train, databaseTest, classes.train, k = 40)
database[,23] = (database[,23]-1) * -1
writeCsv(predi, f = "Prueba.csv")

###################################### 40NN + RUS ####################################

database[,23] = (database[,23]-1) * -1
knn.pred = NULL
for( i in 1:5){
  
  train = database[-CVperm[,i], -23]
  classes.train = database[-CVperm[,i], 23] 
  test  = database[CVperm[,i], -23]
  
  # randomly undersample the minority class (class 1)
  majority.indices = (1:dim(train)[1])[classes.train == 1]
  to.remove = 2* length(majority.indices) - dim(train)[1]
  remove = sample(majority.indices, to.remove, replace = F)
  train = train[-remove,] 
  classes.train = classes.train[-remove]
  
  # use the modified training set to make predictions
  predictions =  knn(train, test, classes.train, k = 40)
  knn.pred = c(knn.pred, predictions)
}

tpr.RUS = sum(database$PV1MATH[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.RUS = sum(database$PV1MATH[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.RUS = sqrt(tpr.RUS * tnr.RUS)
print("RUS database")
print(gmean.RUS)

train = database[-CVperm[,i], -23]
classes.train = database[-CVperm[,i], 23] 
test  = database[CVperm[,i], -23]

# randomly undersample the minority class (class 1)
majority.indices = (1:dim(train)[1])[classes.train == 1]
to.remove = 2* length(majority.indices) - dim(train)[1]
remove = sample(majority.indices, to.remove, replace = F)
train = train[-remove,] 
classes.train = classes.train[-remove]

predi = knn(train, databaseTest, classes.train, k = 40)
database[,23] = (database[,23]-1) * -1
writeCsv(predi, f = "Prueba.csv")


###################################### 40NN + SMOTE ####################################

n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

data = ubBalance(X= input, Y=output, type="ubSMOTE", percOver=1200, percUnder=600, verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

predictions =  knn(balancedData[,-23], balancedData[,-23], balancedData[,23], k = 40)
table(predictions==balancedData[,23])

###################################### 40NN + TOMEK Link ####################################

n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

data = ubBalance(X= input, Y=output, type="ubTomek", percOver=100, percUnder=100, verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

predictions =  knn(balancedData[,-23], databaseTest, balancedData[,23], k = 40)
writeCsv(predictions, f = "Prueba.csv")

###################################### Tree + TOMEK Link ####################################

n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

data = ubBalance(X= input, Y=output, type="ubTomek", percOver=100, percUnder=100, verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

positionClass = length(names(balancedData))
varClass = names(balancedData)[positionClass]
classForm = as.formula(paste(varClass,"~.",sep=""))

inTrain = createDataPartition(y=balancedData[,positionClass], p = 0.80, list = FALSE)
training = balancedData[ inTrain,]
testing  = balancedData[-inTrain,]

ctModel = ctree(classForm, training)
testPred = predict(ctModel, newdata = testing)
confusionMatrix(table(testPred, testing[,positionClass]))

predictions = predict(ctModel, newdata = databaseTest)
writeCsv(predictions, f = "Prueba.csv")

###################################### Tree + RUS ####################################

#RUS
train = database[, -23]
classes.train = database[, 23] 
test  = database[, -23]
# randomly undersample the minority class (class 1)
majority.indices = (1:dim(train)[1])[classes.train == 0]
to.remove = 2* length(majority.indices) - dim(train)[1]
remove = sample(majority.indices, to.remove, replace = F)
train = train[-remove,] 
classes.train = classes.train[-remove]

#Partitions
balancedData = cbind(train,classes.train)
colnames(balancedData)[23] = "PV1MATH"
positionClass = length(names(database))
varClass = names(database)[positionClass]
classForm = as.formula(paste(varClass,"~.",sep=""))
inTrain = createDataPartition(y=balancedData[,positionClass], p = 0.80, list = FALSE)
training = balancedData[ inTrain,]
testing  = balancedData[-inTrain,]

ctModel = ctree(classForm, training)
testPred = predict(ctModel, newdata = database[,-23])
confusionMatrix(table(round(testPred), database[,23]))

predictions = predict(ctModel, newdata = databaseTest)
predictions = as.factor(round(predictions))
writeCsv(predictions, f = "Prueba.csv")

###################################### RandomForest + Tomek Links ####################################

n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

data = ubBalance(X= input, Y=output, type="ubTomek", percOver=100, percUnder=100, verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

model = randomForest::randomForest(PV1MATH ~ ., data=balancedData, ntree=4000)
predictions = predict(model, balancedData[,-23])
confusionMatrix(predictions, balancedData[,23])

predictions = predict(model, databaseTest)
writeCsv(predictions, f = "Prueba.csv")

################# RandomForest + Tomek Links + scaled data ####################################

n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

data = ubBalance(X= input, Y=output, type="ubTomek", percOver=100, percUnder=100, verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

balancedData[,-23] = scale(balancedData[,-23])
databaseTest = scale(databaseTest)

model = randomForest::randomForest(PV1MATH ~ ., data=balancedData, ntree=100)
predictions = predict(model, balancedData[,-23])
confusionMatrix(predictions, balancedData[,23])

predictions = predict(model, databaseTest)
writeCsv(predictions, f = "Prueba.csv")

############################# SVM + Tomek Links ####################################

n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

data = ubBalance(X= input, Y=output, type="ubTomek", verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

inTrain = createDataPartition(y=balancedData[,23], p = 0.80, list = FALSE)
training = balancedData[ inTrain,]
testing  = balancedData[-inTrain,]

model = svm(PV1MATH ~., balancedData)
predictions = predict(model, balancedData[,-23])
confusionMatrix(predictions, balancedData[,23])

predictions = predict(model, databaseTest)
writeCsv(predictions, f = "Prueba.csv")

############################# SVM + Tomek Links + scale data ####################################

n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

data = ubBalance(X= input, Y=output, type="ubTomek", percOver=100, percUnder=100, verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

balancedData[,-23] = scale(balancedData[,-23])
databaseTest = scale(databaseTest)

model = svm(PV1MATH ~., balancedData, scale = FALSE)
predictions = predict(model, balancedData[,-23])
confusionMatrix(predictions, balancedData[,23])

predictions = predict(model, databaseTest)
writeCsv(predictions, f = "Prueba.csv")

############################# SVM + Tomek Links + SMOTE ####################################

n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

data = ubBalance(X= input, Y=output, type="ubTomek", percOver=100, percUnder=100, verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

data = ubBalance(X= balancedData[,-23], Y=balancedData[,23], type="ubSMOTE", percOver=200, percUnder=100, verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

inTrain = createDataPartition(y=balancedData[,23], p = 0.80, list = FALSE)
training = balancedData[ inTrain,]
testing  = balancedData[-inTrain,]

model = svm(PV1MATH ~., training, scale = FALSE)
predictions = predict(model, testing[,-23])
confusionMatrix(predictions, testing[,23])

predictions = predict(model, databaseTest)
writeCsv(predictions, f = "Prueba.csv")

############################# SVM + Tomek Links + remove outliers ####################################

for (i in 1:22){
  outlierData = outlier(database[,i])
  a = which(database[,i] == outlierData,TRUE)
  database <- database[-c(a), ]
}

n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

data = ubBalance(X= input, Y=output, type="ubTomek", percOver=100, percUnder=100, verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

inTrain = createDataPartition(y=balancedData[,23], p = 0.90, list = FALSE)
training = balancedData[ inTrain,]
testing  = balancedData[-inTrain,]

model = svm(PV1MATH ~., training, scale = FALSE)
predictions = predict(model, testing[,-23])
confusionMatrix(predictions, testing[,23])

predictions = predict(model, databaseTest)
writeCsv(predictions, f = "Prueba.csv")

################ SVM or RandmForest + Tomek Links + remove outliers + gaussian noise #################

newDatabase = database
for (j in 1:4){
  for (i in 1:dim(database)[1]){
    a = runif(1, -0.001, 0.001)
    a = database[i,-23]+a
    a = c(a,database[i,23])
    names(a) = names(database[1,])
    newDatabase = rbind(newDatabase,a)
  }
}

for (i in 1:22){
  outlierData = outlier(newDatabase[,i])
  a = which(newDatabase[,i] == outlierData,TRUE)
  newDatabase <- newDatabase[-c(a), ]
}

n = ncol(newDatabase)
output = as.factor(newDatabase[,n])
input = newDatabase[ ,-n]

data = ubBalance(X= input, Y=output, type="ubTomek", percOver=100, percUnder=100, verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

inTrain = createDataPartition(y=balancedData[,23], p = 0.90, list = FALSE)
training = balancedData[ inTrain,]
testing  = balancedData[-inTrain,]

# model = svm(PV1MATH ~., training, scale = FALSE)
model = randomForest::randomForest(PV1MATH ~.^8, data=training, ntree=400)
predictions = predict(model, testing[,-23])
confusionMatrix(predictions, testing[,23])

predictions = predict(model, databaseTest)
writeCsv(predictions, f = "Prueba.csv")

################ NN + SMOTE #################

newDatabase = database
# for (j in 1:5){
#   for (i in 1:dim(database)[1]){
#     a = runif(1, -0.01, 0.01)
#     a = database[i,-23]+a
#     a = c(a,database[i,23])
#     names(a) = names(database[1,])
#     newDatabase = rbind(newDatabase,a)
#   }
# }

n = ncol(newDatabase)
output = as.factor(newDatabase[,n])
input = newDatabase[ ,-n]

data = ubBalance(X= input, Y=output, type="ubSMOTE", percOver=100, percUnder=100, verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

balancedData = balancedData[sample(nrow(balancedData)),]
inTrain = createDataPartition(y=balancedData[,23], p = 0.70, list = FALSE)
training = balancedData[ inTrain,]
testing  = balancedData[-inTrain,]

n = names(training)[1:length(training)-1]
f = as.formula(paste("PV1MATH ~", paste(n[!n %in% "medv"], collapse = " + ")))
model = nnet(f, data=training, size=100, maxit = 1000, MaxNWts = 100000)
predictions = predict(model, testing[,-23])
confusionMatrix(round(predictions), testing[,23])

predictions = predict(model, databaseTest)
predictions = round(predictions)
writeCsv(predictions, f = "Prueba.csv")

################ NN #################

weights = FSelector::chi.squared(PV1MATH~.,database)
sub = FSelector::cutoff.k(weights, 10)
f = as.simple.formula(sub, "PV1MATH")

balancedData = database[sample(nrow(database)),]
inTrain = createDataPartition(y=balancedData[,23], p = 0.80, list = FALSE)
training = balancedData[ inTrain,]
testing  = balancedData[-inTrain,]

model = nnet(f, data=training, size=800, rang = 0.1, decay = 5e-4, maxit = 1000, MaxNWts = 100000)
predictions = predict(model, testing[,-23])
confusionMatrix(round(predictions), testing[,23])

predictions = predict(model, databaseTest)
predictions = round(predictions)
print(length(predictions[predictions == 0])/1600)
writeCsv(predictions, f = "Prueba.csv")

best = read.table("/home/manuelmontero/Escritorio/R Projects/KAGGLE, No balanceado/MejorPrueba.csv", sep=",", header = TRUE)
print(mean(predictions[,1]==best[,2]))

############################# SVM + Tomek Links ####################################

n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

weights = FSelector::chi.squared(PV1MATH~.,database)
sub = FSelector::cutoff.k(weights, 17)
f = as.simple.formula(sub, "PV1MATH")
print(f)

data = ubBalance(X= input, Y=output, type="ubTomek", verbose=TRUE)
balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))

balancedData = balancedData[sample(nrow(balancedData)),]
inTrain = createDataPartition(y=balancedData[,23], p = 0.50, list = FALSE)
training = balancedData[ inTrain,]
testing  = balancedData[-inTrain,]

model = svm(f, training, gamma=0.004, scale = F)
predictions = predict(model, testing[,-23])
confusionMatrix(predictions, testing[,23])

predictions = predict(model, databaseTest)
writeCsv(predictions, f = "Prueba.csv")

print(length(predictions[predictions == 0])/1600)
best = read.table("/home/manuelmontero/Escritorio/R Projects/KAGGLE, No balanceado/MejorPrueba.csv", sep=",", header = TRUE)
print(mean(predictions[1]==best[,2]))

############################# ####################################

range01 <- function(x){(x-min(x))/(max(x)-min(x))}
library(pROC)
n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

data = ubBalance(X= input, Y=output, type="ubTomek", verbose=TRUE)
data = ubBalance(X= data$X, Y=data$Y, type="ubTomek", verbose=TRUE)

balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))
balancedData = balancedData[sample(nrow(balancedData)),]
inTrain = createDataPartition(y=balancedData[,23], p = 0.80, list = FALSE)
training = balancedData[ inTrain,]
testing  = balancedData[-inTrain,]

weights = FSelector::chi.squared(PV1MATH~.,balancedData)
sub = FSelector::cutoff.k(weights, 12)
f = as.simple.formula(sub, "PV1MATH")

# model = svm(f, training, gamma=0.01, cost = 10,  scale = F, probability=TRUE)
predictions = predict(model, testing[,-23], probability=TRUE)
predictions = range01(attr(predictions,"probabilities")[,1])
auc(testing[,23], as.vector(predictions))
predictions = predict(model, training[,-23], probability=TRUE)
predictions = range01(attr(predictions,"probabilities")[,1])
auc(training[,23], as.vector(predictions))

predictions = predict(model, databaseTest, probability=TRUE)
predictions = range01(attr(predictions,"probabilities")[,1])
writeCsv(predictions, f = "Prueba.csv")


############################# NN deepnet package ####################################

range01 <- function(x){(x-min(x))/(max(x)-min(x))}
library(pROC)
n = ncol(database)
output = as.factor(database[,n])
input = database[ ,-n]

data = ubBalance(X= input, Y=output, type="ubTomek", verbose=TRUE)

weights = FSelector::chi.squared(PV1MATH~.,database)
sub = FSelector::cutoff.k(weights, 2)
f = as.simple.formula(sub, "PV1MATH")

balancedData = cbind(data$X,data$Y)
colnames(balancedData)[23] = "PV1MATH"
cols = c(1, 2, 3, 4, 5, 6, 8)
balancedData[,cols] = apply(balancedData[,cols], 2, function(x) as.integer(x))
balancedData = balancedData[sample(nrow(balancedData)),]
inTrain = createDataPartition(y=balancedData[,23], p = 0.80, list = FALSE)
training = balancedData[ inTrain,]
testing  = balancedData[-inTrain,]

dnn = dbn.dnn.train(as.matrix(training[,-length(names(database))]), as.numeric(training[,length(names(database))])-1,
              hidden = c(500), activationfun = "sigm", learningrate = 1e-6,
              momentum = 1, learningrate_scale = 0.99, output = "linear", numepochs = 5,
              batchsize = 2, hidden_dropout = 0, visible_dropout = 0, cd = 1)

predictions = nn.predict(dnn, testing[,-length(names(database))])
auc(testing[,23], as.vector(predictions))

predictions = nn.predict(dnn, databaseTest)
writeCsv(predictions, f = "Prueba.csv")
