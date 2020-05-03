dev.off() # close plots
rm(list=ls()) # wipe environment
library(DataExplorer)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(gmodels)
library(caret)
library(RWeka)
library(rattle)
library(randomForest)
library(e1071)

mushroom <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"))
# Data preparation
colnames(mushroom) = c("edibility", "cap_shape", "cap_surface", 
                       "cap_color", "bruises", "odor", 
                       "gill_attachement", "gill_spacing", "gill_size", 
                       "gill_color", "stalk_shape", "stalk_root", 
                       "stalk_surface_above_ring", "stalk_surface_below_ring", "stalk_color_above_ring", 
                       "stalk_color_below_ring", "veil_type", "veil_color", 
                       "ring_number", "ring_type", "spore_print_color", 
                       "population", "habitat")


levels(mushroom$edibility) <- c("edible", "poisonous")
levels(mushroom$cap_shape) <- c("bell", "conical", "flat", "knobbed", "sunken", "convex")
levels(mushroom$cap_color) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                                "green", "purple", "white", "yellow")
levels(mushroom$cap_surface) <- c("fibrous", "grooves", "scaly", "smooth")
levels(mushroom$bruises) <- c("no", "yes")
levels(mushroom$odor) <- c("almond", "creosote", "foul", "anise", "musty", "none", "pungent", "spicy", "fishy")
levels(mushroom$gill_attachement) <- c("attached", "free")
levels(mushroom$gill_spacing) <- c("close", "crowded")
levels(mushroom$gill_size) <- c("broad", "narrow")
levels(mushroom$gill_color) <- c("buff", "red", "gray", "chocolate", "black", "brown", "orange", 
                                 "pink", "green", "purple", "white", "yellow")
levels(mushroom$stalk_shape) <- c("enlarging", "tapering")
levels(mushroom$stalk_root) <- c("missing", "bulbous", "club", "equal", "rooted")
levels(mushroom$stalk_surface_above_ring) <- c("fibrous", "silky", "smooth", "scaly")
levels(mushroom$stalk_surface_below_ring) <- c("fibrous", "silky", "smooth", "scaly")
levels(mushroom$stalk_color_above_ring) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                                             "green", "purple", "white", "yellow")
levels(mushroom$stalk_color_below_ring) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                                             "green", "purple", "white", "yellow")
levels(mushroom$veil_color) <- c("brown", "orange", "white", "yellow")
levels(mushroom$ring_number) <- c("none", "one", "two")
levels(mushroom$ring_type) <- c("evanescent", "flaring", "large", "none", "pendant")
levels(mushroom$spore_print_color) <- c("buff", "chocolate", "black", "brown", "orange", 
                                        "green", "purple", "white", "yellow")
levels(mushroom$population) <- c("abundant", "clustered", "numerous", "scattered", "several", "solitary")
levels(mushroom$habitat) <- c("wood", "grasses", "leaves", "meadows", "paths", "urban", "waste")

mushroom <- mushroom[-c(17)]
str(mushroom)
dim(mushroom)

table(mushroom$edibility)
table(mushroom$habitat)
plot_correlation(mushroom)
summary(is.na(mushroom))
plot_missing(mushroom)

summary(mushroom)

#Data Visualization
ggplot(mushroom, aes(x = edibility, y = odor, col = edibility))+
  geom_jitter(alpha= 0.5)+  scale_color_manual(breaks = c("edible", "poisonous"), 
                                               values = c("green", "red"))
ggplot(mushroom, aes(x = edibility, y = cap_shape, col = edibility)) + 
  geom_jitter(alpha = 0.5) + 
  scale_color_manual(breaks = c("edible", "poisonous"), 
                     values = c("green", "red"))

ggplot(aes(x = bruises), data = mushroom) +
  geom_histogram(stat = "count") +
  facet_wrap(~edibility) +
  xlab("Bruises")

ggplot(aes(x = odor), data = mushroom) +
  geom_histogram(stat = "count") +
  facet_wrap(~edibility) +
  xlab("Odor")

ggplot(aes(x = gill_size), data = mushroom) +
  geom_histogram(stat = "count") +
  facet_wrap(~edibility) +
  xlab("Gill Size")

ggplot(aes(x = stalk_color_below_ring), data = mushroom) +
  geom_histogram(stat = "count") +
  facet_wrap(~edibility) +
  xlab("Stalk Surface Below Ring")

#Model
mushroom_1R<-OneR(edibility ~., data = mushroom)
mushroom_1R
summary(mushroom_1R)

mushroom_jr<- JRip(edibility ~., data = mushroom)
mushroom_jr
summary(mushroom_jr)

#mushroom_pr<-predict(mushroom_jr, test) 
#confusionMatrix(mushroom_pr, test$edibility)

#Training,Validation and Test datasets
set.seed(1810)
sample <- sample(1:3, size = nrow(mushroom), prob = c(0.6,0.2,0.2),replace = TRUE)
train <- mushroom[sample == 1, ] 
test <-mushroom[sample == 2, ]
valid <-mushroom[sample == 3, ]
prop.table(table(train$edibility))
prop.table(table(test$edibility))
prop.table(table(valid$edibility))
str(valid)

#Decision Tree
#Hyperparameter Tuning
set.seed(123)
entropy_model <- rpart(edibility ~. , valid, method = "class",
                        parms = list(split = "information"),
                        cp= 0.00001)

printcp(entropy_model)
plotcp(entropy_model)

set.seed(123)
gini_model <- rpart(edibility ~ . , valid, method = "class",
                     parms = list(split = "gini"),
                     cp= 0.00001)
 
printcp(gini_model)
plotcp(gini_model)

#Final Model
mushroom_model<- rpart(edibility ~ . , train, method = "class",
                      parms = list(split = "gini"),cp=0.00001)
printcp(mushroom_model)
plotcp(mushroom_model)
bestcp<- round(mushroom_model$cptable[which.min(mushroom_model$cptable[,"xerror"]),"CP"],5)
mushroom_pruned<- prune(mushroom_model, cp = bestcp)

rpart.plot(mushroom_pruned, extra = 104, box.palette = "GnBu", 
           branch.lty = 3, shadow.col = "gray", nn = TRUE)

mushroom_pred<- predict(mushroom_pruned, test, type= 'class')
CrossTable(mushroom_pred, test$edibility,prop.chisq = FALSE, prop.t = FALSE,
            dnn = c('predicted', 'actual'))
confusionMatrix(table(mushroom_pred,test$edibility))

caret::confusionMatrix(data = predict(mushroom_pruned, newdata = test, type = "class"), 
                        reference = test$edibility, 
                        positive = "edible")
 
prp(mushroom_model, box.palette = "Reds")
fancyRpartPlot(mushroom_model)

#Random Forest
model_rf <- randomForest(edibility ~ ., ntree = 50, data = train)
plot(model_rf)
print(model_rf)

caret::confusionMatrix(data = model_rf$predicted, reference = train$edibility , 
                       positive = "edible")
varImpPlot(model_rf, sort = TRUE, 
           n.var = 10, main = "The 10 Variable Importance")

#Variable Importance
var.imp = data.frame(importance(model_rf, type=2))
# make row names as columns
var.imp$Variables = row.names(var.imp)  
print(var.imp[order(var.imp$MeanDecreaseGini,decreasing = T),])

rf_pred <- predict(model_rf, test)
table(rf_pred, test$edibility)
CrossTable(rf_pred, test$edibility,prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))
confusionMatrix(table(rf_pred,test$edibility))

#SVM
model_svm <- svm(edibility ~. , data=train, cost = 1000, gamma = 0.01)
test_svm <- predict(model_svm, newdata = test)
table(test_svm, test$edibility)
CrossTable(test_svm, test$edibility,prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))
confusionMatrix(table(test_svm,test$edibility))
