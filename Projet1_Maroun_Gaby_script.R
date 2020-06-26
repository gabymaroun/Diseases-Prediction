####### Prédiction d’une maladie multifactorielle à l’aide des génotypes individuels #######
####### Gaby Maroun #######

library(ggplot2 )
library(caret)
library(caretEnsemble)
library(C50)
library(tidyverse)
library(reshape2)
library(glmnet)
library(randomForest)
library(ranger)
library(gbm)

# Chargeant les données du fichier DATA_projet1.RData
load(file = "C:/Users/Gaby\'s/Desktop/Semestre 2 UPPA/Machine Learning/Projet 1/DATA_projet1.RData")

# Mettant les données en un table de type data.frame
df<-data.frame(DATA_projet1)
View(df)

# Vérifier s'il y a des cases de données N/a ou vide 
#pour utiliser preprocessing
sum(is.na(df))

# Construire une table de données comme notre base d'étude en remplissant la première
# colonne des statuts de maladies des patients de chaques études,
# l'une au-dessus de l'autre
aEtudier <- NULL
for(i in 1:5){
  aEtudier <- (M =c(aEtudier,df[,6000+i]))
}

# Transformer les valeurs du statuts de maladies en des facteurs (classes):
aEtudier <-data.frame(M = factor(aEtudier, labels = c("Non", "Oui")))

# Construire une table de données en remplissant la première colonne en 0
# initialement comme statut de maladies des patients de la 6eme étude:
aPredire<- NULL
for(i in 1:100){
  aPredire<-append(aPredire,0)
}
aPredire <- as.factor(aPredire)
aPredire <- data.frame(M = aPredire)


# Remplir la table aEtudier par les données genotypiques de chaque patients
# des premiers 5 études:
for (i in 1:1000) {
  j <- 0
  k <- 1
  for (genotypes in DATA_projet1$Genotypes[1:5]) {
    j <- j + 100
    aEtudier[k:j,paste("x",as.character(i),sep="")]<-genotypes[,i]
    k <- k + 100
  }
}

# Remplir la table aPredire par les données genotypiques de chaque patients
# du 6eme études:
for (i in 1:1000) {
  for (genotypes in DATA_projet1$Genotypes[6]) {
    aPredire[,paste("x",as.character(i),sep="")]<-genotypes[,i]
  }
}

# Définir une graine aléatoire afin que notre travail soit reproductible:
set.seed(42)

# Création des plis
myFolds <- createFolds(aEtudier$M, k = 5)
# summary(myFolds)


# Création du reUsable trainControl object
myControl <- trainControl(
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)



# Dans ce qui suit, on va utiliser que les choix optimaux 
# tandis que dans le rapport, vous trouverez une description 
# montrant le chemin pour arriver à ces conséquences.

# Adaptez le modèle glmnet à nos données:

# Précision des alpha (Ridge regression(alpha = 0),Lasso regression(or alpha = 1))
# et lambda (taille du pénalité) pour empêcher l'overfitting.
glmnet_grid = expand.grid(alpha = 0,#:1,
            lambda = seq(0.0001, 1, length = 100))

model_glmnet <- train(
  M~., aEtudier,
  metric = "ROC",
  tuneGrid= glmnet_grid,
  method = "glmnet",
  trControl = myControl
)
model_glmnet
plot(model_glmnet)

#Montrez le modèle sur un graphique:
plot(model_glmnet$finalModel)

#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_glmnet,scale=F),top = 15)

#Adaptez le modèle ranger représentant le type de modèle RainForest à nos données:
#Création d'un tuneGrid pour le modèle
rangerGrid <- data.frame(.mtry =11, #c(2,3,5,7,10,11,14,19,27,37,52,73,101,140,194,270,374,519,721,1000),
                       .splitrule = "gini",
                       .min.node.size = 1)
model_ranger <- train(
  M~., aEtudier,
  tuneGrid=rangerGrid,
  metric = "ROC",
  method = "ranger",
  tuneLength=5,
  trControl = myControl
)
model_ranger
model_ranger$finalModel


# Adapter le modèle d'arbres à partir d'algorithmes génétiques evtree
# à notre données: 
model_evtree <- train(
  M~., aEtudier,
  metric = "ROC",
  tuneGrid= data.frame(alpha =1),
  method = "evtree",
  trControl = myControl
)
model_evtree

#La forme de l'arbre:
plot(model_evtree$finalModel)

#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_evtree,scale=F),top = 15)


# Adaptez le modèle gbm à nos données:
gbmGrid <- data.frame(n.trees = 50, 
                      interaction.depth = 4, 
                      shrinkage = 0.1 ,
                      n.minobsinnode = 10
                      )
model_gbm <- train(
  M~., aEtudier,
  tuneGrid = gbmGrid,
  metric = "ROC",
  method = "gbm",
  tuneLength=5,
  trControl = myControl,
)
model_gbm
summary(model_gbm)

#Montrez le modèle sur un graphique:
model_gbm$finalModel


#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_gbm,scale=F),top = 15)

# Création d'une liste de modèles:
model_list <- list(
  glmnet = model_glmnet,
  ranger = model_ranger,
  evtree = model_evtree,
  gbm = model_gbm
  # svmRadial = model_svmRadial
)

# Insérez la liste des modèles dans les resamples ():
resamps <- resamples(model_list)
resamps

# Résumez les résultats
summary(resamps)

# Création d'une boîte à moustache des points du ROC:
bwplot(resamps, metric = "ROC")

# Création du DotPlot des points du ROC:
dotplot(resamps, metric = "ROC")

# Création du diagramme des densité de points du ROC:
densityplot(resamps, metric = "ROC")

# Création du nuage des points du ROC:
xyplot(resamps, metric = "ROC")



## Prédiction1
# Prédire le statut des maladies des patients du 6eme étude avec le 
# modèle choisit
Prediction_s6 <- data.frame(Predicted = predict(model_glmnet,aPredire))

#Affichage des prédictions
View(Prediction_s6)


# Sauvegarder les prédictions dans un fichier .RData
save(Prediction_s6, file = "Projet1_Maroun_Gaby_prediction.RData")

## prediction 2

# Distribution des prédictions de chaque modèle et leur certitude:

# glmnet
bwplot(model_glmnet$pred$Non)
bwplot(model_glmnet$pred$Oui)
dotplot(model_glmnet$pred$Non)
dotplot(model_glmnet$pred$Oui)

#rainforest ou ranger
bwplot(model_ranger$pred$Non)
dotplot(model_ranger$pred$Non)
bwplot(model_ranger$pred$Oui)
dotplot(model_ranger$pred$Oui)

#gbm
bwplot(model_gbm$pred$Non)
dotplot(model_gbm$pred$Non)
bwplot(model_gbm$pred$Oui)
dotplot(model_gbm$pred$Oui)

#evtree
bwplot(model_evtree$pred$Non)
dotplot(model_evtree$pred$Non)
bwplot(model_evtree$pred$Oui)
dotplot(model_evtree$pred$Oui)

#prediction
# Prédire l'état de maladie des patients de la 6e étude 
#avec le modèle choisi
Prediction2_s6 <- data.frame(Predicted2 = predict(model_evtree,aPredire))

#Affichage des prédiction
View(Prediction2_s6)

#Comparaison visuel entre les prédictions des 2 modèles:
Comparaison <-data.frame(c(Prediction_s6,Prediction2_s6))

View(Comparaison)

# Distribution des probabilités des 'Non' après prédictions 
#qui assure notre choix 
Predicted2_Prob = predict(model_evtree,aPredire, type='prob')
dotplot(Predicted2_Prob$Non)

# Sauvegarder les prédictions dans un fichier .RData
save(Prediction2_s6, file = "Projet1_Maroun_Gaby_prediction2.RData")








############################################################
############################################################

# A partir de là, les fonctions seront hors projet 
# mais je les laisse ici car j'ai mis beaucoup de temps à les exécuter 
# pour mieux comprendre comment travailler avec ce package et parce qu'ils 
# fonctionnent bien avec nos données mais je pense que ce que 
# j'ai fait en haut suffit


## Un autre modèle pour faire notre prédiction sera un ensemble 
## de plusieurs modèle avec le caretEnsemble qui combien plusieur
## modèle du paquet Caret en un seule comme montré là


## CaretEnsemble
# Creation du modèle caretList qui combien plus qu'un modèle normal
models <- caretList(
  M~., aEtudier,
  metric = "ROC",
  trControl = myControl,
  methodList = c("glmnet", "ranger")
)

# Creation l'ensemble de modèle:
stack <- caretStack(all.models = models, method = "glm") 
stack

# Description du stack
summary(stack)

# Résumer des résultats
resampstack <- resamples(models)

# Creation boite à moustache des points du ROC
bwplot(resampstack, metric = "ROC")

# Creation du DotPlot des points du ROC
dotplot(resampstack, metric = "ROC")

# Creation du diagramme de densité des points du ROC
densityplot(resampstack, metric = "ROC")

# Creation du nuage des points du ROC
xyplot(resampstack, metric = "ROC")




## Regression
# on suit la même démarche pour la regression mais on ne transforme pas les valeurs
# des maladies en facteurs

aEtudier <- NULL
for(i in 1:5){
  aEtudier <- (M =c(aEtudier,df[,6000+i]))
}
aEtudier <-data.frame(M = aEtudier)

aPredire<- NULL
for(i in 1:100){
  aPredire<-append(aPredire,0)
}
aPredire <- data.frame(M = aPredire)

for (i in 1:1000) {
  j <- 0
  k <- 1
  for (genotypes in DATA_projet1$Genotypes[1:5]) {
    j <- j + 100
    aEtudier[k:j,paste("x",as.character(i),sep="")]<-genotypes[,i]
    k <- k + 100
  }
}

for (i in 1:1000) {
  for (genotypes in DATA_projet1$Genotypes[6]) {
    aPredire[,paste("x",as.character(i),sep="")]<-genotypes[,i]
  }
}

objControl <- trainControl(method='cv', 
                           number=5, 
                           returnResamp='final')

# Fit glmnet model: model_glmnet
getModelInfo()$gbm$type
model_glmnet <- train(
  M~., aEtudier,
  metric = "RMSE",
  method = "glmnet",
  trControl = objControl
)
summary(model_glmnet)
plot(model_glmnet)
plot(model_glmnet$finalModel)
plot(varImp(model_glmnet,scale=F),top = 15)

# Fit random forest: model_ranger
model_rainforest <- train(
  M~., aEtudier,
  metric = "RMSE",
  method = "rf",
  trControl = objControl
)
plot(model_ranger)
getModelInfo()$ranger$type

# Create model_list
model_list <- list(
  glmnet = model_glmnet,
  rf = model_rainforest
)
# Pass model_list to resamples(): resamples
# Collect resamples from the CV folds
resamps <- resamples(model_list)
resamps

# Summarize the results
summary(resamps)

# Create bwplot
bwplot(resamps, metric = "RMSE")

# Create DotPlot
dotplot(resamps, metric = "RMSE")

# Create densityplot
densityplot(resamps, metric = "RMSE")

# Create xyplot
xyplot(resamps, metric = "RMSE")

# On obtient un résultat à peu près pareil à celui du Classification en cherchant
# le plus petit RMSE  





## Des modèles essayé mais qui je trouve pas qu'ils ont une place 
## forte dans notre situation

# Adapter le modèle des machines à vecteur de support 
# à notre données:
# Étant donné un ensemble d'exemples de formation, chacun marqué
# comme appartenant à l'une ou l'autre des deux catégories présenté, 
# un algorithme de formation SVM construit un modèle qui attribue 
# de nouveaux exemples à une catégorie ou à l'autre, ce qui en 
# fait un classificateur linéaire binaire non probabiliste

model_svmRadial <- train(
  M~., aEtudier,
  method = "svmRadial",
  tuneLength = 8,
  metric = "ROC",
  trControl = myControl
)

#Montrer le modèle sur un graphe:
plot(model_svmRadial)
model_svmRadial$finalModel
bwplot(model_svmRadial$pred$Non)
dotplot(model_svmRadial$pred$Non)
bwplot(model_svmRadial$pred$Oui)
dotplot(model_svmRadial$pred$Oui)

#Les 15 variables les plus influentes sur la décision de prédiction de ce modèle:
plot(varImp(model_evtree,scale=F),top = 15)