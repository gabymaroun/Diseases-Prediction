---
title: "Prédiction d’une maladie multifactorielle à l’aide des génotypes individuels"
author: "Gaby Maroun"
date: "5/5/2020"
output:
  rmdformats::readthedown:
  # prettydoc::html_pretty:
    highlight: kate
    df_print: paged
    
    # number_sections: yes
  #   theme: cayman
  #   kramdown:
  #     toc_levels: 2..3
    # toc: yes
    # toc_float:
    #   collapsed: yes
    #   smooth_scroll: yes
  # pdf_document:
  #   toc: yes
resource_files: DATA_PROJET1.RData
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
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
```

# Introduction

  Dans ce document, on explorera plusieurs types différents de modèles prédictifs: `glmnet`, `rainforest`, `gmb` et `evtree`, et proposerons même une autre solution technique adaptée à la réalisation d'un apprentissage supervisé afin de produire un modèle capable de prédire l'état d'une maladie avec des données à notre disposition issues de 6 études sur 6 centres européens concernant 100 patients et leurs 1000 allèles (il n'existe que deux allèles différents dans la population)..
  
* La valeur 0 = AA : l’individu a hérité de l’allèle majeur A de chacun de ses deux parents.
  
* La valeur 1 = AG : l’individu a hérité d’un allèle A, et d’un allèle G (l’information de transmission de l’allèle par le père ou la mère est inconnue).

* La valeur 2 = GG : l’individu a hérité de l’allèle mineur G de chacun de ses deux parents.

Tout d'abord, on va commencer par mettre les données dans un bloc de données pour avoir une bonne vue sur leur distribution:

```{r df, echo=FALSE}
# Charger les données du fichier DATA_projet1.RData
load(file = "C:/Users/Gaby\'s/Desktop/Semestre 2 UPPA/Machine Learning/Projet 1/DATA_projet1.RData")
# Mettant les données en un table de type data.frame
df<-data.frame(DATA_projet1)
df
```

Ensuite, on s'assure s'il y a des valeurs manquantes avec `is.na ()` pour tourner notre concentration sur le `preprocessing`, ce qui n'est pas le cas ici:
```{r na}
# Vérifier s'il y a des cases de données N/a ou vide pour utiliser preprocessing
sum(is.na(df))
```

# Création d'une partition de données

En créant une partition de données adaptée à l'apprentissage, nous aurons une table de données plus lisible et facilement accessible.

Donc, pour ce faire, on prend les 6 dernières colonnes du `df` qui représentent les statuts de maladie des 100 patients dans chaque étude et on les met les uns au-dessus des autres dans la même colonne` M` dans notre nouveau tableau `aEtudier` et avec `factor`, on change leur type pour qu'il soit «facteurs» de "Oui"(Il va être malade) et "Non" (Il ne va pas être malade) au lieu d'«entier naturel»:
```{r aEtudier, results = 'hide'}
# Construire une table de données comme notre base d'étude 
aEtudier <- NULL
for(i in 1:5){
  aEtudier <- (M =c(aEtudier,df[,6000+i]))
}
# Transformer les valeurs du statuts de maladies en des facteurs (classes)
aEtudier <-data.frame(M = factor(aEtudier, labels = c("Non", "Oui")))
```

Dans une nouvelle table de données `aPredire` qui représente la liste dont les résultats prévus seront mis, on remplisse une colonne qui ne représente que «0» initialement:
```{r aPredire, results = 'hide'}
# Construire une table de données à prédire
aPredire<- NULL
for(i in 1:100){
  aPredire<-append(aPredire,0)
}
aPredire <- as.factor(aPredire)
aPredire <- data.frame(M = aPredire)
```

Pour terminer cette étape, nous joignons les valeurs génotypiques de chaque patient avec son propre statut de maladie:

```{r aEtudieraPredire, results = 'hide'}
# Remplir la table aEtudier par les données genotypiques de chaque patients
# des premiers 5 études :
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
```
 
# Création d'indices de train / test personnalisés

La première chose à faire est donc de créer un objet `trainControl` réutilisable que nous pouvons utiliser pour les comparer de manière fiable.

On définisse également une «répartition aléatoire» avec `set.seed` afin que notre travail soit reproductible et on obtient la même distribution aléatoire chaque fois qu'on exécute notre script.

On utilise `createFolds ()` pour faire 5 plis `CV` sur` M`, notre variable cible pour cet exercice:
```{r myFolds, results = 'hide'}
# Définir une graine aléatoire afin que notre travail soit reproductible:
set.seed(42)
# Création des plis
myFolds <- createFolds(aEtudier$M, k = 5)
```

On les passe à `trainControl ()` pour créer un trainControl réutilisable pour comparer les modèles comme déjà décrit:
```{r myControl, results = 'hide'}
# Création du reUsable trainControl object
myControl <- trainControl(
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)
```

# Adapter les modèles au `train`

On a choisi d'essayer 4 modèles différents, afin d'avoir plusieurs points de vue et d'être sûr d'avoir des résultats plus exacts et précis.

Dans ce qui suit, on décrit notre approche pour tourner un modèle avec les choix optimaux mais les résultats optimaux seront tournés dans le fichier `script`.

## Generalized Linear Model

Les modèles linéaires généralisés (GLM) sont une extension des modèles de régression linéaire «simples», qui prédisent la variable de réponse en fonction de plusieurs variables prédictives. Les modèles de régression linéaire fonctionnent sur quelques hypothèses, telles que l'hypothèse selon laquelle nous pouvons utiliser une ligne droite pour décrire la relation entre la réponse et les variables prédictives.

On a choisi `glmnet` car il est simple, rapide, facile à interpréter et pénalise les modèles de régression linéaire et logistique sur la taille et le nombre de coefficients pour éviter le `overfitting.`

Pour cela, on a spécifié l'alpha ("Ridge regression" (ou ` alpha = 0`), "Lasso regression" (ou` alpha = 1`)) et `lambda`  (taille de la pénalité).

Ensuite, on va commencé à adapter `glmnet` à notre ensemble de données sur les maladies et à évaluer sa précision prédictive à l'aide de `myControl` déjà créé:

```{r model_glmnet, results = 'hide'}
# Adaptez le modèle glmnet à nos données:
glmnet_grid = expand.grid(alpha = 0:1,
            lambda = seq(0.0001, 1, length = 100))
model_glmnet <- train(
  M~., aEtudier,
  metric = "ROC",
  tuneGrid= glmnet_grid,
  method = "glmnet",
  trControl = myControl
)
```

Après avoir essayé beaucoup de `lambda` et `alpha`, nous pouvons voir qu'un `alpha = 0` (`Ridge`) est le meilleur choix pour empêcher le `overfitting`, comme le montre le graphique suivant:

```{r plotmodel_glmnet3}
#Montrez une comparaison entre les coefficients:
plot(model_glmnet)
```

On peut voir dans ce qui suit que lorsque nous augmentons `lambda` c. à d. la pénalité, les points sortent du modèle:
```{r plotmodel_glmnet2}
plot(model_glmnet$finalModel)
```

Dans le graphique suivant, nous pouvons voir les allèles les plus importants ou les plus influents sur les prédictions de ce modèle:
```{r plotmodel_glmnet}
#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_glmnet,scale=F),top = 15)
```

### AVANTAGES

* La variable de réponse peut avoir n'importe quelle forme de type de distribution exponentielle,
* Capable de gérer des prédicteurs catégoriques,
* Relativement facile à interpréter et permet de comprendre clairement comment chacun des prédicteurs influence le résultat,
* Moins sensible au `overfitting`.

### LIMITES

* Nécessite des ensembles de données relativement volumineux. Plus il y a de variables prédictives, plus la taille d'échantillon requise est grande. En règle générale, le nombre de variables prédictives doit être inférieur à N / 10 (5000/10=500>100 comme dans notre cas ce qui est bon),
* Sensible aux valeurs aberrantes.

## Rainforest

Les forêts aléatoires (`Rainforest`) sont une extension d'arbres de classification uniques dans lesquels plusieurs arbres de décision sont construits avec des sous-ensembles aléatoires des données. Il combine un ensemble d'arbres de décision non linéaires en un modèle très flexible et généralement assez précis.

Les `rainforest` sont un peu plus difficiles à interpréter que les modèles linéaires, bien qu'il soit toujours possible de les comprendre.

On utilise le paquet `ranger` qui est une réimplémentation de` randomForest` qui produit presque exactement les mêmes résultats, mais qui est plus rapide, plus stable et utilise moins de mémoire

Le `tuneGrid` nous donne un contrôle plus fin sur les paramètres de réglage qui sont explorés et ensuite une vitesse pour tourner le modèle sur les points qui nous importent. Les `mtry` ont été choisis après plusieurs exécutions et une grande` tuneLength.`

Et ensuite, nous commencerons à adapter `ranger` à notre ensemble de données sur les maladies et à évaluer sa précision prédictive à l'aide de `trainControl` déjà créé:
```{r model_ranger, results = 'hide'}
#Adaptez le modèle ranger représentant le type de modèle RainForest à nos données:
#Création d'un tuneGrid pour le modèle
rangerGrid <- data.frame(.mtry =c(2,3,5,7,10,11,14,19,27,37,52,73,101,140,194,270,374,519,721,1000), 
                       .splitrule = "gini", 
                       .min.node.size = 1)
model_ranger <- train(
  M~., aEtudier,
  #tuneGrid=rangerGrid,
  metric = "ROC",
  method = "ranger",
  tuneLength=5, #20,30,50
  trControl = myControl
)
```

Alors pour un `splitrule = "gini"` et un `mtry=11`, on arrive à un `ROC` maximum pour ce modèle:
```{r plotmodel_ranger1}
plot(model_ranger)
```

### AVANTAGES

* L'un des algorithmes d'apprentissage les plus précis disponibles,
* Il peut gérer de nombreuses variables prédictives,
* Fournit des estimations de l'importance de différentes variables prédictives,
* Maintient la précision même lorsqu'une grande partie des données est manquante(ce qui n'est pas le cas là).

### LIMITES

* Peut surcharger des ensembles de données particulièrement bruyants,
* Pour les données comprenant des variables prédictives catégorielles avec différents nombres de niveaux, les forêts aléatoires sont biaisées en faveur des prédicteurs avec plus de niveaux. Par conséquent, les scores d'importance variable de la forêt aléatoire ne sont pas toujours fiables pour ce type de données.

## Generalized Boosting Model

Ces modèles sont une combinaison de deux techniques: les algorithmes d'arbre de décision et les méthodes de boosting. Les modèles de boosting généralisés s'adaptent à plusieurs reprises à de nombreux arbres de décision pour améliorer la précision du modèle. Alors que les forêts aléatoires construisent un ensemble d'arbres indépendants profonds, les GBM construisent un ensemble d'arbres successifs peu profonds et faibles, chaque arbre apprenant et améliorant le précédent. 

Pour chaque nouvelle arborescence du modèle, un sous-ensemble aléatoire de toutes les données est sélectionné à l'aide de la méthode de boosting. Pour chaque nouvel arbre du modèle, les données d'entrée sont pondérées de telle sorte que les données mal modélisées par les arbres précédents ont une probabilité plus élevée d'être sélectionnées dans le nouvel arbre. 

Cela signifie qu'après l'ajustement du premier arbre, le modèle tiendra compte de l'erreur de prédiction de cet arbre pour l'ajustement de l'arbre suivant, etc. Cette approche séquentielle est unique au boosting.

On a choisi ce modèle en raison de sa bonne réputation :

```{r model_gbm, results = 'hide'}
# Adaptez le modèle gbm à nos données:
gbmGrid <- data.frame(n.trees = c(50,300), 
                      interaction.depth = c(2,5), 
                      shrinkage = 0.1 ,
                      n.minobsinnode = 10
                      )
model_gbm <- train(
  M~., aEtudier,
 # tuneGrid = gbmGrid,
  metric = "ROC",
  method = "gbm",
  tuneLength=5,
  trControl = myControl,
)
```

Après plusieurs exécutions, nous avons pu choisir ce `tuneGrid` qui diminue l'influence des prédicteurs non utiles jusqu'à 81 pour 1000.

Nous pouvons voir dans ce graphique, les variables les plus influentes:

```{r plotmodel_gbm3}
#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_gbm,scale=F),top = 15)
```

Un `n.trees =50` et `interaction.depth = 4` est le choix optimal:
```{r plotmodel_gbm1}
plot(model_gbm)
```

```{r plotmodel_gbm2}
model_gbm$finalModel
```

### AVANTAGES
* Peut être utilisé avec une variété de types de réponses (binôme, gaussien, poisson),
Stochastique, qui améliore les performances prédictives
* Le meilleur ajustement est automatiquement détecté par l'algorithme,
* Le modèle représente l'effet de chaque prédicteur après avoir pris en compte les effets des autres prédicteurs,
* Robuste aux valeurs manquantes et aux valeurs aberrantes.

### LIMITES

* Nécessite au moins 2 variables prédictives pour s'exécuter

## Evolutionnary Trees

Les modèles arborescents répartissent les données en groupes de présence ou d'absence de plus en plus homogènes en fonction de leur relation avec un ensemble de variables environnementales, les variables prédictives. L'arbre de classification unique est la forme la plus élémentaire d'un modèle d'arbre de décision. Comme son nom l'indique, les arbres de classification ressemblent à un arbre et se composent de trois types de nœuds différents, reliés par des bords dirigés (branches).

On a choisi ce modèle pour plus de clarté, sa certitude de prédiction et l'occasion de voir la forme d'arbre réalisée:
```{r model_evtree, results = 'hide'}
# Adapter le modèle d'arbres à partir d'algorithmes génétiques evtree
# à notre données: 
model_evtree <- train(
  M~., aEtudier,
  metric = "ROC",
#  tuneGrid= data.frame(alpha =1),
  method = "evtree",
  trControl = myControl
)
#model_evtree
```

On peut voir cet arbre à la `root` ici:
```{r plotmodel_evtree1}
#La forme de l'arbre:
plot(model_evtree$finalModel)
```

On peut constater de ce qui suit, que `alpha=1` est le choix optimal pour le plus grand `ROC` possible:
```{r plotmodel_evtree2}
#Montrez le modèle sur un graphique:
plot(model_evtree)
```

On peut voir dans ce graphique, les variables qui influencent le plus la décision:
```{r plotmodel_evtree3}
#Les 15 variables qui influencent le plus la décision de 
#prédiction de ce modèle:
plot(varImp(model_evtree,scale=F),top = 15)
```

### AVANTAGES

* Simple à comprendre et à interpréter,
* Peut gérer à la fois des données numériques et catégoriques,
* Identifier les interactions hiérarchiques entre les prédicteurs,
* Caractériser les effets de seuil des prédicteurs sur la présence d'espèces,
* Robuste aux valeurs manquantes et aux valeurs aberrantes.

### LIMITES

* Moins efficace pour les réponses d'espèces linéaires ou lisses en raison de l'approche par étapes,
* Nécessite de grands ensembles de données pour détecter les modèles, en particulier avec de nombreux prédicteurs,
* Très instable: de petits changements dans les données peuvent changer considérablement l'arbre,
* Prend beaucoup de temps pour tourner.

# Création d'un objet `resamples`

Maintenant qu'on a ajusté les modèles à notre ensemble de données, il est temps de comparer leurs prédictions hors échantillon et de choisir celle qui est la meilleure dans notre cas.

Nous pouvons le faire en utilisant, et selon `caret`, la fonction` resamples () `:

```{r resamps, results = 'hide'}
# Création d'une liste de modèles:
model_list <- list(
  glmnet = model_glmnet,
  ranger = model_ranger,
  evtree = model_evtree,
  gbm = model_gbm
)
# Insérez la liste des modèles dans les resamples ():
resamps <- resamples(model_list)
resamps
```

```{r sumresamps}
# Résumez les résultats
summary(resamps)
```

# Comparaison graphiques

Pour que la comparaison entre les modèles soit facilement visible, on affiche les distributions de précision prédictive dans des diagrammes:

## Boîte à Moustaches

```{r bwplot}
# Création d'une boîte à moustache de points ROC:
bwplot(resamps, metric = "ROC")
```

On peut voir sur ce diagramme de boîte à moustaches que `glmnet` a une médiane plus grande que celle des 3 autres modèles, en plus la différence entre le minimum et le maximum est plus petite pour `glmnet`, ce qui indique une distribution plus courte, c'est-à-dire moins données dispersées.

## DotPlot

```{r dotplot}
# Creation du DotPlot des points du ROC
dotplot(resamps, metric = "ROC")
```

Ce diagramme nous montre les mêmes informations de la boîte à moustaches mais on ne voit que la moyenne qui semble plus grand pour `glmnet` que les autres.

## Diagramme de densité

```{r densityplot}
# Création du diagramme de densité des points du ROC:
densityplot(resamps, metric = "ROC")
```

De ce diagramme de densité, on n'arrive pas à avoir une bonne déduction à cause du grand nombre de courbes.

## Nuage de Points

```{r xyplot}
# Création du nuage des points du ROC:
xyplot(resamps, metric = "ROC")
```

Enfin, selon le `scattertrot` ci-dessus, nous pouvons comparer directement le` AUC` sur les 5 plis `cv`. On voit donc que la plupart du temps, `glmnet` avait le plus grand` AUC` et que les 2 modèles sont totalement différents, puis comparables.

# Prédiction

On peut voir sur les diagrammes et les résultats ci-dessus, que le modèle `glmnet` est le plus approprié de ces 4 modèles étudiés et donc, nous choisirons ce modèle pour la prédiction des résultats de la 6ème étude:

```{r predicted}
## Prédiction1
# Prédire le statut des maladies des patients du 6eme étude avec le 
# modèle choisit
Prediction_s6 <- data.frame(Predicted = predict(model_glmnet,aPredire))
#Affichage des prédictions
Prediction_s6
```

# Sauvgarder

Pour enregistrer nos résultats de prédiction dans un fichier `.RData`, ce sera comme suit:

```{r save, eval = 'False'}
# Sauvegarder les prédictions dans un fichier .RData
save(Prediction_s6, file = "Projet1_MarounGaby_prediction.RData")
```

# Modèle Proposer
Après être parvenu à cette conclusion et prédire l'état de la maladie avec `glmnet` comme modèle en raison de son `AUC` plus grande que les autres modèles.

On va faire des graphiques discutant la distribution des prédictions de `Non` ou `Oui` de ces modèles (c'est à peu près la même distribution pour les 2 classes alors on va continué avec un seul) et leur certitude:

## Generalized Linear Model

```{r plotmodel_glmnet4}
bwplot(model_glmnet$pred$Non)
dotplot(model_glmnet$pred$Non)
```

De ce graphique, on peut déduire que ce modèle n'est pas toujours «sûr» de ses résultats car il a une médiane et un moyen de prédiction de «Non» qui est proche de `50%` et ensuite il conduit à une certitude insuffisante des résultats .

## RainForest

```{r plotmodel_ranger4}
bwplot(model_ranger$pred$Non)
dotplot(model_ranger$pred$Non)
```

De ce graphique, on peut déduire que ce modèle n'est pas, également aux précédents, toujours "sûr" de ses résultats car il a un moyen de prédictions de "Non" qui est proche de "50%" et ensuite il conduit à résultats de précision insuffisants.

## Generalized Boosting Model

```{r plotmodel_gbm4}
bwplot(model_gbm$pred$Non)
dotplot(model_gbm$pred$Non)
```

D'après ce graphe, on peut déduire que ce modèle n'est pas, également aux précédents, toujours "sûr" de ses résultats parce qu'il a un moyen des predictions de "Non" qui est près à 50% et alors ca ammène à des resultats pas assez precis.

## Evolutionnary Trees

```{r plotmodel_evtree4}
bwplot(model_evtree$pred$Non)
dotplot(model_evtree$pred$Non)
```

De ce graphique, on peut déduire que ce modèle est, contrairement aux modèles précédents, très "sûr" de ses résultats car il dispose d'un moyen de prédictions et d'une médiane de "Non" qui est toujours loin de l'intervalle [40% -60%], alors c'est "Oui" ou "Non" et cela conduit alors à des résultats certains.

## Prédiction 2

En regardant la distribution des prédictions, on peut voir que `evtree` est le plus approprié et le plus fiable de ces résultats et, par conséquent, je suggère `evtree` pour calculer la prédiction des statuts de la maladie comme une deuxième option.

```{r predicted2}
## prediction 2
# Prédire l'état de maladie des patients de la 6e étude 
#avec le modèle choisi
Prediction2_s6 <- data.frame(Predicted2 = predict(model_evtree,aPredire))
#Affichage des prédictions:
Prediction2_s6
```

# Conclusion
Il n'est pas facile de prédire l'avenir et ce qui limite la prédiction est l'ambiguïté sur l'avenir et l'incertitude. De plus, 500 patients ne constituent pas un grand échantillon et ne ciblent pas une grande majorité.

Pour conclure, j'espère avoir été clair et avoir bien discuté et interprété les problèmes et les procédures pour arriver à ces 2 prédictions choisies après avoir bien compris ce que je fais et ce que j'ai écrit.

J'ai vraiment aimé travailler avec ces outils (ce qui est évident je pense). Je voudrais avoir des remarques, qui comptent plus pour moi que les notes, pour m'améliorer et rester sur la bonne voie et je serai à votre disposition pour toutes sortes de questions.

                                          La fin.
