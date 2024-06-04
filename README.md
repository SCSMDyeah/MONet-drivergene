# SCSMD : Single Cell Consistent Clustering Based on Spectral Matrix Decomposition
In order to enhance the stability, accuracy, and adaptability of single-cell clustering, we implemented cluster integration as a method to cluster scRNA-seq data. The entire process was comprised of four modules. In the initial module, we selected seven clustering algorithms that have demonstrated promising results and recognition in the field: SC3, Seurat, SHARP, CIDR, SINCERA, Rphenograph, and RaceID. Subsequently, we inputted 15 real single-cell RNA-seq datasets sequentially into the seven scRNA-seq clustering algorithms and evaluated their performance to obtain clustering results. Next, the second module integrated the seven clustering results to generate a consensus matrix. The third module constructed a consensus distance matrix and performed distance selection based on the consensus matrix obtained from the second module. Finally, the fourth module employed spectral clustering on the consensus distance matrix to generate the final clustering results. This approach allowed for improved stability, accuracy, and adaptability in single-cell clustering.

# How to use the package for new data 
To use our method for new data, the method includes these functions:  
- SCSMD: main function is SCSMD. The input is a matrix with rows as genes and columns as samples.

# Example
library(mclust)
library(SingleCellExperiment)

#read data

d <- readRDS("data/yan/yan.rds")
data <- assay(d)                  #20214  90
label <- as.numeric(factor(d$cell_type1))
n <- length(unique(label))

######Find the label of each method######
source('SCSMD.R')
cluster_results <-SCSMD(data, n)

labels_SC3 <- cluster_results[["labels_SC3"]]
labels_Seurat <- cluster_results[["labels_Seurat"]]
labels_SHARP <- cluster_results[["labels_SHARP"]]
labels_cidr <- cluster_results[["labels_cidr"]]
labels_SINCERA <- cluster_results[["labels_SINCERA"]]
labels_Rphenograph <- cluster_results[["labels_Rphenograph"]]
labels_RaceID <- cluster_results[["labels_RaceID"]]

###### calculate ARI values for the 7 methods ######
ARI_sc3 <- adjustedRandIndex(label,as.vector(labels_SC3))
ARI_Seurat <- adjustedRandIndex(label,as.vector(labels_Seurat))
ARI_SHARP <- adjustedRandIndex(label,as.vector(labels_SHARP))
ARI_cidr <- adjustedRandIndex(label,as.vector(labels_cidr))
ARI_SINCERA <- adjustedRandIndex(label,as.vector(labels_SINCERA))
ARI_Rphenograph <- adjustedRandIndex(label,as.vector(labels_Rphenograph))
ARI_RaceID <- adjustedRandIndex(label,as.vector(labels_RaceID))

source("select method.R")
source("the most k.R")
######the number of the cluster######
kr <- cluster_number_range(labels_SC3,
                           labels_Seurat,
                           labels_SHARP,
                           labels_cidr,
                           labels_SINCERA,
                           labels_Rphenograph,
                           labels_RaceID)
m <-min(kr)-1

######Screening clustering method######
select_method<- select_method(labels_SC3,labels_Seurat,labels_SHARP,labels_cidr,labels_SINCERA,labels_Rphenograph,labels_RaceID)
select_method[["rank"]]

cluster_result <- cluster_results[["label_7method"]]
cluster_result <-cluster_result[-c(1,3,7),,drop=FALSE]

library(SC3)
consensus_result<-consensus_matrix(t(cluster_result))
library('philentropy')
library(proxy)
distance_cell<-distance(consensus_result, method = "chebyshev")

library("ctv")
library("kernlab")

source("the most k.R")
k_select <- cluster_number(distance_cell,kr,m)
k_select

k <-kr[which.max(k_select)]
k

ARI_result <- list()
label_result <- list()
for (i in 1:200){
sc<-specc(distance_cell,centers=k, kernel ="rbfdot", kpar = "automatic",nystrom.red = FALSE, nystrom.sample =dim(x)[1]/6, iterations = 200,mod.sample = 0.75, na.action =na.omit)
label_sc<-sc@.Data
ARI_yan <- adjustedRandIndex(label,as.vector(label_sc))
ARI_result[[i]] <- ARI_yan
label_result[[i]] <- label_sc
}


ARI_result_vector <- as.vector(unlist(ARI_result))

ARI_yan <-getmode(ARI_result_vector)

ARI_yan
yan_ARI_overall <- c(ARI_sc3, ARI_Seurat, ARI_SHARP, ARI_cidr, ARI_SINCERA, ARI_Rphenograph, ARI_RaceID,ARI_yan)
yan_ARI_overall
yan_overall_rank <-rank(yan_ARI_overall)
yan_overall_rank
as.numeric(table(ARI_result_vector)) 
names(table(ARI_result_vector))   
