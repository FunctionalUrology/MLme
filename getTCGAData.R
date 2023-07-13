################################################################
# title: "Script Used for Fetching and Processing TCGA-BRCA mRNA
#        and miRNA Datasets for the MLcps Manuscript."

# author: "Akshay"
################################################################

library(SummarizedExperiment)
library(TCGAbiolinks)

setwd("~/Desktop/")



################################################################
#                       mRNA Dataset
################################################################
query.exp <- GDCquery(
  project = "TCGA-BRCA", 
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification", 
  workflow.type = "STAR - Counts",
  sample.type = c("Primary Tumor","Solid Tissue Normal")
)


GDCdownload(
  query = query.exp,
  files.per.chunk = 100
)

brca.exp <- GDCprepare(
  query = query.exp, 
)


# get subtype information
infomation.subtype <- TCGAquery_subtype(tumor = "BRCA")

# get clinical data
information.clinical <- GDCquery_clinic(project = "TCGA-BRCA",type = "clinical") 

# Which samples are Primary Tumor
samples.primary.tumour <- brca.exp$barcode[brca.exp$shortLetterCode == "TP"]

# which samples are solid tissue normal
samples.solid.tissue.normal <- brca.exp$barcode[brca.exp$shortLetterCode == "NT"]

dataPrep <- TCGAanalyze_Preprocessing(
  object = brca.exp, 
  cor.cut = 0.6
)   

dataNorm <- TCGAanalyze_Normalization(
  tabDF = dataPrep,
  geneInfo = geneInfoHT,
  method = "gcContent"
)     

dataFilt <- TCGAanalyze_Filtering(
  tabDF = dataNorm,
  method = "quantile", 
  qnt.cut =  0.25
)   

dataDEGs <- TCGAanalyze_DEA(
  mat1 = dataFilt[,samples.solid.tissue.normal],
  mat2 = dataFilt[,samples.primary.tumour],
  Cond1type = "Normal",
  Cond2type = "Tumor",
  fdr.cut = 0.001 ,
  logFC.cut = 2,
  method = "glmLRT",
  pipeline = "edgeR"
) 

degCM=dataFilt[rownames(dataDEGs),]
degCM=as.data.frame(t(degCM))
degCM$status <- ifelse(rownames(degCM) %in% samples.solid.tissue.normal, "Normal", "Tumor")
write.csv(degCM, "TCGA-BRCA_new.csv", row.names = TRUE)




################################################################
#                         miRNA Dataset
################################################################

query.miRNA <- GDCquery(
  project = "TCGA-BRCA", 
  experimental.strategy = "miRNA-Seq",
  data.category = "Transcriptome Profiling", 
  data.type = "miRNA Expression Quantification"
)

GDCdownload(query = query.miRNA)

dataAssy.miR <- GDCprepare(query = query.miRNA)
rownames(dataAssy.miR) <- dataAssy.miR$miRNA_ID

# using read_count's data 
read_countData <-  colnames(dataAssy.miR)[grep("count", colnames(dataAssy.miR))]
dataAssy.miR <- dataAssy.miR[,read_countData]

colnames(dataAssy.miR) <- gsub("read_count_","", colnames(dataAssy.miR))

dataFilt <- TCGAanalyze_Filtering(
  tabDF = dataAssy.miR,
  method = "quantile", 
  qnt.cut =  0.25
)   


### metadata

samplesDown.miR <- getResults(query.miRNA,cols=c("cases"))

dataSmTP.miR <- TCGAquery_SampleTypes(barcode = samplesDown.miR,
                                      typesample = "TP")

dataSmNT.miR <- TCGAquery_SampleTypes(barcode = samplesDown.miR,
                                      typesample = "NT")


brcaMIR=as.data.frame(t(dataFilt))
brcaMIR$status <- ifelse(rownames(brcaMIR) %in% dataSmNT.miR, "Normal", "Tumor")
write.csv(brcaMIR, "TCGA-BRCA-miRNA.csv", row.names = TRUE)
