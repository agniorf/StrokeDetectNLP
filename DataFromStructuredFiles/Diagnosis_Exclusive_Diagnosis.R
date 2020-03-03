library(dplyr)
library(tidyverse)
library(icd)
require(devtools)
require(dplyr)
require(data.table)
library(icdcoder)
library(tibble)
library(touch)

setwd("~/Dropbox (Partners HealthCare)/MVP/RPDR/Cerebral Edema 2014-2018/Data/Raw/Structured")
dea <- read.table("TS100_20180206_125826_Dea.txt",sep="|", header=TRUE)
dia <- read.table("TS100_20180206_125826_Dia.txt",sep="|", header=TRUE,fill = TRUE, quote = "")

# We would like to see what are the diagnosis codes associated with dia files
# What are the types of diagnosis codes that we notice in the dataset
table(dia$Code_Type)
dia%>%
  filter(Code_Type=="Oncall") %>%
  select(Diagnosis_Name,Code)

# Question: how concentrated are the diagnosis codes that we notice?
# We note that ICD9 and ICD10 codes are the vast majority of the diagnosis codes identified. So in this part we will focus on those
d_icd9<- dia %>%
          filter( Code_Type == "ICD9")
length(unique(d_icd9$Code))
d_icd10<-dia %>%
          filter( Code_Type == "ICD10")

length(unique(d_icd10$Code))
length(unique(d_icd10$Code))+length(unique(d_icd9$Code))

ahrq_comorb<-icd9_comorbid_ahrq(d_icd9[,c("Code","Encounter_number")])
#Convert icd10 to icd9 
d_icd10$Code_icd9<-NA
for (i in 1:nrow(d_icd10)){
#for (i in 1:10) {
  d_icd10$Code_icd9[i]<-convICD(d_icd10$Code[i], "icd10")[2]
}

backup<-d_icd10
d_icd10<-backup
for (i in 1:nrow(d_icd10)) {
  if(length(unique(unlist(d_icd10$Code_icd9[i], use.names=FALSE)))>1){
    y <- unlist(d_icd10$Code_icd9[i], use.names=FALSE)
    d_icd10$Code_icd9[i]<-y[1]
  }
}
d_icd10$Code_icd9<-as.factor(unlist(d_icd10$Code_icd9))
#backup<-d_icd10
#for (i in 1:nrow(d_icd10)) {
#  if(length(unique(unlist(d_icd10$Code_icd9[i], use.names=FALSE)))>1){
#    y <- paste(unlist(d_icd10$Code_icd9[i], use.names=FALSE), sep='', collapse=', ')
#    d_icd10$Code_icd9[i]<-y
#  }
#}
d_icd9$Code_icd9<-icd_decimal_to_short(d_icd9$Code)

#We create a new diagnosis file that contains only the ICD observations all mapped to simplified ICD9
new_dia<-as.data.frame(rbind(d_icd9,d_icd10))

setwd("~/Dropbox (Partners HealthCare)/MVP/RPDR/Cerebral Edema 2014-2018/Data/Processed/Structured")
write.csv(new_dia,"new_dia.csv", row.names = FALSE)

# PRIMARY DIAGNOSIS CODES ANALYSIS
length(unique(new_dia$Code_icd9[which(new_dia$Diagnosis_Flag=="Primary")]))

prim_diag<-new_dia %>% 
                filter(Diagnosis_Flag == "Primary")%>%
                count(Code_icd9, sort = TRUE)

prim_diag %>%
      filter(n > 200) 
      
sum(prim_diag$n[which(prim_diag$n>100)])/nrow(new_dia %>% 
                                                filter(Diagnosis_Flag == "Primary"))

length(prim_diag$n[which(prim_diag$n>100)])


    

