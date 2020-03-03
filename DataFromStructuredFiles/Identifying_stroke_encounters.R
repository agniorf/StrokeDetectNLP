library(data.table)
library(dplyr)
library(ggplot2)
library(tidyverse)
require(devtools)
library(scales)

#Our goal is to track every unique patient ID and find the day of the stroke 
# with its encounter number along with the day of an edema and its encounter number.

#First we will start with the encounter numbers and try to see if all the 288 patients
#are there
length(unique(enc$EMPI))
#Perfect, we have everyone, let's see how we will track down the edema
edema_principl<-enc[enc$Principle_Diagnosis %like% "edema", ]
#Find all the encounters with edema in the diagnosis code
edema_admit<-enc[enc$Admitting_Diagnosis %like% "edema" | enc$Principle_Diagnosis %like% "edema" | enc$Diagnosis_1 %like% "edema" | enc$Diagnosis_2 %like% "edema" |enc$Diagnosis_3 %like% "edema" |enc$Diagnosis_4 %like% "edema" |enc$Diagnosis_5 %like% "edema" |enc$Diagnosis_6 %like% "edema" |enc$Diagnosis_7 %like% "edema" |enc$Diagnosis_8 %like% "edema" |enc$Diagnosis_9 %like% "edema" |enc$Diagnosis_10 %like% "edema", ]

cer_edema_admit<-enc[enc$Admitting_Diagnosis %like% "Cerebral edema" | enc$Principle_Diagnosis %like% "Cerebral edema" | enc$Diagnosis_1 %like% "Cerebral edema" | enc$Diagnosis_2 %like% "Cerebral edema" |enc$Diagnosis_3 %like% "Cerebral edema" |enc$Diagnosis_4 %like% "Cerebral edema" |enc$Diagnosis_5 %like% "Cerebral edema" |enc$Diagnosis_6 %like% "Cerebral edema" |enc$Diagnosis_7 %like% "Cerebral edema" |enc$Diagnosis_8 %like% "Cerebral edema" |enc$Diagnosis_9 %like% "Cerebral edema" |enc$Diagnosis_10 %like% "Cerebral edema", ]
length(unique(cer_edema_admit$EMPI))

cer_edema<-cer_edema_admit %>%
              group_by(EMPI)

length(unique(c(edema_principl$EMPI,edema_admit$EMPI)))
#We are missing 14 patients that had a cerebral edema but none of their diagnoses appears to indicate that
missing_EMPI<-unique(enc$EMPI)[!(unique(enc$EMPI)) %in% (unique(cer_edema_admit$EMPI))]

#Let's try to find where it was identified that missing_EMPI[1] had an edema
mis1<-enc[which(enc$EMPI==missing_EMPI[1]),]
x<- enc %>%
    filter(EMPI == missing_EMPI[1])

x$Admit_Date <- as.Date(x$Admit_Date, "%m/%d/%Y")
ggplot( data = x[which(year(x$Admit_Date)==2015),], aes(Admit_Date, Clinic_Name)) + geom_point() +scale_x_date(labels = date_format("%m-%Y"))
#We see that she had the stroke at that time but we have no record of cerebral edema yet.
x_2<-dia %>%
  filter(EMPI == missing_EMPI[1], Diagnosis_Name %like% "edema")
#We got it here!
edema_diag<-dia %>%
  filter(Diagnosis_Name %like% "Cerebral edema")
length(unique(edema_diag$EMPI))
#We got them all here, we have for every patient one encounter with diagnosis: Cerebral Edema
edema_diag_cl<-edema_diag%>%
    group_by(EMPI) %>% 
    slice(which.min(Date))

#Now we will build our reference matrix that will contain the day of the edema the type of the edema and the stroke 
# date.
df<-edema_diag_cl[,c("EMPI" ,"Date","Encounter_number","Code")]
names(df)<-c("EMPI" ,"edema_Date","edema_enc_num","edema_Code")

setwd("~/Dropbox (Partners HealthCare)/MVP/RPDR/Cerebral Edema 2014-2018/Data/Processed/Structured")
write.csv(df, "edema_dates_patients.csv", row.names = FALSE)

#Now let's find the first instance of a stroke for those patients in the diagnosis files
cer_occl_diag<-dia %>%
  filter(Diagnosis_Name %like% "Cerebral")%>%
  group_by(EMPI) %>%
  slice(which.min(Date))

df_2<-cer_occl_diag[,c("EMPI" ,"Date","Encounter_number","Code")]
names(df_2)<-c("EMPI" ,"cerebral_Date","cerebral_enc_num","cerebral_Code")

length(unique(cer_occl_diag$EMPI))
df_3<-merge(df,df_2, by = "EMPI")

df_3$date_diff_cer<-difftime(as.Date(as.character(df_3$edema_Date), "%m/%d/%Y"),as.Date(as.character(df_3$cerebral_Date), "%m/%d/%Y"), units = c("days"))

cer_diag<-dia %>%
  filter(Diagnosis_Name %like% "Cerebral")%>%
  group_by(EMPI) 

M_diag<-merge(cer_diag, df, by="EMPI")
M_diag$date_diff_cer<-difftime(as.Date(as.character(M_diag$edema_Date), "%m/%d/%Y"),as.Date(as.character(M_diag$Date), "%m/%d/%Y"), units = c("days"))

rel_diag<-M_diag %>%
    filter(abs(date_diff_cer) < 14, !Diagnosis_Name%like% "Cerebral edema") %>%
    group_by(EMPI, Date)%>%
    select(EMPI, Date, Diagnosis_Name, Code, Encounter_number, edema_Date, edema_enc_num, edema_Code, date_diff_cer)

ex<-rel_diag%>%
      group_by(EMPI) %>%
      summarise(max_date = max(date_diff_cer))

df$flag_sam<-0
df$flag_sam[which(df$EMPI %in% ex$EMPI[which(ex$max_date==0)])]<-1

df$strokeDate<-df$edema_Date
df$strokeDate[which(df$flag_sam==0)]<-NA

table(df$flag_sam)

classb<-df[which(df$flag_sam==0),]
length(unique(classb$EMPI))

classb_rec<-rel_diag %>%
            filter(EMPI %in% classb$EMPI)

ex1<-classb_rec%>%
  group_by(EMPI) %>%
  summarise(max_date = max(date_diff_cer))

ex1[1,]

classb_rec<-classb_rec[order(as.Date(classb_rec$Date, format="%d/%m/%Y")),]

classb_rec[which(classb_rec$EMPI==100461947),]

df[which(df$EMPI %in% c(100461947,109450172,111007276,108945688,103897458,104404450,104385018)),]
ex[which(ex$EMPI %in% c(100461947,109450172,111007276,108945688,103897458,104404450,104385018)),]
