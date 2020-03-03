library(dplyr)
library(ggplot2)
library(caTools)
library(caret)
library(ROCR)
library(data.table)
library(pROC)
library(calibrate)
library(gtable)
library(gbm)
library(grid)
library(gridExtra)
library(OptimalCutpoints)
library(cowplot)
# First we will aggregate the results per method

#We need to return: accuracy, sensitivity, specificity, auc for a given threshold
eval_performance <- function(predictTest, test, threshold){
  #Get the probability prediction
  confusion_mat = table(test[,1], predictTest > threshold)
  accuracy = sum(diag(confusion_mat))/sum(confusion_mat)
  
  predictTest_res = predictTest
  predictTest_res[which(predictTest > threshold)]=1
  
  test = mutate(test, predictTest_res = as.factor(ifelse(predictTest > threshold, 1, 0 )))
  if(ncol(confusion_mat)>1){
    sensitivity = confusion_mat[2,2] / sum(confusion_mat[2,])
    specificity = confusion_mat[1,1] / sum(confusion_mat[1,])
    precision = confusion_mat[2,2] / sum(confusion_mat[,2])
    recall = confusion_mat[2,2] / sum(confusion_mat[2,])
  }
  else{
    sensitivity = nrow(test[which(test[,1]==1 & predictTest > threshold),])/nrow(test[which(test[,1]==1),])
    specificity = nrow(test[which(test[,1]==0 & predictTest <= threshold),])/nrow(test[which(test[,1]==0),])
    recall = nrow(test[which(test[,1]==1 & predictTest > threshold),])/nrow(test[which(test[,1]==1),])
    precision = length(test[which(test[,1]==1 & predictTest > threshold),])/length(predictTest_res[which(predictTest_res==1)])
    if(threshold==1){precision=1}
    if(length(predictTest_res[which(predictTest_res==1)])==0){precision=1}
  }
  
  ROCpred = prediction(predictTest, test[,1])
  ROCperf = performance(ROCpred, "tpr", "fpr")
  
  auc = as.numeric(performance(ROCpred, "auc")@y.values)
  return(c(threshold, accuracy, sensitivity, specificity,recall,precision, auc))
}

tableofperformance <- function(predictTest, test, thresholds){
  #Create a table of results
  res <- data.frame(matrix(ncol = 7, nrow = length(thresholds)))
  x <- c("Threshold", "Accuracy", "Sensitivity", "Specificity", "Recall", "Precision","AUC")
  colnames(res) <- x
  
  i=1
  for (threshold in thresholds) {
    res[i,] = eval_performance(predictTest,test, threshold)
    i=i+1
  }
  
  return(res)
}

generate_tables<-function(thresholds, tasks,embeddings, algs,seeds){
  #Create tables for each seed and combination of parameters
  for (task in tasks) {
    for (embedding in embeddings) {
      for (s in seeds) {
        y_name = paste("clean_data/",task,"/y_test_",embedding,"_",s,".csv",sep="")
        y = read.table(y_name, quote="\"", comment.char="")
        for (alg in algs) {
          pred_name = paste("proba_",task,"/proba_",embedding,"_",alg,"_",s,".csv",sep="")
          predictTest = read.table(pred_name, quote="\"", comment.char="")[,1]
          
          table_name = paste("SensitivitySpecificityTables/tables_",task,"/tables_",embedding,"_",alg,"_",s,".csv",sep="")
          write.csv(tableofperformance(predictTest, y, thresholds), table_name, row.names = FALSE)
        }
      }
    }
  }
}

generate_table<-function(thresholds, task,embedding, alg,seed){
  #Create tables for each seed and combination of parameters
        y_name = paste("clean_data/",task,"/y_test_",embedding,"_",seed,".csv",sep="")
        y = read.table(y_name, quote="\"", comment.char="")
        pred_name = paste("proba_",task,"/proba_",embedding,"_",alg,"_",seed,".csv",sep="")
        predictTest = read.table(pred_name, quote="\"", comment.char="")[,1]
          
        return(tableofperformance(predictTest, y, thresholds))
}

aggregate_tables<-function(thresholds, tasks,embeddings, algs,seeds){
  #Create Aggregate Tables that are averaging the results for all 5 seeds
  for (task in tasks) {
    for (embedding in embeddings) {
      for (alg in algs) {
        #Create a table of results
        res <- data.frame(matrix(ncol = 7, nrow = length(thresholds)))
        x <- c("Threshold", "Accuracy", "Sensitivity", "Specificity", "Recall", "Precision", "AUC")
        colnames(res) <- x
        res$Threshold = thresholds
        
        #Create a list for each combination for all the seeds, aggregating the results
        t_list = list()
        for (s in seeds) {
          table_name = paste("SensitivitySpecificityTables/tables_",task,"/tables_",embedding,"_",alg,"_",s,".csv",sep="")
          t_list[[length(t_list)+1]] = read.csv(table_name)
        }
        agg_table_name = paste("SensitivitySpecificityTables/tables_",task,"/agg_tables_",embedding,"_",alg,".csv",sep="")
        agg = rbindlist(t_list)[,lapply(.SD,mean), list(Threshold)]
        write.csv(agg, agg_table_name, row.names = FALSE)
      }
    }
  }
}

roc_aggregate_tables<-function(thresholds, task,embedding, alg,seeds){
  #Create Aggregate Tables that are averaging the results for all 5 seeds
        #Create a table of results
        res <- data.frame(matrix(ncol = 7, nrow = length(thresholds)))
        x <- c("Threshold", "Accuracy", "Sensitivity", "Specificity", "Recall", "Precision", "AUC")
        colnames(res) <- x
        res$Threshold = thresholds
        
        #Create a list for each combination for all the seeds, aggregating the results
        t_list = list()
        for (s in seeds) {
          table_name = paste("SensitivitySpecificityTables/tables_",task,"/tables_",embedding,"_",alg,"_",s,".csv",sep="")
          t_list[[length(t_list)+1]] = generate_table(thresholds, task,embedding, alg,s)
        }
        agg = rbindlist(t_list)[,lapply(.SD,mean), list(Threshold)]
        return(agg)
}

create_roc_table<-function(thresholds, tasks,embeddings, algs,seeds){
  #Create Aggregate Tables that are averaging the results for all 5 seeds
  res <- data.frame(matrix(ncol = 10, nrow = 0))
  x <- c("Threshold", "Accuracy", "Sensitivity", "Specificity","Recall", "Precision", "AUC","method" ,"embedding","task")
  colnames(res) <- x
  for (task in tasks) {
    for (embedding in embeddings) {
      for (alg in algs) {
        t = roc_aggregate_tables(thresholds, task,embedding, alg,seeds)
        t$method = alg
        t$embedding = embedding
        t$task = task
        t$Threshold = thresholds
        
        res = rbind(res,t)
      }
    }
  }
  return(res)
}

create_mult_roc_curve<-function(res, task_to_plot){
  plot_name<-paste("Average ROC Curve over 5 seeds for ",task_to_plot,sep="")
  file_name = paste("ROC_curves/roc_curves_",task_to_plot,".png",sep="")
  
  ggplot(data=res%>%filter(task==task_to_plot), aes(x=1-Specificity, y=Sensitivity, color = method)) + facet_grid(.~embedding)+ 
    geom_line() +
    expand_limits(y=0) +
    xlab("1 - Specificity") + ylab("Sensitivity") +
    ggtitle(plot_name)
  ggsave(file_name, width = 18, height = 10,limitsize = FALSE)
}

create_general_roc_curve<-function(res){
  g1<-ggplot(data=res%>%filter(task=="stroke"), aes(x=1-Specificity, y=Sensitivity, color = method)) + facet_grid(.~embedding)+ 
    geom_line() +
    expand_limits(y=0) +
    xlab("1 - Specificity") + ylab("Sensitivity") +
    ggtitle("Stroke Presence")+ theme_light()+
    theme(strip.background = element_rect(colour=NA, fill=NA),text = element_text(size=14),
          panel.border = element_rect(color = "black"))+
    theme(strip.text = element_text(colour = 'black'))
  g2<-ggplot(data=res%>%filter(task=="location"), aes(x=1-Specificity, y=Sensitivity, color = method)) + facet_grid(.~embedding)+ 
    geom_line() +
    expand_limits(y=0) +
    xlab("1 - Specificity") + ylab("Sensitivity") +
    ggtitle("MCA Location")+ theme_light()+
    theme(strip.background = element_rect(colour=NA, fill=NA),text = element_text(size=14),
          panel.border = element_rect(color = "black"))+
    theme(strip.text = element_text(colour = 'black'))
  g3<-ggplot(data=res%>%filter(task=="acuity"), aes(x=1-Specificity, y=Sensitivity, color = method)) + facet_grid(.~embedding)+ 
    geom_line() +
    expand_limits(y=0) +
    xlab("1 - Specificity") + ylab("Sensitivity") +
    ggtitle("Acuity")+ theme_light()+
    theme(strip.background = element_rect(colour=NA, fill=NA),text = element_text(size=14),
          panel.border = element_rect(color = "black"))+
    theme(strip.text = element_text(colour = 'black'))
  g=grid.arrange(g1, g2, g3, nrow = 3)
  return(g)
}

create_figure_roc_curve_mixed<-function(res){
  g1<-ggplot(data=res%>%filter(task=="stroke" ,embedding == "BOW"), aes(x=1-Specificity, y=Sensitivity, color = method)) + 
    geom_line() +
    expand_limits(y=0) +
    xlab("1 - Specificity") + ylab("Sensitivity") + theme_light() +
    ggtitle("Stroke Presence with BOW")+ theme(legend.position = "none",text = element_text(size=18)) 
  g2<-ggplot(data=res%>%filter(task=="location",embedding == "TF-IDF"), aes(x=1-Specificity, y=Sensitivity, color = method)) + 
    geom_line() +
    expand_limits(y=0) +
    xlab("1 - Specificity") + ylab("Sensitivity") + theme_light()+
    ggtitle("MCA Location with TF-IDF")+ theme(legend.position = "none",text = element_text(size=18)) 
  g3<-ggplot(data=res%>%filter(task=="acuity" ,embedding == "GloVe"), aes(x=1-Specificity, y=Sensitivity, color = method)) + 
    geom_line() +
    expand_limits(y=0) +
    xlab("1 - Specificity") + ylab("Sensitivity") + theme_light()+theme(legend.position = "none",text = element_text(size=18))+
    ggtitle("Acuity with GloVe")
  g=grid.arrange(g1, g2, g3, nrow = 1)
  return(g)
}

create_legend<-function(res){
  g3<-ggplot(data=res%>%filter(task=="acuity" ,embedding == "GloVe"), aes(x=1-Specificity, y=Sensitivity, color = Method)) + 
    geom_line() +
    expand_limits(y=0) +
    xlab("1 - Specificity") + ylab("Sensitivity") + theme_light()+theme(text = element_text(size=18))+
    ggtitle("Acuity with GloVe")
  legend <- cowplot::get_legend(g3)
  grid.newpage()
  grid.draw(legend)
}

create_figure_roc_curve_glove<-function(res){
  g1<-ggplot(data=res%>%filter(task=="stroke" ,embedding == "GloVe"), aes(x=1-Specificity, y=Sensitivity, color = method)) + 
    geom_line() +
    expand_limits(y=0) +
    xlab("1 - Specificity") + ylab("Sensitivity") + theme_light() +
    ggtitle("Stroke Presence")+ theme(legend.position = "none",text = element_text(size=18)) 
  g2<-ggplot(data=res%>%filter(task=="location",embedding == "GloVe"), aes(x=1-Specificity, y=Sensitivity, color = method)) + 
    geom_line() +
    expand_limits(y=0) +
    xlab("1 - Specificity") + ylab("Sensitivity") + theme_light()+
    ggtitle("MCA Location")+ theme(legend.position = "none",text = element_text(size=18)) 
  g3<-ggplot(data=res%>%filter(task=="acuity" ,embedding == "GloVe"), aes(x=1-Specificity, y=Sensitivity, color = method)) + 
    geom_line() +
    expand_limits(y=0) +
    xlab("1 - Specificity") + ylab("Sensitivity") + theme_light()+theme(legend.position = "none",text = element_text(size=18))+
    ggtitle("Acuity")
  g=grid.arrange(g1, g2, g3, nrow = 1)
  return(g)
}

create_mult_precision_curve<-function(res, task_to_plot){
  plot_name<-paste("Average Precision Recall Curve over 5 seeds for ",task_to_plot,sep="")
  file_name = paste("Precision_Recall_curves/precision_recall_curves_",task_to_plot,".png",sep="")
  
  ggplot(data=res%>%filter(task==task_to_plot), aes(x=Recall, y=Precision, color = method)) + facet_grid(.~embedding)+ 
    geom_line() +
    expand_limits(y=0) +
    xlab("Recall") + ylab("Precision") +
    ggtitle(plot_name)
  ggsave(file_name, width = 18, height = 10,limitsize = FALSE)
}

#For every method and seed we will create the calibration table and then we will average them out
create_calibration_table<-function(task,alg,embedding,s){
  y_name = paste("clean_data/",task,"/y_test_",embedding,"_",s,".csv",sep="")
  y = read.table(y_name, quote="\"", comment.char="")
  pred_name = paste("proba_",task,"/proba_",embedding,"_",alg,"_",s,".csv",sep="")
  predictTest = read.table(pred_name, quote="\"", comment.char="")[,1]
  
  class_probs = data.frame(
    target = factor(y$V1),
    prediction = predictTest)
  
  cal_plot_data = calibration(target ~ prediction, 
                              data = class_probs, class = 1)$data
  return(cal_plot_data)
}

get_midpoint_percent<-function(alg, task, embedding){
  Percent = create_calibration_table(task,alg,embedding,1)[,"Percent"]
  midpoint = create_calibration_table(task,alg,embedding,1)[,"midpoint"]
  for (s in 2:5) {
    d = create_calibration_table(task,alg,embedding,s)
    midpoint = cbind(d[,"midpoint"],midpoint)
    Percent = cbind(d[,"Percent"],Percent)
  } 
  Percent = rowMeans(Percent, na.rm = FALSE, dims = 1)
  midpoint = rowMeans(midpoint, na.rm = FALSE, dims = 1)
  
  dat = as.data.frame(cbind(Percent,midpoint))
  dat$alg = alg
  dat$task = task
  dat$embedding = embedding
  dat$id = 1:nrow(dat)
  return(dat)
}

create_mult_calibration_curve<-function(df, task_to_plot,embed){
  plot_name<-paste("Calibration Curve over 5 seeds for ",task_to_plot," and the ", embed, " embedding.", sep="")
  file_name = paste("Calibration_curves/calibration_curves_",task_to_plot,"_", embed,".png",sep="")
  ggplot()  + xlab("Bin Midpoint")  +
    geom_line(data = df%>%filter(task==task_to_plot, embedding== embed), aes(midpoint, Percent,color = Algorithm)) +
    geom_point(data = df%>%filter(task==task_to_plot, embedding== embed), aes(midpoint, Percent,color = Algorithm), size = 3) +
    geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
              color = 'grey50')+
    ggtitle(plot_name)
  ggsave(file_name, width = 18, height = 10,limitsize = FALSE)
}

create_df_calibration<-function(tasks, embeddings,algs){
  #Calibration Curves
  alg = "CART"
  task = "stroke"
  embedding = "bow"
  df = get_midpoint_percent(alg, task, embedding)
  for (task in tasks) {
    for (alg in algs) {
      for (embedding in embeddings) {
         temp = get_midpoint_percent(alg, task, embedding)
         
         #Save the calibration Tables in a dedicated folder
         write_name = paste("CalibrationTables/table_",task,"_",alg,"_",embedding,".csv",sep = "")
         write.csv(temp, write_name)
         df = rbind(df,temp)
      }
    }
  }
  return(df)
  
}

thresholds = c(0.1,0.2, 0.3, 0.5, 0.9)

#Set a task
tasks = c("stroke","location","acuity")
embeddings = c("bow","tfidf","glove")
algs = c("CART","knn","logreg","RF","OCT","OCTH","lstm")
seeds = 1:5

#Generate Sensitivity / Specificity /Accuracy Tables for all the seeds
generate_tables(thresholds, tasks,embeddings, algs,seeds)
aggregate_tables(thresholds, tasks,embeddings, algs,seeds)

thresholds = seq(0,1,0.01)
res = create_roc_table(thresholds, tasks,embeddings, algs,seeds)
res$Precision[which(res$Precision==0 & res$Recall==0)]=1

res$method[which(res$method=="lstm")]="RNN"
res$method[which(res$method=="OCTH")]="OCT-H"
res$method[which(res$method=="knn")]="k-NN"
res$method[which(res$method=="logreg")]="Logistic.Regression"
res$embedding[which(res$embedding=="bow")]="BOW"
res$embedding[which(res$embedding=="tfidf")]="TF-IDF"
res$embedding[which(res$embedding=="glove")]="GloVe"

res$method<-factor(res$method, levels = c("Logistic.Regression","k-NN","CART","OCT","OCT-H","RF","RNN"))
res$embedding<-factor(res$embedding, levels = c("BOW","TF-IDF","GloVe"))

#ROC CURVES
create_mult_roc_curve(res,"stroke")
create_mult_roc_curve(res,"acuity")
create_mult_roc_curve(res,"location")

#One aggregated ROC curve
g = create_general_roc_curve(res)
#I would suggest you to open the image from the panel on the side and save it in the dimensions you would like

#ROC curves for illustration
create_figure_roc_curve_mixed(res)
create_figure_roc_curve_glove(res)
names(res)[8]<-"Method"
create_legend(res)

#Precision - Recall curves
create_mult_precision_curve(res,"stroke")
create_mult_precision_curve(res,"acuity")
create_mult_precision_curve(res,"location")

#Calibration Curves
df = create_df_calibration(tasks, embeddings,algs)
names(df) = c("Percent"  , "midpoint" , "Algorithm"   ,    "task"    ,  "embedding" ,"id"  )
for (task in tasks) {
  for (embedding in embeddings) {
    create_mult_calibration_curve(df, task,embedding)
  }
}

#Optimal Cut-off points
#We are interested only in the Glove - RNN and Stroke - Location - Acuity
df_stroke = read.csv("SensitivitySpecificityTables/tables_stroke/agg_tables_glove_lstm.csv")
df_location = read.csv("SensitivitySpecificityTables/tables_location/agg_tables_glove_lstm.csv")
df_acuity  = read.csv("SensitivitySpecificityTables/tables_acuity/agg_tables_glove_lstm.csv")

data(elas)