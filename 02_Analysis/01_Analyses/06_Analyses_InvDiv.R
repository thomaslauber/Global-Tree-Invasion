#####################################
#####################################
###### GFBI INVASION ANALYSES #######
########### SPRICH ~ INV ############
#####################################
#####################################

# using repeat census data: determine species change due to invasion
# rerun orignial models correcting for percent change due to invasion (correcting nativect)

#####################################
###### Load packages; read dat ######
#####################################

library(tidyverse)
packageVersion("tidyverse")
library(lsmeans)
packageVersion("lsmeans")
library(ggplot2)
packageVersion("ggplot2")

#####################################
####### Repeat density plot #########
#####################################

repeat.final <- readRDS("data/GFBI_invasion_foranalyses_repeat.rds")
# determine median years
repeat.final.i <- repeat.final %>% filter(Finalinvstat==1)
medY <- median(repeat.final.i$Y)

#####################################
########### Repeat models ###########
#####################################

#####################################
########### deltaspecies ############
#####################################

repeat.mod <- glm(deltaspecies ~ Finalinvstat + dist.ports + popdensity + MAT + MAP + 
                    SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                    SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, 
                    data = repeat.final)
summary(repeat.mod)

#####################################
###### deltaspecies per year ########
#####################################

repeat.mod <- glm(deltaspecies.yr ~ Finalinvstat + dist.ports + popdensity + MAT + MAP + 
                    SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                    SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, 
                    data = repeat.final)
summary(repeat.mod)

#####################################
######### percent change ############
#####################################

repeat.mod <- glm(percentchange ~ Finalinvstat + dist.ports + popdensity + MAT + MAP + 
                    SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                    SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, 
                    data = repeat.final)
summary(repeat.mod)

# coefficient:
summary <- summary(repeat.mod)
coef <- summary$coefficients["Finalinvstat","Estimate"]
# lower 95th of coef: 
SE <- summary$coefficients["Finalinvstat","Std. Error"]
coef.ext.l <- coef - 2*SE
# higher 95th of coef: 
coef.ext.h <- coef + 2*SE

#####################################
###### percent change per year ######
#####################################

repeat.mod <- glm(percentchange.yr ~ Finalinvstat + dist.ports + popdensity + MAT + MAP + 
                    SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                    SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, data = repeat.final)
summary(repeat.mod)

#####################################
######### Full data models ##########
#### Corrected for inv -> sp rich ###
#####################################

#load data and mutate nativect based on repeat plot model
dat <- readRDS("data/GFBI_invasion_PDforanalyses.rds") %>%
  mutate(nativect.corr = nativect - (coef*nativect)) %>%
  mutate(nativect.corr.ext.l = nativect - (coef.ext.l*nativect)) %>%
  mutate(nativect.corr.ext.h = nativect - (coef.ext.h*nativect))

#scale
dat.scale <- dat %>% mutate_at(c("MAP","MAT","nat.faith","nat.mntd","lgsprich","SG_Absolute_depth_to_bedrock", "SG_Coarse_fragments_000cm", 
                                 "SG_Sand_Content_000cm", "SG_Silt_Content_000cm", "SG_Soil_pH_H2O_000cm",
                                 "SG_Clay_Content_000cm","CHELSA_exBIO_AridityIndex"), scale) 
dat.stat.mod <- dat.scale %>%                                                                                      # Start with dat
  mutate(instatus = ifelse(in.stat == "invaded", 1, 0))                                                            # Make instatus 0/1

#####################################
########### original model ##########
#####################################

mod.stat.sp <- glm(instatus ~ nativect +    
                          dist.ports + popdensity + MAT + MAP + 
                          SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                          SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, 
                          family = binomial,data = dat.stat.mod)
summary(mod.stat.sp)
rsq(mod.stat.sp, adj = TRUE)

#original coef
summary.o <- summary(mod.stat.sp)
coef.o <- summary.o$coefficients["nativect","Estimate"]

#####################################
######## corrected native ct ########
#####################################

mod.stat.sp.corr <- glm(instatus ~ nativect.corr +    
                     dist.ports + popdensity + MAT + MAP + 
                     SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                     SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, 
                     family = binomial,data = dat.stat.mod)
summary(mod.stat.sp.corr)
rsq(mod.stat.sp.corr, adj = TRUE)

#% change in nativect estimate/biotic resistance:
#corr coef
summary.c <- summary(mod.stat.sp.corr)
coef.c <- summary.c$coefficients["nativect.corr","Estimate"]

((coef.o-coef.c)/coef.o)*100 #6.707556% change

#####################################
#### extreme corrected native ct ####
################ low ################
#####################################

mod.stat.sp.corr.ext.l <- glm(instatus ~ nativect.corr.ext.l +    
                          dist.ports + popdensity + MAT + MAP + 
                          SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                          SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, 
                          family = binomial,data = dat.stat.mod)
summary(mod.stat.sp.corr.ext.l)
rsq(mod.stat.sp.corr.ext.l, adj = TRUE)

#% change in nativect estimate/biotic resistance:
#corr coef
summary.ce.l <- summary(mod.stat.sp.corr.ext.l)
coef.ce.l <- summary.ce.l$coefficients["nativect.corr.ext.l","Estimate"]

((coef.o-coef.ce.l)/coef.o)*100 #10.38395% change

#####################################
#### extreme corrected native ct ####
############### high ################
#####################################

mod.stat.sp.corr.ext.h <- glm(instatus ~ nativect.corr.ext.h +    
                                dist.ports + popdensity + MAT + MAP + 
                                SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                                SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, 
                              family = binomial,data = dat.stat.mod)
summary(mod.stat.sp.corr.ext.h)
rsq(mod.stat.sp.corr.ext.h, adj = TRUE)

#% change in nativect estimate/biotic resistance:
#corr coef
summary.ce.h <- summary(mod.stat.sp.corr.ext.h)
coef.ce.h <- summary.ce.h$coefficients["nativect.corr.ext.h","Estimate"]

((coef.o-coef.ce.h)/coef.o)*100 #2.716613% change
