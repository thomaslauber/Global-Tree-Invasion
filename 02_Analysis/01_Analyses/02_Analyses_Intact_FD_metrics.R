#####################################
#####################################
###### GFBI INVASION ANALYSES #######
######### INTACT SUBSET ONLY ########
#####################################
#####################################

# 'Status' = Non-native introduction status
#  Degree of non-native introduction
# 'Colonization' = Relative species richness 
# 'Spread' = Relative species abundance (basal area)

#####################################
########## Load packages ############
#####################################

library(feather)
packageVersion("feather")
library(tidyverse)
packageVersion("tidyverse")
library(lme4)
packageVersion("lme4")
library(lmerTest)
packageVersion("lmerTest")
library(betareg)
packageVersion("betareg")
library(ggeffects)
packageVersion("ggeffects")
library(ggplot2)
packageVersion("ggplot2")
library(gridExtra)
packageVersion("gridExtra")
library(grid) 
packageVersion("grid")
library(viridis)
packageVersion("viridis")
library(ClustOfVar)
packageVersion("ClustOfVar")
library(HH)
packageVersion('HH')
library(MuMIn)
packageVersion('MuMIn')
library(scales)
packageVersion('scales')
library(RColorBrewer)
packageVersion('RColorBrewer')
library(sp)
packageVersion('sp')
library(ncf)
packageVersion('ncf')
library(ape)
packageVersion('ape')
library(spdep)
packageVersion('spdep')
library(scales)
packageVersion('scales')

#####################################
########### Load functions ##########
#####################################

#spatial autocorrelation:
Spat.cor <- function(mod,dat,dist) {
  coords <- cbind(dat$avglon, dat$lat)
  matrix.dist = as.matrix(dist(cbind(dat$avglon, dat$lat)))
  matrix.dist[1:10, 1:10]
  matrix.dist.inv <- 1/matrix.dist
  matrix.dist.inv[1:10, 1:10]
  diag(matrix.dist.inv) <- 0
  matrix.dist.inv[1:10, 1:10]
  myDist = dist
  # calculate residuals autocovariate (RAC)
  rac <- autocov_dist(resid(mod), coords, nbs = myDist, type = "inverse", zero.policy = TRUE, style = "W", longlat=T)
  return(rac)
}

#####################################
######### Load and prep data ######## 
#####################################

dat <- readRDS("data/GFBI_invasion_intactFDforanalyses.rds")

#FOR MODELS; SCALE
dat.scale <- dat %>% mutate_at(c("MAP","MAT","nat.faith","nat.mntd","lgsprich","SG_Absolute_depth_to_bedrock", "SG_Coarse_fragments_000cm", 
                                                           "SG_Sand_Content_000cm", "SG_Silt_Content_000cm", "SG_Soil_pH_H2O_000cm",
                                                           "SG_Clay_Content_000cm","CHELSA_exBIO_AridityIndex"), scale) 

dat.stat.mod <- dat.scale %>%                                                                                      # Start with dat
  mutate(instatus = ifelse(in.stat == "invaded", 1, 0))                                                      # Make instatus 0/1

dat.prop.mod <- dat.scale %>%                                                                                      # Start with dat
  filter(propinvasive > 0)                                                                                   # Keep only propinvasive > 0

#FOR RAC
dat.noscale <- dat 

dat.stat.mod.noscale <- dat.noscale %>%                                                                                      # Start with dat
  mutate(instatus = ifelse(in.stat == "invaded", 1, 0))                                                      # Make instatus 0/1

dat.prop.mod.noscale <- dat.noscale %>%                                                                                      # Start with dat
  filter(propinvasive > 0)                                                                                   # Keep only propinvasive > 0

dat.prop.modb.noscale <- dat.prop.mod.noscale %>% filter(!pinvba == 1 & !pinvba == 0)

#####################################
########### Native models ###########
#####################################

#####################################
########### Native phylo ############
#####################################

#####################################
############## Status ###############
#####################################

# Global
mod.stat <- glm(instatus ~ nat.faith + nat.mntd +    
                  dist.ports + popdensity + MAT + MAP + 
                  SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                  SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, 
                family = binomial,data = dat.stat.mod)

######
rac <- Spat.cor(mod.stat,dat.stat.mod,250)
dat.stat.mod$rac <- rac
dat.stat.mod.noscale$rac <- rac
dat.stat.mod.noscale <- dat.stat.mod.noscale %>%
  mutate(MAT = MAT/10, MAP = MAP/10)
saveRDS(dat.stat.mod.noscale,"data/Intact_FDRACdf_noscale_global_dat.stat.mod.rds")

mod.stat <- glm(instatus ~ nat.faith + nat.mntd +    
                  dist.ports + popdensity + MAT + MAP + 
                  SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                  SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
                  family = binomial,data = dat.stat.mod)

summary(mod.stat)

#####################################
########### Colonization ############
#####################################

# Global
mod.col <- glm(cbind(invasct,nativect) ~ nat.faith + nat.mntd +    
                 dist.ports + popdensity + MAT + MAP + 
                 SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                 SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, 
               family = binomial, data = dat.prop.mod)

rac <- Spat.cor(mod.col,dat.prop.mod,250)
dat.prop.mod$rac <- rac
dat.prop.mod.noscale$rac <- rac
dat.prop.mod.noscale <- dat.prop.mod.noscale %>%
  mutate(MAT = MAT/10, MAP = MAP/10)
saveRDS(dat.prop.mod.noscale,"data/Intact_FDRACdf_noscale_global_dat.prop.mod.rds")

mod.col <- glm(cbind(invasct,nativect) ~ nat.faith + nat.mntd +    
                 dist.ports + popdensity + MAT + MAP + 
                 SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                 SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
                 family = binomial, data = dat.prop.mod)

summary(mod.col)

#####################################
############## Spread ###############
#####################################

# Global
dat.prop.mod.b <- dat.prop.mod %>% filter(!pinvba == 1 & !pinvba == 0)
mod.spread <- betareg(pinvba ~ nat.faith + nat.mntd +    
                        dist.ports + popdensity + MAT + MAP + 
                        SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                        SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, 
                      data = dat.prop.mod.b)

rac <- Spat.cor(mod.spread,dat.prop.mod.b,250)
dat.prop.mod.b$rac <- rac
dat.prop.modb.noscale$rac <- rac
dat.prop.modb.noscale <- dat.prop.modb.noscale %>%
  mutate(MAT = MAT/10, MAP = MAP/10)
saveRDS(dat.prop.modb.noscale,"data/Intact_FDRACdf_noscale_global_dat.prop.modb.rds")

mod.spread <- glm(cbind(invasct,nativect) ~ nat.faith + nat.mntd +    
                 dist.ports + popdensity + MAT + MAP + 
                 SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                 SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
               family = binomial, data = dat.prop.mod.b)

summary(mod.spread)
