#####################################
#####################################
###### GFBI INVASION ANALYSES #######
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
library(rsq)
packageVersion('rsq')

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

dat <- readRDS("data/GFBI_invasion_FDforanalyses.rds")

#####################################
######### !!CHOOSE ONE: BIOME #######
#####################################

##FOR FIGURES
dat <- dat 
dat <- dat %>% filter(biome2=="Temperate")
dat <- dat %>% filter(biome2=="Tropical")

dat.stat.mod <- dat %>%                                                                                      # Start with dat
  mutate(instatus = ifelse(in.stat == "invaded", 1, 0))                                                      # Make instatus 0/1

dat.prop.mod <- dat %>%                                                                                      # Start with dat
  filter(propinvasive > 0)    

#FOR MODELS; SCALE
dat.scale <- dat %>% mutate_at(c("MAP","MAT","nat.faith","nat.mntd","lgsprich","SG_Absolute_depth_to_bedrock", "SG_Coarse_fragments_000cm", 
                                                           "SG_Sand_Content_000cm", "SG_Silt_Content_000cm", "SG_Soil_pH_H2O_000cm",
                                                           "SG_Clay_Content_000cm","CHELSA_exBIO_AridityIndex"), scale) 

dat.scale <- dat %>% filter(biome2=="Temperate") %>% mutate_at(c("MAP","MAT","nat.faith","nat.mntd","lgsprich","SG_Absolute_depth_to_bedrock", "SG_Coarse_fragments_000cm", 
                                                           "SG_Sand_Content_000cm", "SG_Silt_Content_000cm", "SG_Soil_pH_H2O_000cm",
                                                           "SG_Clay_Content_000cm","CHELSA_exBIO_AridityIndex"), scale)

dat.scale <- dat %>% filter(biome2=="Tropical") %>% mutate_at(c("MAP","MAT","nat.faith","nat.mntd","lgsprich","SG_Absolute_depth_to_bedrock", "SG_Coarse_fragments_000cm", 
                                                          "SG_Sand_Content_000cm", "SG_Silt_Content_000cm", "SG_Soil_pH_H2O_000cm",
                                                          "SG_Clay_Content_000cm","CHELSA_exBIO_AridityIndex"), scale)

dat.stat.mod <- dat.scale %>%                                                                                      # Start with dat
  mutate(instatus = ifelse(in.stat == "invaded", 1, 0))                                                      # Make instatus 0/1

dat.prop.mod <- dat.scale %>%                                                                                      # Start with dat
  filter(propinvasive > 0)                                                                                   # Keep only propinvasive > 0

#FOR RAC
dat.noscale <- dat 
dat.noscale <- dat %>% filter(biome2=="Temperate")
dat.noscale <- dat %>% filter(biome2=="Tropical")

dat.stat.mod.noscale <- dat.noscale %>%                                                                                      # Start with dat
  mutate(instatus = ifelse(in.stat == "invaded", 1, 0))                                                      # Make instatus 0/1

dat.prop.mod.noscale <- dat.noscale %>%                                                                                      # Start with dat
  filter(propinvasive > 0)                                                                                   # Keep only propinvasive > 0

dat.prop.modb.noscale <- dat.prop.mod.noscale %>% filter(!pinvba == 1 & !pinvba == 0)

#####################################
######### Colinearity check #########
#####################################

phylo.met <- dat %>%                                                                                         # Start with dat
  dplyr::select(c("nat.faith","nat.mntd","nat.mpd","nat.vntd","nat.vpd","lgsprich"))                                # Subset to cols of interest
phylo.cor <- as.matrix(cor(phylo.met))                                                                       # Create matrix to investigate colinearity

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
saveRDS(dat.stat.mod.noscale,"data/FDRACdf_noscale_global_dat.stat.mod.rds")

mod.stat <- glm(instatus ~ nat.faith + nat.mntd +    
                  dist.ports + popdensity + MAT + MAP + 
                  SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                  SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
                family = binomial,data = dat.stat.mod)

# Global Interaction
mod.stat.int <- glm(instatus ~ nat.faith*dist.ports + nat.mntd +    
                      popdensity + MAT + MAP + 
                      SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                      SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm, 
                    family = binomial,data = dat.stat.mod)

######
rac <- Spat.cor(mod.stat.int,dat.stat.mod,250)
dat.stat.mod$rac <- rac
mod.stat.int <- glm(instatus ~ nat.faith*dist.ports + nat.mntd +    
                      popdensity + MAT + MAP + 
                      SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                      SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
                    family = binomial,data = dat.stat.mod)

datplot <- ggeffect(mod.stat.int, terms = c("nat.faith","dist.ports"), type= "re")
png("figures/FD/dat.stat.int.jpg", width = 6, height = 6, units = 'in', res = 300)
plot(datplot)
dev.off()

# Temperate
mod.stat.temp <- glm(instatus ~ nat.faith + nat.mntd +    
                  dist.ports + popdensity + MAT + MAP + 
                  SG_Absolute_depth_to_bedrock + SG_Clay_Content_000cm + SG_Soil_pH_H2O_000cm,
                  family = binomial,data = dat.stat.mod)

rac <- Spat.cor(mod.stat.temp,dat.stat.mod,250)
dat.stat.mod$rac <- rac

mod.stat.temp <- glm(instatus ~ nat.faith + nat.mntd +    
                  dist.ports + popdensity + MAT + MAP + 
                  SG_Absolute_depth_to_bedrock + SG_Clay_Content_000cm + SG_Soil_pH_H2O_000cm +rac, 
                family = binomial,data = dat.stat.mod)

# Tropical
mod.stat.trop <- glm(instatus ~ nat.faith + nat.mntd +    
                  dist.ports + popdensity + MAT + MAP + 
                  SG_Absolute_depth_to_bedrock + SG_SOC_Content_000cm + SG_Soil_pH_H2O_000cm,
                family = binomial,data = dat.stat.mod)

rac <- Spat.cor(mod.stat.trop,dat.stat.mod,250)
dat.stat.mod$rac <- rac

mod.stat.trop <- glm(instatus ~ nat.faith + nat.mntd +    
                       dist.ports + popdensity + MAT + MAP + 
                       SG_Absolute_depth_to_bedrock + SG_SOC_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
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
saveRDS(dat.prop.mod.noscale,"data/FDRACdf_noscale_global_dat.prop.mod.rds")

mod.col <- glm(cbind(invasct,nativect) ~ nat.faith + nat.mntd +    
                 dist.ports + popdensity + MAT + MAP + 
                 SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                 SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
               family = binomial, data = dat.prop.mod)

# Temperate
mod.col.temp <- glm(cbind(invasct,nativect) ~ nat.faith + nat.mntd +    
                 dist.ports + popdensity + MAT + MAP + 
                 SG_Absolute_depth_to_bedrock + SG_Clay_Content_000cm + SG_Soil_pH_H2O_000cm, 
               family = binomial, data = dat.prop.mod)

rac <- Spat.cor(mod.col.temp,dat.prop.mod,250)
dat.prop.mod$rac <- rac

mod.col.temp <- glm(cbind(invasct,nativect) ~ nat.faith + nat.mntd +    
                 dist.ports + popdensity + MAT + MAP + 
                   SG_Absolute_depth_to_bedrock + SG_Clay_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
               family = binomial, data = dat.prop.mod)

# Tropical
mod.col.trop <- glm(cbind(invasct,nativect) ~ nat.faith + nat.mntd +    
                 dist.ports + popdensity + MAT + MAP + 
                 SG_Absolute_depth_to_bedrock + SG_SOC_Content_000cm + SG_Soil_pH_H2O_000cm,
               family = binomial, data = dat.prop.mod)

rac <- Spat.cor(mod.col.trop,dat.prop.mod,250)
dat.prop.mod$rac <- rac

mod.col.trop <- glm(cbind(invasct,nativect) ~ nat.faith + nat.mntd +    
                      dist.ports + popdensity + MAT + MAP + 
                      SG_Absolute_depth_to_bedrock + SG_SOC_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
                    family = binomial, data = dat.prop.mod)

summary(mod.col)
anova(mod.col,test = "LRT")

#####################################
############## Spread ###############
#####################################

dat.prop.mod.b <- dat.prop.mod %>% filter(!pinvba == 1 & !pinvba == 0)
# Global
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
saveRDS(dat.prop.modb.noscale,"data/FDRACdf_noscale_global_dat.prop.modb.rds")

mod.spread <- glm(cbind(invasct,nativect) ~ nat.faith + nat.mntd +    
                 dist.ports + popdensity + MAT + MAP + 
                 SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                 SG_Sand_Content_000cm + SG_Silt_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
               family = binomial, data = dat.prop.mod.b)


# Temperate
mod.spread.temp <- betareg(pinvba ~ nat.faith + nat.mntd +    
                        dist.ports + popdensity + MAT + MAP + 
                        SG_Absolute_depth_to_bedrock + SG_Clay_Content_000cm + SG_Soil_pH_H2O_000cm, 
                      data = dat.prop.mod.b)

rac <- Spat.cor(mod.spread.temp,dat.prop.mod.b,250)
dat.prop.mod.b$rac <- rac

mod.spread.temp <- glm(cbind(invasct,nativect) ~ nat.faith + nat.mntd +    
                    dist.ports + popdensity + MAT + MAP + 
                      SG_Absolute_depth_to_bedrock + SG_Clay_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
                  family = binomial, data = dat.prop.mod.b)

# Tropical
mod.spread.trop <- betareg(pinvba ~ nat.faith + nat.mntd +    
                        dist.ports + popdensity + MAT + MAP + 
                        SG_Absolute_depth_to_bedrock + SG_SOC_Content_000cm + SG_Soil_pH_H2O_000cm,
                      data = dat.prop.mod.b)

rac <- Spat.cor(mod.spread.trop,dat.prop.mod.b,250)
dat.prop.mod.b$rac <- rac

mod.spread.trop <- glm(cbind(invasct,nativect) ~ nat.faith + nat.mntd +    
                         dist.ports + popdensity + MAT + MAP + 
                         SG_Absolute_depth_to_bedrock + SG_SOC_Content_000cm + SG_Soil_pH_H2O_000cm + rac, 
                       family = binomial, data = dat.prop.mod.b)

summary(mod.spread)

#####################################
######## Invasion Strategy ##########
#####################################

dat.prop.mod <- dat.prop.mod %>% mutate(MAT = MAT/10, MAP = MAP/10)
# Global
mod.glob <- glm(d.mntd ~  MAP +poly(MAT,2,raw = TRUE) + dist.ports + nat.faith + popdensity+
                  SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                  SG_Sand_Content_000cm + SG_Soil_pH_H2O_000cm,
                data= dat.prop.mod)

rac <- Spat.cor(mod.glob,dat.prop.mod,250)
dat.prop.mod$rac <- rac
dat.prop.mod.noscale$rac <- rac
dat.prop.mod.noscale <- dat.prop.mod.noscale %>%
  mutate(MAT = MAT/10, MAP = MAP/10)
saveRDS(dat.prop.mod.noscale,"data/FDRACdf_noscale_dmntd_global_dat.prop.mod.rds")

mod.glob <- glm(d.mntd ~ MAP+ poly(MAT,2,raw = TRUE) + dist.ports + nat.faith + popdensity+
                  SG_Absolute_depth_to_bedrock + SG_Coarse_fragments_000cm +
                  SG_Sand_Content_000cm + SG_Soil_pH_H2O_000cm +rac,
                data= dat.prop.mod)

# Temperate
#full version
#mod.temp <- glm(d.mntd ~  MAP*MAT+MAP*dist.ports+ MAT*dist.ports+ nat.faith +  popdensity+
#                  SG_Absolute_depth_to_bedrock + SG_Clay_Content_000cm + SG_Soil_pH_H2O_000cm,
#              data= dat.prop.mod)
#simplified full version
mod.temp <- glm(d.mntd ~  MAT+MAP*dist.ports+ nat.faith +  popdensity+
                  SG_Absolute_depth_to_bedrock + SG_Clay_Content_000cm + SG_Soil_pH_H2O_000cm,
                data= dat.prop.mod)

rac <- Spat.cor(mod.temp,dat.prop.mod,250)
dat.prop.mod$rac <- rac
dat.prop.mod.noscale$rac <- rac
dat.prop.mod.noscale <- dat.prop.mod.noscale %>%
  mutate(MAT = MAT/10, MAP = MAP/10)
saveRDS(dat.prop.mod.noscale,"data/FDRACdf_noscale_dmntd_temp_dat.prop.mod.rds")

mod.temp <- glm(d.mntd ~  MAT+MAP*dist.ports+ nat.faith +  popdensity+
                  SG_Absolute_depth_to_bedrock + SG_Clay_Content_000cm + SG_Soil_pH_H2O_000cm+rac,
                data= dat.prop.mod)

# Tropical
#full version
#mod.trop <- glm(d.mntd ~  MAP*MAT+MAP*dist.ports+ MAT*dist.ports+nat.faith+   popdensity+
#                  SG_Absolute_depth_to_bedrock + SG_SOC_Content_000cm  + SG_Soil_pH_H2O_000cm,
#               data=dat.prop.mod)
#simplified full version
mod.trop <- glm(d.mntd ~  MAP*MAT+nat.faith+   dist.ports+popdensity+
                  SG_Absolute_depth_to_bedrock + SG_SOC_Content_000cm  + SG_Soil_pH_H2O_000cm,
                data=dat.prop.mod)

rac <- Spat.cor(mod.trop,dat.prop.mod,250)
dat.prop.mod$rac <- rac
dat.prop.mod.noscale$rac <- rac
dat.prop.mod.noscale <- dat.prop.mod.noscale %>%
  mutate(MAT = MAT/10, MAP = MAP/10)
saveRDS(dat.prop.mod.noscale,"data/FDRACdf_noscale_dmntd_trop_dat.prop.mod.rds")

mod.trop <- glm(d.mntd ~  MAP*MAT+nat.faith+  dist.ports+ popdensity+
                  SG_Absolute_depth_to_bedrock + SG_SOC_Content_000cm  + SG_Soil_pH_H2O_000cm+rac,
                data= dat.prop.mod)

mod <- mod.glob
mod <- mod.temp
mod <- mod.trop

summary(mod)
rsq(mod)

par(mfrow=c(1,2))
plot(mod)

#check relationships
datplot<-ggeffect(mod, terms= c("MAT","dist.ports"))
plot(datplot,raw=FALSE)

#####################################
###### Spatial Autocorrelation ###### 
#####################################

# Global
my_comb <- dat.prop.mod %>%
  mutate(ypred = predict(mod)) %>%
  mutate(resids = d.mntd - ypred) 

sp <- ncf::spline.correlog(x = as.numeric(my_comb$avglon),
                                  y = as.numeric(my_comb$lat),
                                  z = as.numeric(my_comb$resids),
                                  xmax = 500, resamp = 100, latlon=TRUE) 

sp.global <- sp
saveRDS(sp.global,"data/sp.global.strat.spline.rds")

sp.temp <- sp
saveRDS(sp.temp,"data/sp.temp.strat.spline.rds")

sp.trop <- sp
saveRDS(sp.trop,"data/sp.trop.strat.spline.rds")

png("figures/spat.auto_phylo.strat.jpg", width = 6, height = 10, units = 'in', res = 300)
par(mfrow = c(3, 1))
plot(sp.global,main="global", ylim=c(-0.25, 0.25))
plot(sp.temp,main="temp", ylim=c(-0.25, 0.25))
plot(sp.trop,main="trop", ylim=c(-0.25, 0.25))
dev.off()

#####################################
########### Color scale #############
#####################################

display.brewer.pal(n = 3, name = 'Pastel1')
brewer.pal(n = 3, name = "Pastel1") #"#FBB4AE" "#B3CDE3" "#CCEBC5"

colScale <- scale_colour_manual(values =c ("chartreuse4","darkcyan","grey46"))
fillScale <- scale_fill_manual(values =c ("chartreuse4","darkcyan","grey46"))

#####################################
############### Plot ################
#####################################

dp.nat.faith <- ggeffect(mod, terms = c("nat.faith [all]")) %>% 
  mutate(par = "nat.faith")

dp.dist.ports <- ggeffect(mod, terms = c("dist.ports [all]")) %>% 
  mutate(par = "dist.ports")

dp.popdensity <- ggeffect(mod, terms = c("popdensity [all]")) %>% 
  mutate(par = "popdensity")

dp.MAT <- ggeffect(mod, terms = c("MAT [all]")) %>% 
  mutate(par = "MAT")

dp.MAP <- ggeffect(mod, terms = c("MAP [all]")) %>% 
  mutate(par = "MAP")

datplot <- rbind(dp.nat.faith, dp.dist.ports, dp.popdensity, dp.MAT, dp.MAP)
diff.faith <- 
  ggplot(data = dp.nat.faith,aes(x = x, y = predicted))+
  geom_line() +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = .1) + 
  ylab("Invasion Strategy")+
  xlab("Native Phylogenetic Richness")+
  theme_classic(base_size = 10)+
  geom_hline(yintercept = 0, linetype = "dashed")+
  #theme(axis.title.x = element_blank(),
  #      axis.title.y = element_blank())+
  ylim(-0.5,0.5)

diff.MAT <- ggplot(data = dp.MAT,aes(x = x, y = predicted))+
  geom_line() +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = .1) + 
  ylab(" ")+
  xlab("MAT")+
  theme_classic(base_size = 10) +
  geom_hline(yintercept = 0, linetype = "dashed")+
  #theme(axis.title.x = element_blank(),
  #      axis.title.y = element_blank())+
  ylim(-0.5,0.5)

diff.MAP <- ggplot(data = dp.MAP,aes(x = x, y = predicted))+
  geom_line() +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = .1) + 
  ylab(" ")+
  xlab("MAP")+
  theme_classic(base_size = 10) +
  geom_hline(yintercept = 0, linetype = "dashed")+
  #theme(axis.title.x = element_blank(),
  #      axis.title.y = element_blank())+
  ylim(-0.5,0.5)

diff.dist.ports <- ggplot(data = dp.dist.ports,aes(x = x, y = predicted))+
  geom_line() +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = .1) + 
  ylab(" ")+
  xlab("Distance to ports")+
  theme_classic(base_size = 10) +
  geom_hline(yintercept = 0, linetype = "dashed")+
  #theme(axis.title.x = element_blank(),
  #      axis.title.y = element_blank())+
  ylim(-0.5,0.5)

grid.arrange(diff.faith,diff.MAT,diff.MAP,diff.dist.ports,ncol=4)

diff.all<- ggplot(data = datplot,aes(x = x, y = predicted))+
  geom_line() +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = .1) + 
  ylab("Invasion Strategy")+
  xlab("Variable")+
  theme_classic(base_size = 10)+
  facet_grid(cols=vars(par),
             scales = "free")

diff.all

#####################################
########### MAT (global) ############
#####################################

dp.MAT <- ggeffect(mod, terms = c("MAT [all]")) %>% 
  mutate(par = "MAT")

resid.resp <- residuals.glm(mod, type ="response")
pred.for.resid <- predict.glm(mod,type="response")

resid <- dat.prop.mod %>%
  mutate(resid = resid.resp, pred = pred.for.resid) %>%
  mutate(resid.d.mntd = resid+pred) 

resid$biome2 <- factor(resid$biome2, levels = c("Temperate","Tropical","Other"))

diff.MAT_resid <-
  ggplot(data = resid,aes(x = MAT, y = resid.d.mntd, color=biome2))+
  geom_point(alpha = 0.35, size =2.5)+
  xlab("Mean annual temperature (\u00B0C)")+
  ylab("Invasion Strategy")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  theme_classic(base_size = 20) +
  colScale+
  fillScale+
  theme(legend.position = "none", 
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 15))+
  guides(color = guide_legend("Bioclimatic Zone"))+
  ylim(-1.5,1.5)+
  geom_line(data = dp.MAT,aes(x = x, y = predicted),color="grey46",size=1.5) +
  geom_ribbon(data = dp.MAT,aes(y=NULL,x=x,ymin = conf.low, ymax = conf.high), alpha = .2,color=NA) +
  scale_y_continuous(labels = label_number(accuracy = 0.1),limits=c(-0.6,0.6))

png("figures/FD/Global_MAT.jpg", width = 6, height = 6, units = 'in', res = 300)
diff.MAT_resid
dev.off()

#determine extremes (d.mntd < 0)
quantile(resid$MAT, 0.005)
quantile(resid$MAT, 0.995)
resid.lowtemp <- resid %>% filter(MAT < 3.252617 & d.mntd < 0) %>% select(lat, avglon, MAT) %>% mutate(extreme = "low")
resid.hightemp <- resid %>% filter(MAT > 26.15517 & d.mntd < 0) %>% select(lat, avglon, MAT) %>% mutate(extreme = "high")

resid.ext <- rbind(resid.lowtemp,resid.hightemp)
saveRDS(resid.ext, "data/FDstrategy_tempextremes.rds")

###################################
########### MAP (global) ############
#####################################

dp.MAP <- ggeffect(mod, terms = c("MAP [all]")) %>% 
  mutate(par = "MAP")

resid.resp <- residuals.glm(mod, type ="response")
pred.for.resid <- predict.glm(mod,type="response")

resid <- dat.prop.mod %>%
  mutate(resid = resid.resp, pred = pred.for.resid) %>%
  mutate(resid.d.mntd = resid+pred) 

diff.MAP_resid <-
  ggplot(data = resid,aes(x = MAP, y = resid.d.mntd, color=biome2))+
  geom_point(alpha = 0.5)+
  xlab("Mean annual precipitation")+
  ylab("Intoduction Strategy")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  theme_classic(base_size = 20) +
  colScale+
  fillScale+
  theme(legend.position = c(.97, .97),legend.justification = c("right", "top"),legend.title = element_blank())+
  ylim(-1,1)+
  geom_line(data = dp.MAP,aes(x = x, y = predicted),color="black") +
  geom_ribbon(data = dp.MAP,aes(y=NULL,x=x,ymin = conf.low, ymax = conf.high), alpha = .2,color = "black")

png("figures/FD/Global_MAP.jpg", width = 6, height = 6, units = 'in', res = 300)
diff.MAP_resid
dev.off()

#####################################
###### dist.ports * MAP (precip) ######
#####################################
cut_tol<-0.1

display.brewer.pal(n = 3, name = 'Dark2')
brewer.pal(n = 3, name = "Dark2") #"#1B9E77" "#D95F02" "#7570B3"

colScale <- scale_colour_manual(values =c ("mistyrose", "#FBB4AE", "lightcoral"))
fillScale <- scale_fill_manual(values =c ("mistyrose", "#FBB4AE", "lightcoral"))

dist.ports.minmax <- data.frame(dist.ports = quantile(mod$data$dist.ports, c(cut_tol,0.5, 1-cut_tol)), level =c ("low","medium", "high")) 
MAP.range <- seq(from = quantile(dat.prop.mod$MAP, 0.001),to= quantile(dat.prop.mod$MAP, 0.999),length.out = 100) 
ex.grid <- expand.grid(MAP = MAP.range, dist.ports = dist.ports.minmax$dist.ports) 
pred.dat <- cbind(ex.grid,t(colMeans(na.omit(dat.prop.mod[,c("nat.faith", "MAT", "dist.ports", "popdensity",
                                                               "SG_Absolute_depth_to_bedrock", "SG_Clay_Content_000cm", "SG_Soil_pH_H2O_000cm","rac")])))) 
pred <- predict.glm(mod,type="response",newdata = pred.dat,se=TRUE)

pdat <- cbind(ex.grid,pred) %>% 
  left_join(dist.ports.minmax)

pdat.low <- pdat %>% filter(level=="low")
pdat.mid <- pdat %>% filter(level=="medium")
pdat.high <- pdat%>% filter(level=="high")

resid.resp <- residuals.glm(mod, type ="response")
pred.for.resid <- predict.glm(mod,type="response")

resid <- dat.prop.mod %>%
  mutate(resid = resid.resp, pred = pred.for.resid) %>%
  mutate(resid.d.mntd = resid+pred) 

resid.low <- resid %>% filter(dist.ports < dist.ports.minmax[1,1])
resid.high <- resid %>% filter(dist.ports < dist.ports.minmax[3,1])

diff.MAP_dist.ports_resid <-
  ggplot(data = resid,aes(x = MAP, y = resid.d.mntd, color=dist.ports))+
  geom_point(alpha = 0.35, size =2.5)+
  xlab("Mean Annual Precipitation")+
  ylab("Invasion Strategy")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  scale_color_viridis(option= "rocket",direction=-1)+
  theme_classic(base_size = 20)+
  theme(legend.position = c(.99, .99),legend.justification = c("right", "top"), 
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 15))+
  guides(color = guide_legend("Port distance (km)"))+
  xlim(50,250) +
  geom_line(data = pdat.low, aes(x = MAP, y = fit),color="#F6A47BFF") +
  geom_ribbon(data = pdat.low,aes(y=NULL,x=MAP,ymin = fit-se.fit, ymax = fit+se.fit), fill = "#F6A47BFF", alpha = .6,color = NA)+
  geom_line(data = pdat.high, aes(x = MAP, y = fit),color="#701F57FF") +
  geom_ribbon(data = pdat.high,aes(y=NULL,x=MAP,ymin = fit-se.fit, ymax = fit+se.fit), fill = "#701F57FF", alpha = .4,color = NA) +
  scale_y_continuous(labels = label_number(accuracy = 0.1),limits=c(-0.5,0.75))

png("figures/FD/Temperate_MAP*dist.ports_wpoints.jpg", width = 6, height = 6, units = 'in', res = 300)
diff.MAP_dist.ports_resid 
dev.off()

diff.MAP_dist.ports_resid.low <-
  ggplot(data = resid.low, aes(x = MAP, y = resid.d.mntd, color = "#F6A47BFF"))+
  geom_point(alpha = 0.35, size =2.5, color = "#F6A47BFF")+
  xlab("Mean annual precipitation")+
  ylab("Invasion Strategy")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  theme_classic(base_size = 20)+
  ylim(-0.5, 0.5) +
  xlim(0, 250) +
  theme(legend.position = "none", 
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 15))+
  geom_ribbon(data = pdat.low,aes(y=NULL,x=MAP,ymin = fit-se.fit, ymax = fit+se.fit), fill = "#F6A47BFF", alpha = 0.8,color = NA)+
  geom_line(data = pdat.low, aes(x = MAP, y = fit), color = "black", size = 1.5, alpha = 1) +
  scale_y_continuous(labels = label_number(accuracy = 0.1),limits=c(-0.5,0.75))

png("figures/FD/Temperate_MAP*dist.ports_wpoints.low.jpg", width = 6, height = 6, units = 'in', res = 300)
diff.MAP_dist.ports_resid.low
dev.off()

diff.MAP_dist.ports_resid.high <-
ggplot(data = resid.high, aes(x = MAP, y = resid.d.mntd, color = "#701F57FF"))+
  geom_point(alpha = 0.35, size = 2.5, color = "#701F57FF")+
  xlab("Mean annual precipitation")+
  ylab("Invasion Strategy")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  theme_classic(base_size = 20)+
  ylim(-0.5, 0.5) +
  xlim(0, 250) +
  theme(legend.position = "none", 
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 15))+
  geom_ribbon(data = pdat.high,aes(y = NULL,x = MAP,ymin = fit-se.fit, ymax = fit + se.fit), fill = "#701F57FF", alpha = 0.8,color = NA)+
  geom_line(data = pdat.high, aes(x = MAP, y = fit), color = "black", size = 1.5, alpha = 1)  +
  scale_y_continuous(labels = label_number(accuracy = 0.1),limits=c(-0.5,0.75))

png("figures/FD/Temperate_MAP*dist.ports_wpoints.high.jpg", width = 6, height = 6, units = 'in', res = 300)
diff.MAP_dist.ports_resid.high
dev.off()

#####################################
##### Forest plots (Temp/Trop) ######
#####################################

colScale <- scale_colour_manual(values =c ("chartreuse4","darkcyan"))
fillScale <- scale_fill_manual(values =c ("chartreuse4","darkcyan"))

mod.temp.est<-coef(summary(mod.temp))[, "Estimate"] %>% as.data.frame()
mod.temp.stderr<-coef(summary(mod.temp))[, "Std. Error"] %>% as.data.frame()
mod.temp.dat<- cbind(mod.temp.est,mod.temp.stderr) 
colnames(mod.temp.dat)<- c("est","std.err")
mod.temp.dat<- mod.temp.dat %>% mutate(mod = "temperate") %>% 
  rownames_to_column("variable") %>%
  filter(variable == "MAT" | variable == "MAP" | variable == "dist.ports" | variable == "nat.faith" | variable == "popdensity"| variable == "SG_Absolute_depth_to_bedrock"| variable == "SG_Soil_pH_H2O_000cm")

mod.trop.est<-coef(summary(mod.trop))[, "Estimate"] %>% as.data.frame()
mod.trop.stderr<-coef(summary(mod.trop))[, "Std. Error"] %>% as.data.frame()
mod.trop.dat<- cbind(mod.trop.est,mod.trop.stderr) 
colnames(mod.trop.dat)<- c("est","std.err")
mod.trop.dat<- mod.trop.dat %>% mutate(mod = "tropical") %>%
  rownames_to_column("variable") %>%
  filter(variable == "MAT" | variable == "MAP" | variable == "dist.ports" | variable == "nat.faith" | variable == "popdensity"| variable == "SG_Absolute_depth_to_bedrock"| variable == "SG_Soil_pH_H2O_000cm")

forest.dat <- rbind(mod.temp.dat,mod.trop.dat)

forest.dat <- forest.dat %>%
  mutate(variable2 = case_when(variable=="MAP" ~ "Mean annual precipitation",
                               variable=="MAT" ~ "Mean annual temperature", 
                               variable=="dist.ports" ~ "Distance to ports",
                               variable=="nat.faith" ~ "Native phylogenetic diversity",
                               variable=="popdensity" ~ "Population density",
                               variable=="SG_Absolute_depth_to_bedrock" ~ "Absolute bedrock depth",
                               variable=="SG_Soil_pH_H2O_000cm" ~ "Soil pH"))

forest.dat$variable2 <- factor(forest.dat$variable2, levels = c("Absolute bedrock depth", "Soil pH", "Population density","Distance to ports","Native phylogenetic diversity","Mean annual precipitation","Mean annual temperature"))

forest.plot <-
  ggplot(data=forest.dat, aes(x=variable2, y=est, ymin=est-std.err, ymax=est+std.err, color = mod)) +
  geom_pointrange(alpha = .8) + 
  geom_hline(yintercept=0, lty=2, color='darkgrey') +  # add a dotted line at x=1 after flip
  coord_flip() +  # flip coordinates (puts labels on y axis)
  xlab(" ") + ylab("Model Estimate") +
  colScale+
  fillScale+
  theme_classic(base_size = 20)+
  theme(legend.position = "none")+
  theme(axis.text.y = element_text(angle = 0))

png("figures/FD/diff_forestplot.jpg", width = 8, height = 6, units = 'in', res = 300)
forest.plot
dev.off()

