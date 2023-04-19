#####################################
########## Load packages ############
#####################################

library(tidyverse)
packageVersion("tidyverse")

#####################################
######### Load and prep data ######## 
#####################################

####!!! CHOOSE ONE
#Non down-sampled data:
dat <- readRDS("data/GFBI_invasion_foranalyses_allplots.rds") #471888

#Down-sampled data:
dat.pd <- readRDS("data/GFBI_invasion_PDforanalyses.rds")  # 17271
dat.fd <- readRDS("data/GFBI_invasion_FDforanalyses.rds")  # 17640

#Joined downsampled data #17738
dat <- rbind(dat.pd, dat.fd) %>%
  distinct(lat, avglon, .keep_all = TRUE)  

#####################################
########### Summary Stats ###########
#####################################

#each biome
#across all
dat %>% group_by(biome) %>% tally(in.stat == "invaded")
dat %>% group_by(biome) %>% tally(in.stat == "non-invaded")

#within invaded
dat %>% group_by(biome) %>% filter(propinvasive > 0) %>% summarize(meanpropinv=mean(propinvasive)) 

#temp v trop
#across all
dat %>% group_by(biome2) %>% tally(in.stat == "invaded")
dat %>% group_by(biome2) %>% tally(in.stat == "non-invaded")

#within invaded
dat %>% group_by(biome2) %>% filter(propinvasive > 0) %>% summarize(meanpropinv = mean(propinvasive)) 

#all plots
#across all
dat %>% tally(in.stat == "invaded")
dat %>% tally(in.stat == "non-invaded")

#within invaded
dat %>% filter(propinvasive > 0) %>% summarize(meanpropinv = mean(propinvasive)) 

#####################################
######### Freq of invaders ##########
########### Non-rarefied ############
#####################################

sp <- read.csv("data/GFBI_biome_rarefaction_invasives.csv") %>%                                               # Read condensed GFBI 
  #filter(keep_rarefied == 1) %>%                                                                             # Subset to rarefied data
  filter(Glonaf_collapsed_status == "invasive") %>%                                                           # Subset to invasives
  select(c("plot_id", "accepted_bin", "avglat", "avglon", "Glonaf_collapsed_status")) %>%                     # Subset to cols of interest
  distinct(accepted_bin, avglat, avglon, .keep_all = TRUE) %>%                                                # Drop duplicates of species and lat/lon                                                                              
  as.data.frame()                                                                                             # Make a df

inv.sp.com <- sp %>%                                                                                          # Start with sp
  group_by(accepted_bin) %>%
  summarize(invfreq = length(plot_id)) %>%
  arrange(desc(invfreq))

write.csv(inv.sp.com,"data/Summary_GlobalInvasion_allplots.csv")

#####################################
######### Freq of invaders ##########
############## Rarefied #############
#####################################

sp.r <- read.csv("data/GFBI_biome_rarefaction_invasives.csv") %>%                                             # Read condensed GFBI 
  filter(keep_rarefied == 1) %>%                                                                              # Subset to rarefied data
  filter(Glonaf_collapsed_status == "invasive") %>%                                                           # Subset to invasives
  select(c("plot_id", "accepted_bin", "avglat", "avglon", "Glonaf_collapsed_status")) %>%                     # Subset to cols of interest
  distinct(accepted_bin, avglat, avglon, .keep_all = TRUE) %>%                                                # Drop duplicates of species and lat/lon                                                                              
  as.data.frame()                                                                                             # Make a df

inv.sp.com.r <- sp.r %>%                                                                                      # Start with sp
  group_by(accepted_bin) %>%
  summarize(invfreq = length(plot_id)) %>%
  arrange(desc(invfreq))

write.csv(inv.sp.com.r,"data/Summary_GlobalInvasion.csv")

#####################################
######### MAT MNTD extremes #########
#####################################

sp.r.nat <- read.csv("data/GFBI_biome_rarefaction_invasives.csv") %>%                                         # Read condensed GFBI 
  filter(keep_rarefied == 1) %>%                                                                              # Subset to rarefied data
  filter(!Glonaf_collapsed_status == "invasive") %>%                                                          # Subset to natives
  select(c("plot_id", "accepted_bin", "avglat", "avglon", "Glonaf_collapsed_status")) %>%                     # Subset to cols of interest
  distinct(accepted_bin, avglat, avglon, .keep_all = TRUE) %>%                                                # Drop duplicates of species and lat/lon                                                                              
  as.data.frame()                                                                                             # Make a df

sp.r.inv <- read.csv("data/GFBI_biome_rarefaction_invasives.csv") %>%                                         # Read condensed GFBI 
  filter(keep_rarefied == 1) %>%                                                                              # Subset to rarefied data
  filter(Glonaf_collapsed_status == "invasive") %>%                                                           # Subset to invasives
  select(c("plot_id", "accepted_bin", "avglat", "avglon", "Glonaf_collapsed_status")) %>%                     # Subset to cols of interest
  distinct(accepted_bin, avglat, avglon, .keep_all = TRUE) %>%                                                # Drop duplicates of species and lat/lon                                                                              
  as.data.frame()  

#####################################
############ PD extremes ############
#####################################

#filter by lat-lon determined in pd analyses
#lowest and highest MAT 0.005, d.mntd < 0
extremes <- readRDS("data/PDstrategy_tempextremes.rds") %>% mutate(latlon= paste(lat, avglon, sep="_")) %>% select(MAT, latlon, extreme)

sp.r.nat.ext <- sp.r.nat %>% mutate(latlon = paste(avglat, avglon, sep="_")) %>% 
  left_join(extremes, by = "latlon") %>%
  drop_na()
sp.r.inv.ext <- sp.r.inv %>% mutate(latlon = paste(avglat, avglon, sep="_")) %>% 
  left_join(extremes, by = "latlon") %>%
  drop_na()

sp.r.nat.ext.low <- sp.r.nat.ext %>% filter(extreme=="low") %>% 
  select(accepted_bin, latlon) %>%
  rename(natlow = accepted_bin)
sp.r.nat.ext.high <- sp.r.nat.ext %>% filter(extreme=="high") %>% 
  select(accepted_bin, latlon) %>%
  rename(nathigh = accepted_bin)

sp.r.inv.ext.low <- sp.r.inv.ext %>% filter(extreme=="low") %>% 
  select(accepted_bin, latlon) %>%
  rename(invlow = accepted_bin)
sp.r.inv.ext.high <- sp.r.inv.ext %>% filter(extreme=="high") %>% 
  select(accepted_bin, latlon) %>%
  rename(invhigh = accepted_bin)

sp.low <- sp.r.nat.ext.low %>%
  full_join(sp.r.inv.ext.low , by = "latlon") %>%
  mutate(latlon = as.numeric(as.factor(latlon)))

sp.high <- sp.r.nat.ext.high %>%
  full_join(sp.r.inv.ext.high , by = "latlon") %>%
  mutate(latlon = as.numeric(as.factor(latlon)))

unique(sp.r.nat.ext.low$accepted_bin)
unique(sp.r.inv.ext.low$accepted_bin)

unique(sp.r.nat.ext.high$accepted_bin)
unique(sp.r.inv.ext.high$accepted_bin)

#####################################
############ FD extremes ############
#####################################

#filter by lat-lon determined in fd analyses
#lowest and highest MAT 0.005, d.mntd < 0
extremes <- readRDS("data/FDstrategy_tempextremes.rds") %>% mutate(latlon= paste(lat, avglon, sep="_")) %>% select(MAT, latlon, extreme)

sp.r.nat.ext <- sp.r.nat %>% mutate(latlon = paste(avglat, avglon, sep="_")) %>% 
  left_join(extremes, by = "latlon") %>%
  drop_na()
sp.r.inv.ext <- sp.r.inv %>% mutate(latlon = paste(avglat, avglon, sep="_")) %>% 
  left_join(extremes, by = "latlon") %>%
  drop_na()

sp.r.nat.ext.low <- sp.r.nat.ext %>% filter(extreme=="low") %>% 
  select(accepted_bin, latlon) %>%
  rename(natlow = accepted_bin)
sp.r.nat.ext.high <- sp.r.nat.ext %>% filter(extreme=="high") %>% 
  select(accepted_bin, latlon) %>%
  rename(nathigh = accepted_bin)

sp.r.inv.ext.low <- sp.r.inv.ext %>% filter(extreme=="low") %>% 
  select(accepted_bin, latlon) %>%
  rename(invlow = accepted_bin)
sp.r.inv.ext.high <- sp.r.inv.ext %>% filter(extreme=="high") %>% 
  select(accepted_bin, latlon) %>%
  rename(invhigh = accepted_bin)

sp.low <- sp.r.nat.ext.low %>%
  full_join(sp.r.inv.ext.low , by = "latlon") %>%
  mutate(latlon = as.numeric(as.factor(latlon)))

sp.high <- sp.r.nat.ext.high %>%
  full_join(sp.r.inv.ext.high , by = "latlon") %>%
  mutate(latlon = as.numeric(as.factor(latlon)))

unique(sp.r.nat.ext.low$accepted_bin)
unique(sp.r.inv.ext.low$accepted_bin)

unique(sp.r.nat.ext.high$accepted_bin)
unique(sp.r.inv.ext.high$accepted_bin)
