#####################################
#####################################
###### JOIN DATA FOR ANALYSES #######
#####################################
#####################################

#####################################
###### Load packages; read dat ######
#####################################

library(feather)
packageVersion("feather")
library(tidyverse)
packageVersion("tidyverse")

#####################################
###### Generate repeat subset #######
#####################################

# read in data for analyses
repeat.pc <- readRDS("data/GFBI_na_inv_prop_counts_year.rds") %>%
  select(c(plot_id, year, nativect, invasct))

# Get earliest year and rename counts
repeat.pc.first <- repeat.pc %>% 
  group_by(plot_id) %>%                                              
  arrange(year) %>%                                               
  slice(1) %>%                                                         
  rename(first.year = year, first.nativect = nativect, first.invasct = invasct) %>%
  ungroup()

# Get latest year and rename counts
repeat.pc.last <- repeat.pc %>% 
  group_by(plot_id) %>%                                                                     
  arrange(desc(year)) %>%                                                    
  slice(1) %>%                                                               
  rename(last.year = year, last.nativect = nativect, last.invasct = invasct) %>%
  ungroup()

# Join first and last and only keep different years; filter for no invasion first year; assign final year status
repeat.pc.true <- repeat.pc.first %>% 
  left_join(repeat.pc.last, by = "plot_id") %>%
  filter(!first.year == last.year)  %>%
  filter(first.invasct == 0) %>%
  mutate(Finalinvstat = ifelse(last.invasct > 0, 1, 0)) %>%
  mutate(PosInvChange = ifelse(last.invasct > first.invasct, "Yes","No")) %>%
  mutate(Y = last.year - first.year) %>%
  mutate(deltaspecies = last.nativect - first.nativect) %>%
  mutate(deltaspecies.yr = deltaspecies/Y) %>%
  mutate(percentchange = (last.nativect - first.nativect)/first.nativect) %>%
  mutate(percentchange.yr = percentchange/Y) 

#####################################
############ Load data ##############
#####################################

pc <- repeat.pc.true %>%
  select(c("plot_id", "Y","Finalinvstat", "deltaspecies", "deltaspecies.yr", "percentchange", "percentchange.yr"))

comp.human <- read.table("data/20211019_InvasiveSpeciesComposite.csv", sep = ",", header=TRUE) %>%                     # Read human influence composite data
  select(c("plot_id", "WFP_DistanceToPorts","WorldBank_DistanceToAirports",
           "PopulationDensity2015_GHS", "HumanFootprint2013","Resolve_Biome")) 

lat <- read.csv("data/GFBI_biome_rarefaction_invasives.csv") %>%                                                       # Read condensed GFBI 
  #filter(keep_rarefied == 1) %>%                                                                                      # Subset to rarefied data
  select(c("plot_id", "avglat", "avglon")) %>%                                                                         # Subset to cols of interest
  distinct(plot_id, .keep_all = TRUE) %>%                                                                              # Drop duplicates of species and plot                                                                              
  rename(lat = avglat) %>%                                                                                             # Rename lat
  as.data.frame()                                                                                                      # Make a df

comp.env <- read_feather("data/20211019_InvasiveSpeciesCrowtherLabComposite.feather") %>%                              # Read environmental composite
  select(starts_with("SG_") | starts_with("CHELSA") | starts_with("plot_id") | starts_with("GFAD_regrowthForestAge") | starts_with("CrowtherLab_IntactLandscapes"))  # Subset to cols of interest

#####################################
######### Join all datasets #########
#####################################

#all information is per plot, so no need to do per year
#join met with comp.static and lat
met.env.comp <- pc %>%     # Start with pc                             
  left_join(comp.human, by = "plot_id") %>%                                                                             # Join with comp.human
  left_join(lat, by = "plot_id") %>%                                                                                    # Join with lat
  left_join(comp.env, by = "plot_id") %>%                                                                               # Join with comp.env                                                                                                      # Drop NAs
  mutate(biome = case_when(Resolve_Biome == "1" ~ "Tropical_Moist_Broadleaf",                                           # Create biome2 and rename 3 major biomes
                           Resolve_Biome == "2" ~ "Tropical_Deciduous_Broadleaf", 
                           Resolve_Biome == "3" ~ "Tropical_Coniferous", 
                           Resolve_Biome == "4" ~ "Temperate_BroadLeaf", 
                           Resolve_Biome == "5" ~ "Temperate_Coniferous", 
                           Resolve_Biome == "6" ~ "Boreal", 
                           Resolve_Biome == "7" ~ "Tropical_Grasslands",
                           Resolve_Biome == "8" ~ "Temperate_Grasslands",
                           Resolve_Biome == "9" ~ "Flooded_Grasslands", 
                           Resolve_Biome == "10" ~ "Montane_Grasslands",
                           Resolve_Biome == "11" ~ "Tundra", 
                           Resolve_Biome == "12" ~ "Mediterranean_woodlands",
                           Resolve_Biome == "13" ~ "Xeric_shrublands", 
                           Resolve_Biome == "14" ~ "Mangroves")) %>%
  mutate(biome2 = case_when(biome == "Tropical_Moist_Broadleaf" ~ "Tropical",                                                  # Create biome2 and rename 3 major biomes
                            biome == "Tropical_Deciduous_Broadleaf" ~ "Tropical", 
                            biome == "Tropical_Coniferous" ~ "Tropical",      
                            biome == "Tropical_Grasslands" ~ "Tropical", 
                            biome == "Temperate_BroadLeaf" ~ "Temperate",      
                            biome == "Temperate_Coniferous" ~ "Temperate",
                            biome == "Temperate_Grasslands" ~ "Temperate")) %>%
  mutate(biome2 = ifelse(is.na(biome2),"Other",biome2)) %>%                                                                 # Call non trop/temp "Other"
  mutate(abslat = abs(lat)) %>%                                                                                           # Add absolute lat
  rename(dist.ports = WFP_DistanceToPorts,                                                                               # Rename vars
         dist.airports = WorldBank_DistanceToAirports,
         popdensity = PopulationDensity2015_GHS,
         humanfootprint = HumanFootprint2013,
         forestage = GFAD_regrowthForestAge_Mean_downsampled50km,
         intact = CrowtherLab_IntactLandscapes) %>% 
  mutate_at(c("Y", "Finalinvstat", "deltaspecies", "deltaspecies.yr", "percentchange", "percentchange.yr", 
              "dist.ports", "dist.airports", "popdensity", "humanfootprint", "forestage","lat","abslat"), as.numeric) %>%
  drop_na(-c(biome2))                                                                                                              # Drop na

#####################################
######### Further simplify ##########
########### for analyses ############
#####################################

dat <- met.env.comp %>%                                                                                      # Start met.env.comp
  mutate(MAT = CHELSA_BIO_Annual_Mean_Temperature, MAP = CHELSA_BIO_Annual_Precipitation) %>%                # Simplify MAT MAP
  group_by(lat,avglon) %>%                                                                                   # Group by latlon
  summarise(across(-c(plot_id, biome, biome2, Resolve_Biome, Y, Finalinvstat, deltaspecies, deltaspecies.yr, percentchange, percentchange.yr), mean, na.rm = TRUE)) %>%                # Summarize across latlon (mean)                                                # Get means across latlon
  ungroup() %>%                                                                                              # Ungroup
  as.data.frame() 

met.env.comp.simp <-  met.env.comp %>%                                                                       # Start met.env.comp
  select(lat, avglon, biome, biome2, Y, Finalinvstat,  deltaspecies, deltaspecies.yr, percentchange, percentchange.yr)                                                                            # Select variables of interest

dat.final.analyses <- dat %>%
  left_join(met.env.comp.simp, by = c('lat', 'avglon')) %>%
  distinct()

write.csv(dat.final.analyses, "data/GFBI_invasion_foranalyses_repeat.csv")                                     # Write out file
saveRDS(dat.final.analyses, "data/GFBI_invasion_foranalyses_repeat.rds")
