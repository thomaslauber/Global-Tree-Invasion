#####################################
#####################################
###### JOIN DATA FOR ANALYSES #######
##### DOWNSAMPLE & NO UPSAMPLE ######
#####################################
#####################################

#####################################
###### Load packages; read dat ######
#####################################

library(feather)
packageVersion("feather")
library(tidyverse)
packageVersion("tidyverse")

pc <- readRDS("data/GFBI_na_inv_prop_counts_rareandnon.rds")                                                                      # Read prop/counts/ba invasive data 

comp.human <- read.table("data/20211019_InvasiveSpeciesComposite.csv", sep = ",", header=TRUE) %>%                     # Read human influence composite data
  select(c("plot_id", "WFP_DistanceToPorts","WorldBank_DistanceToAirports",
           "PopulationDensity2015_GHS", "HumanFootprint2013","Resolve_Biome")) 

lat <- read.csv("data/GFBI_biome_rarefaction_invasives_noupsample.csv") %>%                                                        # Read condensed GFBI 
  filter(keep_rarefied == 1) %>%                                                                                       # Subset to rarefied data
  select(c("plot_id", "avglat", "avglon")) %>%                                                                         # Subset to cols of interest
  distinct(plot_id, .keep_all = TRUE) %>%                                                                              # Drop duplicates of species and plot                                                                              
  rename(lat = avglat) %>%                                                                                             # Rename lat
  as.data.frame()                                                                                                      # Make a df

comp.env <- read_feather("data/20211019_InvasiveSpeciesCrowtherLabComposite.feather")  %>%                              # Read environmental composite
  select(starts_with("SG_") | starts_with("CHELSA") | starts_with("plot_id") | starts_with("GFAD_regrowthForestAge") | starts_with("CrowtherLab_IntactLandscapes"))   # Subset to cols of interest

fdmet.inv <- readRDS("data/FDmetrics_Allspecies_noupsample.rds") %>%                                                    # Read all functional data
  spread(type, value) %>%                                                                                              # Spread data
  rename(plot_id = plot_id, all.faith = faith, all.mntd = mntd, all.mpd = mpd, all.vntd = vntd, all.vpd = vpd)         # Rename cols

fdmet.nat <- readRDS("data/FDmetrics_Nativespecies_noupsample.rds") %>%                                                 # Read native functional data
  spread(type, value) %>%                                                                                              # Spread dat
  rename(plot_id = plot_id, nat.faith = faith, nat.mntd = mntd, nat.mpd = mpd, nat.vntd = vntd, nat.vpd = vpd)         # Rename cols

#####################################
###### Join FD nat & invasive #######
#####################################

fdmet <- fdmet.nat %>%                                                                                                 # Start with fdmet.nat
  left_join(fdmet.inv, by = c("plot_id")) %>%                                                                          # Join dataframes
  mutate(d.faith = ((all.faith-nat.faith)/nat.faith),                                                                  # Calculate difference between all and natd.mntd = (all.mntd-nat.mntd)/nat.mntd),
         d.mntd = ((all.mntd-nat.mntd)/nat.mntd),
         d.mpd = ((all.mpd-nat.mpd)/nat.mpd),
         d.vntd = ((all.vntd-nat.vntd)/nat.vntd),
         d.vpd = ((all.vpd-nat.vpd)/nat.vpd)) 
  
#####################################
######### Join all datasets #########
#####################################

#join fdmet with comp.static and lat
fdmet.env.comp <- lat %>%                                                                                               # Start with lat                                 
  left_join(pc, by = c("plot_id")) %>%                                                                                  # Join with invasion dat
  left_join(comp.human, by = "plot_id") %>%                                                                             # Join with comp.human
  left_join(fdmet, by = "plot_id") %>%                                                                                  # Join with fdmet
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
  mutate(biome2 = ifelse(is.na(biome2), "Other" ,biome2)) %>%                                                                 # Call non trop/temp "Other"
  mutate(abslat = abs(lat)) %>%                                                                                           # Add absolute lat
  mutate(in.stat = ifelse(propinvasive == 0, "non-invaded", "invaded")) %>%                                               # Add plot invaded or not
  rename(dist.ports = WFP_DistanceToPorts,                                                                               # Rename vars
         dist.airports = WorldBank_DistanceToAirports,
         popdensity = PopulationDensity2015_GHS,
         humanfootprint = HumanFootprint2013,
         forestage = GFAD_regrowthForestAge_Mean_downsampled50km,
         intact = CrowtherLab_IntactLandscapes) %>% 
  mutate_at(c("nat.vntd", "nat.vpd", "nat.mntd", "nat.mpd" , "nat.faith",                                             # Convert all numeric to numeric
                "all.faith", "all.mntd","all.mpd", "all.vntd", "all.vpd", "d.faith", "d.mntd", "d.mpd", "d.vntd", "d.vpd",  
                "nativect", "invasct", "propinvasive", "sprich", "nativeba", "invba", "pinvba",       
                "dist.ports", "dist.airports", "popdensity", "humanfootprint", "forestage","lat","abslat"), as.numeric) %>%
  drop_na(-c(biome2))                                                                                                              # Drop na

#1. Tropical moist broadleaf forests
#2. Tropical dry broadleaf forests
#3. Tropical coniferous forests
#4. Temperate broadleaf forests
#5. Temperate conifer forests
#6. Boreal forests or taiga
#7. Tropical grasslands
#8. Temperate grasslands
#9. Flooded grasslands
#10. Montane grasslands
#11. Tundra
#12. Mediterranean woodlands
#13. Xeric shrublands
#14. Mangroves

#####################################
######### Further simplify ##########
########### for analyses ############
#####################################

dat <- fdmet.env.comp %>%                                                                                    # Start fdmet.env.comp
  mutate(lgsprich = log(sprich + 1)) %>%                                                                     # Add log species richness
  mutate(nat.mntd = max(nat.mntd) - nat.mntd) %>%                                                            # Change mntd to 'redundancy'
  mutate(MAT = CHELSA_BIO_Annual_Mean_Temperature, MAP = CHELSA_BIO_Annual_Precipitation) %>%                # Simplify MAT MAP
  group_by(lat,avglon) %>%                                                                                   # Group by latlon
  summarise(across(-c(plot_id, in.stat, biome, biome2, Resolve_Biome), mean, na.rm = TRUE)) %>%              # Summarize across latlon (mean)                                                # Get means across latlon
  mutate(invasct = as.integer(invasct), nativect = as.integer(nativect)) %>%                                 # Make these integers
  ungroup() %>%                                                                                              # Ungroup
  mutate(in.stat= ifelse(propinvasive == 0, "non-invaded", "invaded")) %>%                                   # Add plot invaded or not
  as.data.frame() 

fdmet.env.comp.simp <-  fdmet.env.comp %>%                                                                   # Start fdmet.env.comp
  select(lat, avglon, biome, biome2)                                                                         # Select variables of interest

dat.final.analyses <- dat %>%
  left_join(fdmet.env.comp.simp, by = c('lat','avglon')) %>%
  distinct() 

write.csv(dat.final.analyses, "data/GFBI_invasion_FDforanalyses_noupsample.csv")
saveRDS(dat.final.analyses, "data/GFBI_invasion_FDforanalyses_noupsample.rds")
