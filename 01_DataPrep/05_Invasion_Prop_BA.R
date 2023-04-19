#####################################
#####################################
###### CALCULATE PROP SPECIES #######
####### AND PROP BA INVADERS ########
#####################################
#####################################

#1. downsample/upsample version
#2. full dataset

# Glonaf_collapsed_status chosen instead of status, as status can be unknown glonaf and known status.

#####################################
###### Load packages; read dat ######
#####################################

library(tidyverse)
packageVersion("tidyverse")

sp <- read.csv("data/GFBI_biome_rarefaction_invasives.csv") %>%                                                # Read condensed GFBI 
  filter(keep_rarefied == 1) %>%                                                                               # Subset to rarefied data
  select(c("plot_id", "accepted_bin", "plot_id", "Glonaf_collapsed_status","status")) %>%                      # Subset to cols of interest
  distinct(accepted_bin, plot_id, .keep_all = TRUE) %>%                                                        # Drop duplicates of species and plot                                                                              
  mutate(accepted_bin = gsub(" ", "_", accepted_bin)) %>%                                                      # Add underscore
  as.data.frame()  

#!!!OPTION KEEPING ALL PLOTS
#!!!REMAINDER IS EQUIVALENT TO NORMAL, JUST SAVE AS ALL AT END OF SCRIPT
sp <- read.csv("data/GFBI_biome_rarefaction_invasives.csv")  %>%                                               # Read condensed GFBI 
  #filter(keep_rarefied == 1) %>%                                                                              # Subset to rarefied data
  select(c("plot_id", "accepted_bin", "plot_id", "Glonaf_collapsed_status","status")) %>%                      # Subset to cols of interest
  distinct(accepted_bin, plot_id, .keep_all = TRUE) %>%                                                        # Drop duplicates of species and plot                                                                              
  mutate(accepted_bin = gsub(" ", "_", accepted_bin)) %>%                                                      # Add underscore
  as.data.frame()  

#!!!OPTION KEEPING ALL PLOTS AND YEARS
sp.yr <- read.csv("data/GFBI_Glonaf_Kew_Master_multiyear.csv") %>%                                                                                 # Read condensed GFBI *per year
  select(-1) %>%                                                                                                                                   # Remove first column
  rename(plot_id = GFBI_plot_id) %>%                                                                                                               # Rename plot_id
  mutate(status = ifelse(Glonaf_collapsed_status == "invasive" | Kew_status =="invasive", "invasive", "native")) %>%                               # Create consensus status, inv in both, or left as native
  filter(!(Glonaf_collapsed_status == "invasive" & Kew_status=="native") | !(Glonaf_collapsed_status == "native" & Kew_status =="invasive"))  %>%  # Filter mistmatches
  select(c("plot_id", "accepted_bin", "plot_id", "Glonaf_collapsed_status","status", "year")) %>%                                                  # Subset to cols of interest
  distinct(accepted_bin, plot_id, year, .keep_all = TRUE) %>%                                                                                      # Drop duplicates of species and plot and year                                                                              
  mutate(accepted_bin = gsub(" ", "_", accepted_bin)) %>%                                                                                          # Add underscore
  as.data.frame()  

#####################################
###### Calculate native counts ######
#####################################

native <- sp %>%                                                                                               # Start with sp
  filter(Glonaf_collapsed_status == "native") %>%                                                              # Subset to natives
  group_by(plot_id) %>%                                                                                        # Group by plot 
  summarize(nativect = length(accepted_bin))                                                                   # Count species number per plot

#!!!OPTION KEEPING ALL PLOTS AND YEARS
native.yr <- sp.yr %>%                                                                                         # Start with sp
  filter(Glonaf_collapsed_status == "native") %>%                                                              # Subset to natives
  group_by(plot_id, year) %>%                                                                                  # Group by plot and year
  summarize(nativect = length(accepted_bin))                                                                   # Count species number per plot and year

#####################################
##### Calculate invasive counts #####
#####################################

invasive <- sp %>%                                                                                             # Start with sp
  filter(Glonaf_collapsed_status == "invasive")  %>%                                                           # Subset to invasives
  group_by(plot_id) %>%                                                                                        # Group by plot 
  summarize(invasct = length(accepted_bin))                                                                    # Count species number per plot

#!!!OPTION KEEPING ALL PLOTS AND YEARS
invasive.yr <- sp.yr %>%                                                                                       # Start with sp
  filter(Glonaf_collapsed_status == "invasive")  %>%                                                           # Subset to invasives
  group_by(plot_id, year) %>%                                                                                  # Group by plot 
  summarize(invasct = length(accepted_bin))                                                                    # Count species number per plot

########################################
########### Join these two & ###########
############ Calculate props ###########
########################################

invas.pc <- native %>% 
  full_join(invasive, by = c("plot_id")) %>%                                                                   # Join native and invasive counts by plot_id, full join in case of all in one or other
  mutate(nativect = ifelse(is.na(nativect), 0, nativect)) %>%                                                  # Replace NA with zero
  mutate(invasct = ifelse(is.na(invasct), 0, invasct))  %>%                                                    # Replace NA with zero
  mutate(propinvasive = (invasct)/(nativect + invasct)) %>%                                                    # Add proportion invasive (invasive + alien)
  mutate(propinvasive = ifelse(invasct == 0, 0, propinvasive))                                                 # Where no invasives, prop invasive is zero 

#!!!OPTION KEEPING ALL PLOTS AND YEARS
#!!!ENDS HERE.
invas.pc.yr <- native.yr %>% 
  full_join(invasive.yr, by = c("plot_id","year"))  %>%                                                        # Join native and invasive counts by plot_id, full join in case of all in one or other
  mutate(nativect = ifelse(is.na(nativect), 0, nativect)) %>%                                                  # Replace NA with zero
  mutate(invasct = ifelse(is.na(invasct), 0, invasct))  %>%                                                    # Replace NA with zero
  mutate(propinvasive = (invasct)/(nativect + invasct)) %>%                                                    # Add proportion invasive (invasive + alien)
  mutate(propinvasive = ifelse(invasct == 0, 0, propinvasive))                                                 # Where no invasives, prop invasive is zero 

saveRDS(invas.pc.yr, "data/GFBI_na_inv_prop_counts_year.rds")                                                  # Write out file

#####################################
#### Calculate species richness #####
######## & join to sprich df ########
#####################################

sprich <- sp %>%                                                                                               # Start with sp
  group_by(plot_id) %>%                                                                                        # Group by plot 
  summarize(sprich = length(accepted_bin)) %>%                                                                 # Summarize counts of accepted_bin
  ungroup()                                                                                                    # Ungroup

invas.pc.sp <- invas.pc %>%                                                                                    # Start with invas.pc
  left_join(sprich, by = c("plot_id"))                                                                         # Join with sprich by plot_id

#####################################
####### Calculate BA invasive #######
#####################################

full.plots <- readRDS("data/GFBI_fixed_plots_acceptname.rds") %>%                                              # Full original accepted name
  mutate(accepted_bin = gsub(" ", "_", accepted_bin)) %>%                                                      # Add underscore
  select(plot_id, dbh, tph, accepted_bin)                                                                      # Subset to cols of interest
                         
full.plots <- sp %>%                                                                                           # Start with sp 
  left_join(full.plots, by = c("plot_id", "accepted_bin"))                                                     # Left join with full.plots to get additional info

ba <- full.plots %>%                                                                                           # Start with full.plots
   mutate(ba = ((pi*(dbh/2)^2)*tph))                                                                           # Calculate basal area per area (BA *trees per hectare est)

#####################################
######## Calculate native BA ########
#####################################

nat.ba <- ba %>%                                                                                               # Start with ba
  filter(Glonaf_collapsed_status == "native") %>%                                                              # Subset to natives
  group_by(plot_id)%>%                                                                                         # Group by plot
  summarize(nativeba = sum(ba))                                                                                # Sum BA per plot

#####################################
####### Calculate invasive BA #######
#####################################

inv.ba <- ba %>%                                                                                               # Start with ba
  filter(Glonaf_collapsed_status == "invasive") %>%                                                            # Subset to invasives
  group_by(plot_id)%>%                                                                                         # Group by plot
  summarize(invba = sum(ba))                                                                                   # Sum BA per plot

########################################
########### Join these two & ###########
############ Calculate props ###########
########################################

invas.pba <- nat.ba %>%                                                                                        # Start with native BA
  full_join(inv.ba, by = c("plot_id")) %>%                                                                     # Join with invasive
  mutate(nativeba = ifelse(is.na(nativeba), 0, nativeba)) %>%                                                  # Replace NA with zero
  mutate(invba = ifelse(is.na(invba), 0, invba))  %>%                                                          # Replace NA with zero
  mutate(pinvba = (invba)/(nativeba + invba)) %>%                                                              # Add proportion invasive (invasive + alien)
  mutate(pinvba = ifelse(invba == 0, 0, pinvba))                                                               # Where no invasive ba, prop invasive is zero 

########################################
########### Join all data ##############
########################################

invas.pc.sp.ba <- invas.pc.sp %>%                                                                              # Start with invas.pc.sp
  left_join(invas.pba, by = c("plot_id"))                                                                      # Join with prop BA by plot

saveRDS(invas.pc.sp.ba, "data/GFBI_na_inv_prop_counts.rds")                                                    # Write out file

#!!!OPTION KEEPING ALL PLOTS
saveRDS(invas.pc.sp.ba, "data/GFBI_na_inv_prop_counts_rareandnon.rds")                                         # Write out file
