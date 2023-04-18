#####################################
#####################################
###### CONDENSE GFBI DATAFRAME ######
########## PER PLOT/YEAR ############
#####################################
#####################################

#####################################
###### Load packages; read dat ######
#####################################

library(feather)
packageVersion("feather")
library(tidyverse)
packageVersion("tidyverse")

best.match <- read_feather("data/BEST_MATCH_all_species_TPL.feather")                  # Read in tpl match   

######################################
####### Match species names ##########
######################################

# 1184925 plots
full.plots <- read_feather("data/GFBI_fixed_plots.feather") %>%                        # Read in full GFBI data       
  left_join(best.match, by = "raw_name") %>%                                           # Join with best.match
  select(-raw_name)                                                                    # Remove raw name

saveRDS(full.plots,"data/GFBI_fixed_plots_acceptname.rds")                             # Write out file
write.csv(full.plots,"data/GFBI_fixed_plots_acceptname.csv")                           # Write out file

######################################
####### Plot average lat lon #########
######################################

# 1184925 plots
full.plots.avgll <- full.plots %>%                                                     # Start with full.plots
  group_by(plot_id) %>%                                                                # Group by plot_id
  summarize(avglat = mean(lat), avglon = mean(lon)) %>%                                # Determine average lat/lon
  ungroup()                                                                            # Ungroup

######################################
########### Unique species ###########
####### per plot_id per year #########
######################################

# 1184925 plots
full.plots.plotsp <- full.plots %>%                                                     # Start with full.plots
  select(plot_id, accepted_bin, year) %>%                                               # Select variables
  distinct(.keep_all = TRUE)                                                            # Keep only distinct combinations

######################################
########### Join these two ###########
##### Add lat/lon to species df ######
######################################

# 1184817 plots
full.plots.join <- full.plots.plotsp %>%                                                # Start with full.plots.plotsp
  left_join(full.plots.avgll, by = c("plot_id")) %>%                                    # Add lat/lon information
  drop_na()                                                                             # Drop NAs

######################################
######## Remove communities ##########
########## with <3 species ###########
######################################

# 797692 plots
full.plots.min <- full.plots.join %>%                                                   # Start with full.plots.join
  group_by(plot_id, year) %>%                                                           # Group by plot_id and year
  summarize(speciesnum = length(accepted_bin)) %>%                                      # Count species number 
  filter(speciesnum > 2) %>%                                                            # Keep only where min 3 species
  ungroup() %>%                                                                         # Ungroup
  mutate(plot_year = paste(plot_id, year, sep = "_"))                                   # Make plot_year var

# 797692 plots
full.plots.final <- full.plots.join %>%                                                 # Start with full.plots.join
  mutate(plot_year = paste(plot_id, year, sep = "_")) %>%                               # Make plot_year var
  filter(plot_year %in% full.plots.min$plot_year) %>%                                   # Keep only plot_year present in full.plots.min
  select(-"plot_year")                                                                  # Remove plot_year var

saveRDS(full.plots.final,"data/GFBI_fixed_plots_final.rds")                             # Write out file
write.csv(full.plots.final,"data/GFBI_fixed_plots_final.csv")                           # Write out file

# GEE code uses this output to match Glonaf and Kew to GFBI
# Then subset to only plots with full data assigned and no conflicts
# One plot per year (most recent)
# Then rarified to prop represented by biome (C_GFBI_Downsample.R)