#####################################
#####################################
##### GENERATE DOWNSAMPLED DATA #####
########## UPSAMPLED OR NOT #########
#####################################
#####################################

rm(list = ls())
gc()

library(feather)
library(geosphere)
library(tidyverse)

# read in other data for LAT LON and biomes
env <- read_feather("data/GFBI_full_new_composite_7_21.feather")
ll <- env %>% select(LAT, LON, ll_id, Resolve_Biome, Resolve_Ecoregion) %>% distinct()
gf <- read_feather("data/GFBI_fixed_plots.feather") %>% 
	select(lat, lon, plot_id) %>% distinct() %>% rename(LAT = lat, LON = lon) %>% 
	left_join(ll) 

# read in data
dto <- dt <- read_csv("data/GFBI_Glonaf_Kew_Master_Ver2.csv") %>% select(-1) %>% rename(plot_id = GFBI_plot_id)

# number of plots
length(unique(dto$plot_id))

# assign a cleaned invaded status version
dt <- dto %>% left_join(gf) %>% 
	mutate(status = ifelse(Glonaf_collapsed_status == "invasive" | Kew_status =="invasive", "invasive", "native"))

# get the plots with a mismatch
bad_plots <- dt %>% filter((Glonaf_collapsed_status == "invasive" & Kew_status=="native") | (Glonaf_collapsed_status == "native" & Kew_status =="invasive")) %>% 
	select(plot_id) %>% distinct() %>% unlist() %>% as.character()

# assign a plot status
dt <- dt %>% mutate(plot_status = ifelse(plot_id%in%bad_plots, "mismatch", "agreement"))

# read in the forest cover dataset and assign the names
biome_mat <- rbind(c(1, 'Tropical moist broadleaf forests', 14.9),
				   c(2, 'Tropical dry broadleaf forests', 2.9),
				   c(3, 'Tropical coniferous forests', 0.5),
				   c(4, 'Temperate broadleaf forests', 9.6),
				   c(5, 'Temperate conifer forests', 2.9),
				   c(6, 'Boreal forests or taiga', 11.5),
				   c(7, 'Tropical grasslands', 16.3),
				   c(8, 'Temperate grasslands', 8.0),
				   c(9, 'Flooded grasslands', 0.9),
				   c(10, 'Montane grasslands', 3.6),
				   c(11, 'Tundra', 6.1),
				   c(12, 'Mediterranean woodlands', 2.5),
				   c(13, 'Xeric shrublands', 20.0),
				   c(14, 'Mangroves', 0.3)) %>% data.frame() %>% setNames(c("Resolve_Biome", "biome", "Area")) %>% as_tibble() %>% 
	mutate(Resolve_Biome = as.numeric(as.character(Resolve_Biome)),
		   Area = as.numeric(as.character(Area))) %>% mutate(prop0 = Area/sum(Area)) %>% 
	left_join(bm <- read_csv("data/biome_forest_cover.csv") %>%
			  	mutate(prop10 = area_10perc/sum(area_10perc), prop30 = area_30perc/sum(area_30perc)) %>% select(Resolve_Biome, prop10, prop30)) %>%
	rename(biome_num = Resolve_Biome)

# filter out all of the plots not in agreement, and assign a new variable plot_status to indicate if the plot is invaded or not
dfilt <- dt %>% filter(plot_status == "agreement") %>% rename(biome_num = Resolve_Biome, ecoregion = Resolve_Ecoregion) %>% left_join(biome_mat) %>% filter(!is.na(biome)) %>% 
	mutate(plot_status = ifelse(plot_id%in%(dt %>% filter(plot_status == "agreement") %>% filter(status == "invasive") %>% select(plot_id) %>% distinct() %>% unlist()), "invaded", "not_invaded"))

# make sure we're only getting plots with at least 3 species
nspp3 <- dfilt %>% 
	# filter(status == "native") %>%
	select(plot_id, accepted_bin) %>% distinct() %>% 
	group_by(plot_id) %>% tally() %>% ungroup %>% filter(n > 2)
dfilt <- dfilt %>% filter(plot_id%in%nspp3$plot_id)

# summary of number invaded plots per biome
status_summary <- dfilt %>% select(plot_id, biome, plot_status) %>% distinct() %>% group_by(biome, plot_status) %>% tally() %>% 
	spread(plot_status, n) %>% mutate(invaded = ifelse(is.na(invaded), 0, invaded))

# total number of plots per biome
biome_numbers <- dfilt %>% select(plot_id, biome, biome_num) %>% distinct() %>% group_by(biome, biome_num) %>% tally() %>% ungroup

# total number of plots and prop forest area in the tropical biomes, then scaled to get global total of plots needed
trop_total <- biome_numbers %>% left_join(biome_mat) %>% filter(grepl("Tropical moist", biome)) %>% summarize(n_tropical = sum(n), prop = sum(prop0)) %>% 
	mutate(n_total = ceiling(n_tropical / prop))

max_trop <- biome_numbers %>% filter(grepl("Tropical moist", biome)) %>% summarize(n = max(n)) %>% unlist()

# calculate the number of invaded and non-invaded plots per biome. 
# currently set to ensure no more than half of plots are invaded 
# if there are more invaded plots that total needed
target_plots <- biome_mat %>% 
	left_join(status_summary) %>% 
	rowwise() %>% 
	mutate(total = invaded+not_invaded) %>% 
	mutate(target = min(total, prop0*trop_total$n_total)) %>% 
	mutate(num_invasive = ifelse(invaded > round(target*0.5), round(target*0.5), invaded)) %>% 
	mutate(num_native = target - num_invasive) %>% ungroup

set.seed(10)
filt_data <- tibble()

### are we upsampling to 50%?
UPSAMPLE <- FALSE

# cycle through the biomes and sample the specified number of invaded and non invaded plots, given by target_plots
for(i in 1:nrow(target_plots)){
	
	# possible plots to choose
	my_filt <- dfilt %>% filter(biome_num == target_plots$biome_num[i]) %>% select(plot_id, biome, plot_status, ecoregion) %>% distinct() 
	
	if(UPSAMPLE){
		### Upsample invaded plots to max 50%
		inv_plots <- my_filt %>% filter(plot_status == "invaded") %>% sample_n(target_plots$num_invasive[i])
		native_plots <- my_filt %>% filter(plot_status == "not_invaded") %>% sample_n(target_plots$num_native[i])
		keep_filt <- inv_plots %>% bind_rows(native_plots)
	}else{
		### Or, just sample by biome
		keep_filt <- my_filt %>% sample_n(target_plots$target[i])
	}
	# combine							  
	filt_data <- filt_data %>% bind_rows(keep_filt %>% select(plot_id))
}

# annotate the dataset noting which ones to keep
d_final <- dfilt %>% mutate(keep_rarefied = ifelse(plot_id%in%filt_data$plot_id, 1, 0))

# visualize/compare the results / get summary stats
d_final %>% filter(LON >= 164) %>% filter(LAT <= 179) %>% filter(LON >= -54) %>% filter(LAT <= -34) %>% 
	select(plot_id, keep_rarefied, plot_status) %>% distinct() %>% 
	group_by(keep_rarefied, plot_status) %>% tally()
d_final %>% filter(keep_rarefied == 1) %>% select(LAT, LON, plot_id) %>% distinct() %>% group_by(plot_id) %>% summarize(LAT = mean(LAT), LON = mean(LON)) %>% ggplot(aes(x = LON, y = LAT))+geom_point()
d_final %>% filter(keep_rarefied == 1, biome == 4) %>% select(LAT, LON, plot_id) %>% distinct() %>% group_by(plot_id) %>% summarize(LAT = mean(LAT), LON = mean(LON)) %>% ggplot(aes(x = LON, y = LAT))+geom_point()
d_final %>% filter(keep_rarefied == 1) %>% select(plot_id, biome, plot_status) %>% distinct() %>% group_by(biome, plot_status) %>% tally() %>% ungroup %>% group_by(biome) %>% mutate(n = n/sum(n)) %>% ungroup %>% filter(plot_status == "invaded")
dfilt %>% select(plot_id, biome, plot_status) %>% distinct() %>% group_by(biome, plot_status) %>% tally() %>% ungroup %>% group_by(biome) %>% mutate(n = n/sum(n)) %>% ungroup %>% filter(plot_status == "invaded")
d_final %>% filter(keep_rarefied == 1) %>% select(plot_id) %>% distinct() %>% nrow(.)
d_final %>% filter(keep_rarefied == 1) %>% select(LAT, LON, plot_status) %>% distinct() %>% ggplot(aes(x = LON, y = LAT, color = plot_status))+geom_point()
d_final %>% filter(keep_rarefied == 1) %>% select(plot_id, plot_status) %>% distinct() %>% group_by(plot_status) %>% tally()


if(UPSAMPLE){
  write_csv(d_final, "data/GFBI_biome_rarefaction_invasives.csv")
}else{
  write_csv(d_final, "data/GFBI_biome_rarefaction_invasives_noupsample.csv")
}

