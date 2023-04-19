#####################################
#####################################
###### BUILD TREES FOR DATASET ######
######## ALL SPECIES & NATIVE #######
##### DOWNSAMPLE & NO UPSAMPLE ######
#####################################
#####################################

# This is written to run on a cluster, but can run on a local computer
# All species includes all species in every plot,
# not just status identified species/plots

#####################################
###### Load packages; read dat ######
#####################################

library(ape)
packageVersion("ape")
library(tidyverse)
packageVersion("tidyverse")
library(abdiv)
packageVersion("abdiv")
library(doParallel)
packageVersion("doParallel")
library(foreach)
packageVersion("foreach")
library(pez)
packageVersion("pez")
library(feather)
packageVersion("feather")

set.seed(609)
traits <- read.csv("data/Species_trait_table.csv") %>%                                                           # Bring in species traits
  mutate(accepted_bin = gsub(" ", "_",accepted_bin)) %>%                                                         # Add underscore
  rename(trait = TraitID) %>%                                                                                    # Rename TraitID
  rowwise() %>%                                                                                                  # Group by row
  mutate(noise = runif(1,-1E-8,1E-8)) %>%                                                                        # Set range of variation of numbers to add
  ungroup() %>%                                                                                                  # Ungroup
  mutate(value = value + noise) %>%                                                                              # Add variation to values
  select(-noise)
  
keep_traits <- c(4, 6, 14, 15, 24, 26, 3110, 3106)                                                               # Traits to keep

#####################################
###### CALCULATE FD FOR DATASET #####
########### ALL SPECIES #############
#####################################

#####################################
############# Read dat ##############
#####################################

sp <- read.csv("data/GFBI_biome_rarefaction_invasives_noupsample.csv") %>%                                                  # Read condensed GFBI 
  filter(keep_rarefied == 1) %>%                                                                                 # Subset to rarefied data
  select(c("plot_id", "accepted_bin", "avglat", "avglon")) %>%                                                   # Subset to cols of interest
  mutate(accepted_bin = gsub(" ", "_",accepted_bin)) %>%                                                         # Add underscore
  left_join(traits, by = c("accepted_bin")) %>%                                                                  # Join with traits
  filter(trait %in% keep_traits) %>%                                                                             # Filter only traits of interest
  distinct(accepted_bin, plot_id, trait, value, .keep_all=TRUE)  %>%                                             # Drop duplicates of species, plot,and trait     
  group_by(trait) %>% mutate(value = (value - mean(value))/sd(value)) %>% ungroup() %>%                          # Standardize each trait
  as.data.frame()                                                                                                # Make a df

#####################################
######## FD metrics function ########
#####################################

get_metrics <- function(X){                                                                        # Combined metrics function
  D <- as.matrix(dist(X))                                                                          # Build tree-like matrix                                                       
  hc <- hclust(as.dist(D))                                                                         # Cluster the traits/distance; cluster object
  my_tree <- as.phylo.hclust(hc)                                                                   # Convert to a tree using as.hclust.phylo; dendrogram
  abund_mat <- matrix(nrow=1,ncol=length(my_tree$tip.label)) %>% replace_na(1) %>% as.numeric()    # Make abundance matrix of 1 row to give equal weight to pd
  m1 <- faith_pd(abund_mat, my_tree)                                                               # Run faith_pd to get sum of branch lengths
  diag(D) <- Inf                                                                                   # Now reassign the diagonal and get the metrics
  m2 <- mean(D[upper.tri(D)])                                                                      # MPD; take mean of the upper triangle
  m3 <- mean(apply(D, 1, min))                                                                     # MNTD; get mean of min distances
  m4 <- var(D[upper.tri(D)])                                                                       # VPD; get variance of the upper triangle     
  m5 <- var(apply(D, 1, min))                                                                      # VNTD; get variance of min distances
  return(data.frame(type = c("faith","mpd", "mntd","vpd", "vntd"), value = c(m1, m2, m3, m4, m5))) # output the four values
}

#####################################
####### Calculate FD metrics ########
#####################################

plot = unique(sp$plot_id)  

registerDoParallel(3) 
system.time({FD <- foreach(i = 1:length(plot), .combine = rbind, .inorder = FALSE)%dopar%{# For each plot id
  P <- plot[i]
  print(P)                                                                                # Print plot for trouble shooting
  tr <- sp %>% filter(plot_id == P) %>%                                                   # Make new df for traits of that plotyear                                                                                    
    select(accepted_bin, trait, value) %>%                                                # Select sp name, trait value, keep only distinct
    spread(trait, value) %>% select(-accepted_bin) %>%                                    # Spread data
    drop_na() %>%                                                                         # Remove rows with NAs
    as.matrix()                                                                           # Save as matrix
   if (ncol(tr) == 0 | nrow(tr) < 3) {                                                    # Check to see if df has zero cols/traits/is empty; also if nrow <3 (need to calculate metrics)
    return(tibble())                                                                      # If empty just return empty tibble
  } else {                                                                                # Otherwise, get metrics
    mt <- get_metrics(tr)                                                                 # Get metrics
    mt <- mt %>% mutate(plot_id = P)                                                      # Add plot id
    #return(mt)                                          
  }
}
})

saveRDS(FD, "data/FDmetrics_Allspecies_noupsample.rds")                                              # Write out file

#####################################
###### CALCULATE FD FOR DATASET #####
############# NATIVES ###############
#####################################

#####################################
############# Read dat ##############
#####################################

sp <- read.csv("data/GFBI_biome_rarefaction_invasives_noupsample.csv") %>%                                                   # Read condensed GFBI 
  filter(keep_rarefied == 1) %>%                                                                                 # Subset to rarefied data
  select(c("plot_id", "accepted_bin", "avglat", "avglon", "Glonaf_collapsed_status")) %>%                        # Subset to cols of interest
  filter(Glonaf_collapsed_status == "native")  %>%                                                               # Subset to only natives
  select(-c(Glonaf_collapsed_status)) %>%                                                                        # Remove native/invasive status
  mutate(accepted_bin = gsub(" ", "_",accepted_bin)) %>%                                                         # Add underscore
  left_join(traits, by = c("accepted_bin")) %>%                                                                  # Join with traits
  filter(trait %in% keep_traits) %>%                                                                             # Filter only traits of interest
  distinct(accepted_bin, plot_id, trait, value, .keep_all=TRUE)  %>%                                             # Drop duplicates of species, plot,and trait     
  group_by(trait) %>% mutate(value = (value - mean(value))/sd(value)) %>% ungroup() %>%                          # Standardize each trait
  as.data.frame()                                                                                                # Make a df

#####################################
####### Calculate FD metrics ########
#####################################

plot = unique(sp$plot_id)  

registerDoParallel(3) 
system.time({FD <- foreach(i = 1:length(plot), .combine = rbind, .inorder = FALSE)%dopar%{# For each plot id
  P <- plot[i]
  print(P)                                                                                # Print plot for trouble shooting
  tr <- sp %>% filter(plot_id == P) %>%                                                   # Make new df for traits of that plotyear                                                                                    
    select(accepted_bin, trait, value) %>%                                                # Select sp name, trait value, keep only distinct
    spread(trait, value) %>% select(-accepted_bin) %>%                                    # Spread data
    drop_na() %>%                                                                         # Remove rows with NAs
    as.matrix()                                                                           # Save as matrix
  if (ncol(tr) == 0 | nrow(tr) < 3) {                                                     # Check to see if df has zero cols/traits/is empty; also if nrow <3 (need to calculate metrics)
    return(tibble())                                                                      # If empty just return empty tibble
  } else {                                                                                # Otherwise, get metrics
    mt <- get_metrics(tr)                                                                 # Get metrics
    mt <- mt %>% mutate(plot_id = P)                                                      # Add plot id
    #return(mt)                                          
  }
}
})

saveRDS(FD, "data/FDmetrics_Nativespecies_noupsample.rds")                                           # Write out file
