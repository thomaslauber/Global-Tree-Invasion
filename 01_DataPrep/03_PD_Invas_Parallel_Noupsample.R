#####################################
#####################################
###### BUILD TREES FOR DATASET ######
######## ALL SPECIES & NATIVE #######
##### DOWNSAMPLE & NO UPSAMPLE ######
#####################################
#####################################

# !!!! This is written to run on a cluster, NOT a computer
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

#####################################
###### CALCULATE PD FOR DATASET #####
########### ALL SPECIES #############
#####################################

#####################################
############# Read dat ##############
#####################################

sp <- read.csv("data/GFBI_biome_rarefaction_invasives_noupsample.csv") %>%                                                   # Read condensed GFBI 
  filter(keep_rarefied == 1) %>%                                                                                 # Subset to rarefied data
  select(c("plot_id", "accepted_bin", "avglat", "avglon")) %>%                                                   # Subset to cols of interest
  mutate(accepted_bin = gsub(" ", "_",accepted_bin)) %>%                                                         # Add underscore
  distinct(accepted_bin, plot_id, .keep_all = TRUE) %>%                                                          # Drop duplicates of species and plot                                                                              
  as.data.frame()                                                                                                # Make a df

#####################################
######## Build tree per plot ########
#####################################

plot = unique(sp$plot_id)                                                                                        # Determine unique plot

load("data/GBOTB.extended.rda")                                                                                  # Load master plant phy
GBOTB.extended.congen <- congeneric.merge(GBOTB.extended,unique(sp$accepted_bin), split = "_", cite = FALSE)     # Merge congenerics
alltreesp <- c(GBOTB.extended.congen$tip.label) %>% as.data.frame()                                              # Convert phy to dataframe
colnames(alltreesp) <- "accepted_bin"                                                                            # Rename column

registerDoParallel(48)                                                                                           # Set number of parallel cores
system.time({PD <- foreach(i = 1:length(plot), .combine = c, .inorder = FALSE)%dopar%{                           # Set for each for each plot                                            
  P <- plot[i]                                                                                                   # Set current P to i
  currentSP = filter(sp, plot_id == P)                                                                           # Make a temporary loop df
  filename = paste("noupsampletrees/allsptrees/tree.allsp_", P, sep = "")                                                        # Set file names
  nodrop <- currentSP %>% select(accepted_bin)                                                                   # Select accepted_bin only
  todrop <- setdiff(alltreesp, nodrop) %>% as.vector()                                                           # Get the non overlapping species to drop
  todrop <- c(t(todrop))                                                                                         # Make into vector
  p.tree <- drop.tip(GBOTB.extended.congen, todrop)                                                              # Prune tree 
  if (is_empty(p.tree$tip.label)) {                                                                              # Check to see if tree is empty
    print(paste("empty tree for", P))                                                                            # If empty just return printed statement
  } else {                                                                                                       # Otherwise, write tree
    filepath = paste(filename, ".tre", sep = "")                                                                 # Set tree path
    write.tree(p.tree, filepath)                                                                                 # Write tree
  }
  return(NA) 
}
}
)

#####################################
######## PD metrics function ########
#####################################

get_metrics <- function(X){                                                                                      # Combined metrics function
  abund_mat <- matrix(nrow = 1,ncol = length(my_tree$tip.label)) %>% replace_na(1) %>% as.numeric()              # Make abundance matrix of 1 row to give equal weight to pd
  m1 <- faith_pd(abund_mat, X)                                                                                   # Run faith_pd to get sum of branch lengths
  D <- as.matrix(cophenetic(X))                                                                                  # Do the tree now so you don't have to calculate the distance a second time                                                            
  diag(D) <- Inf                                                                                                 # Reassign the diagonal and get the metrics; for M3 & M5 ignore diaglonal
  m2 <- mean(D[upper.tri(D)])                                                                                    # MPD; take mean of the upper triangle
  m3 <- mean(apply(D, 1, min))                                                                                   # MNTD; get mean of min distances for each row
  m4 <- var(D[upper.tri(D)])                                                                                     # VPD; get variance of the upper triangle     
  m5 <- var(apply(D, 1, min))                                                                                    # VNTD; get variance of min distances
  return(data.frame(type = c("faith", "mpd", "mntd", "vpd", "vntd"), value = c(m1, m2, m3, m4, m5)))             # Output the four values
}

#####################################
####### Calculate PD metrics ########
#####################################

trees <- list.files(path = "noupsampletrees/allsptrees") %>%                                                                     # Set tree paths
       str_replace("tree.allsp_", "") %>%                                                                        # Remove beginning of tree names
       str_replace(".tre", "")                                                                                   # Remove end of tree names = just plot_id with trees

PD <- data.frame()
for(T in trees){ 
    filename <- paste("noupsampletrees/allsptrees/tree.allsp_", T,".tre", sep = "")                                              # Get tree files
    my_tree <- read.tree(filename)                                                                               # Call this file "my_tree"
    if (my_tree$Nnode < 2) {                                                                                     # Check to see if tree has at least 2 tips
      print(paste("1 node for tree", T))                                                                         # If not just return printed statement
    } else {                                                                                                     # Otherwise
      mt <- get_metrics(my_tree)                                                                                 # Get metrics and save as df
      PD <- rbind(PD, mt %>% mutate(plot = T))                                                                   # Add to growing df of PD results
    }
}

saveRDS(PD, "PDmetrics_Allspecies_noupsample.rds")                                                                         # Write out file

#####################################
###### CALCULATE PD FOR DATASET #####
############# NATIVES ###############
#####################################

#####################################
############# Read dat ##############
#####################################

sp <- read.csv("data/GFBI_biome_rarefaction_invasives_noupsample.csv") %>%                                                   # Read condensed GFBI 
  filter(keep_rarefied == 1) %>%                                                                                 # Subset to rarefied data
  select(c("plot_id", "accepted_bin", "avglat", "avglon", "Glonaf_collapsed_status")) %>%                        # Subset to cols of interest
  distinct(accepted_bin, plot_id, .keep_all = TRUE) %>%                                                          # Drop duplicates of species and plot                                                  # select only unique per location species (combinations)
  filter(Glonaf_collapsed_status == "native")  %>%                                                               # Subset to only natives
  select(-c(Glonaf_collapsed_status)) %>%                                                                        # Remove native/invasive status
  mutate(accepted_bin = gsub(" ", "_", accepted_bin)) %>%                                                        # Add underscore
  as.data.frame()                                                                                                # Make a df

#####################################
######## Build tree per plot ########
#####################################

plot = unique(sp$plot)                                                                                           # Determine unique plot

registerDoParallel(48)                                                                                           # Set number of parallel cores
system.time({PD <- foreach(i = 1:length(plot), .combine = c, .inorder = FALSE)%dopar%{                           # Set for each for each plot                                            
  P <- plot[i]                                                                                                   # Set current P to i
  currentSP = filter(sp, plot_id == P)                                                                           # Make a temporary loop df
  filename = paste("noupsampletrees/nativesptrees/tree.nativesp_", P, sep = "")                                                  # Set file names
  nodrop <- currentSP %>% select(accepted_bin)                                                                   # Select accepted_bin only
  todrop <- setdiff(alltreesp, nodrop)%>% as.vector()                                                            # Get the non overlapping species to drop
  todrop <- c(t(todrop))                                                                                         # Make into vector
  p.tree <- drop.tip(GBOTB.extended.congen, todrop)                                                              # Prune tree 
  if (is_empty(p.tree$tip.label)) {                                                                              # Check to see if tree is empty
    print(paste("empty tree for", P))                                                                            # If empty just return printed statement
  } else {                                                                                                       # Otherwise, write tree
    filepath = paste(filename, ".tre", sep = "")                                                                 # Set tree path
    write.tree(p.tree, filepath)                                                                                 # Write tree
  }
  return(NA) 
}
}
)

#####################################
####### Calculate PD metrics ########
#####################################

trees<-list.files(path = "noupsampletrees/nativesptrees") %>%                                                                   # Set tree paths
  str_replace("tree.nativesp_", "") %>%                                                                         # Remove beginning of tree names
  str_replace(".tre", "")                                                                                       # Remove end of tree names = just plot_id with trees

PD <- data.frame()
for(T in trees){ 
  filename <- paste("noupsampletrees/nativesptrees/tree.nativesp_", T,".tre", sep = "")                                         # Get tree files
  my_tree <- read.tree(filename)                                                                                # Call this file "my_tree"
  if (my_tree$Nnode < 2) {                                                                                      # Check to see if tree has at least 2 tips
    print(paste("1 node for tree", T))                                                                          # If not just return printed statement
  } else {                                                                                                      # Otherwise
    mt <- get_metrics(my_tree)                                                                                  # Get metrics and save as df
    PD <- rbind(PD, mt %>% mutate(plot = T))                                                                    # Add to growing df of PD results
  }
}

saveRDS(PD, "PDmetrics_Nativespecies_noupsample.rds")                                                                            # Write out file
