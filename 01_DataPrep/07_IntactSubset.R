#####################################
#####################################
###### INTACT PROTECTED SUBSET ######
#####################################
#####################################

#####################################
###### Load packages; read dat ######
#####################################

library(tidyverse)
packageVersion("tidyverse")

# read in data for analyses
upsamplePD <- readRDS("data/GFBI_invasion_PDforanalyses.rds") 
upsampleFD <- readRDS("data/GFBI_invasion_FDforanalyses.rds") 

# subset to intact = 100%
upsamplePD.intact <- upsamplePD %>% filter(intact == 100)
upsampleFD.intact <- upsampleFD %>% filter(intact == 100)

#
write.csv(upsamplePD.intact, "data/GFBI_invasion_intactPDforanalyses.csv")
saveRDS(upsamplePD.intact, "data/GFBI_invasion_intactPDforanalyses.rds")

write.csv(upsampleFD.intact, "data/GFBI_invasion_intactFDforanalyses.csv")
saveRDS(upsampleFD.intact, "data/GFBI_invasion_intactFDforanalyses.rds")