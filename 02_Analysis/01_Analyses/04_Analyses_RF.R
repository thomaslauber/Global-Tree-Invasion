rm(list = ls())
gc()

library(fastshap)
library(ranger)
library(doParallel)
library(ggbeeswarm)
library(viridis)
library(colorspace)
library(caret)
library(fmsb)
library(cowplot)
library(geosphere)
require(caret)
require(AUC)
library(tidyverse)

# r2 for out of fit
coef_det <- function(xtrue, xpred){
	return(1-sum((xtrue-xpred)^2)/sum((xtrue-mean(xtrue))^2))
}
# prediction function for fastshap
pfun <- function(object, newdata) {
	predict(object, data = newdata)$predictions
}

# number of threads for parallel ranger/fastshap
nthreads <-  12

# variable lookup table to match variable names with printed names
var_lookup <- cbind(c("RAC", "RAC"), 
					c("sprich", "Species richness"),
					c("nativect", "Native species richness"),
					c("MAP", "Mean annual precipitation"),
					c("MAT", "Mean annual temperature"),
					c("dist.ports",  "Distance to ports"),
					c("nat.faith", "Native phylogenetic richness"),
					c("nat.mntd", "Native phylogenetic redundancy"),
					c("popdensity", "Population density"),
					c("SG_Absolute_depth_to_bedrock", "Absolute bedrock depth"),
					c("SG_Coarse_fragments_000cm", "Coarse fragments"),
					c("SG_Sand_Content_000cm", "Sand content"),
					c("SG_Silt_Content_000cm", "Silt content"),
					c("SG_Soil_pH_H2O_000cm", "Soil pH")) %>% t() %>% data.frame() %>% setNames(c("var", "show_name")) %>% as_tibble()


# specify the outcome variables and display names, with \n included depending on where to break the labels when printing
yname <- c("instatus", "propinvasive", "pinvba", "d.mntd", 
		   "instatus", "propinvasive", "pinvba", 
		   "instatus", "propinvasive", "pinvba", "d.mntd",  
		   "instatus", 
		   "instatus", 
		   "instatus", "propinvasive", "pinvba", "d.mntd", 
		   "instatus", "propinvasive", "pinvba", "d.mntd")

yshow <- c("non-native presence", "non-native richness", "non-native abundance","Invasion strategy",
		   "non-native presence", "non-native richness", "non-native abundance",
		   "non-native presence", "non-native richness", "non-native abundance", "Invasion strategy", 
		   "non-native presence",
		   "non-native presence", 
		   "non-native presence", "non-native richness", "non-native abundance","Invasion strategy",
		   "non-native presence", "non-native richness", "non-native abundance","Invasion strategy")

# name of the models to read in
mod_files <- c("RACdf_noscale_global_dat.stat.mod.rds", "RACdf_noscale_global_dat.prop.mod.rds", "RACdf_noscale_global_dat.prop.modb.rds", "RACdf_noscale_dmntd_global_dat.prop.mod.rds",
			   "RACdf_noscale_natsprich_global_dat.stat.mod.rds", "RACdf_noscale_natsprich_global_dat.prop.mod.rds",  "RACdf_noscale_natsprich_global_dat.prop.modb.rds", 
			   "FDRACdf_noscale_global_dat.stat.mod.rds", "FDRACdf_noscale_global_dat.prop.mod.rds", "FDRACdf_noscale_global_dat.prop.modb.rds", "FDRACdf_noscale_dmntd_global_dat.prop.mod.rds",
			   "Intact_FDRACdf_noscale_global_dat.stat.mod.rds", 
			   "Intact_PDRACdf_noscale_global_dat.stat.mod.rds", 
			   "Noupsample_PDRACdf_noscale_global_dat.stat.mod.rds", "Noupsample_PDRACdf_noscale_global_dat.prop.mod.rds", "Noupsample_PDRACdf_noscale_global_dat.prop.modb.rds", "Noupsample_PDRACdf_noscale_dmntd_global_dat.prop.mod.rds",
			   "Noupsample_FDRACdf_noscale_global_dat.stat.mod.rds", "Noupsample_FDRACdf_noscale_global_dat.prop.mod.rds", "Noupsample_FDRACdf_noscale_global_dat.prop.modb.rds", "Noupsample_FDRACdf_noscale_dmntd_global_dat.prop.mod.rds")

registerDoParallel(nthreads)

# number of shap sims to run
nsim <- 100

# set the seed
set.seed(10)

# are we doing cross validation r2, or shap values?
do_cv <- FALSE

# distance for cross validation metrics
dseq <- 250
# dseq <- seq(250, 5000, by = 250)
dseq <- c(c(1, 10, 50, 100, 250, 500, 1000), seq(2000, 5000, by = 1000))

# change to your root directory for the nas
nas_directory <- "~/nas"

# for debugging
iii <- 1

# for saving r2's
big_r2 <- tibble()

if(length(dseq) > 1 & do_cv){
	yname <- yname[1:11]
}

for(iii in 1:length(yname)){
	
	# read in the data	
	dt0 <- readRDS(paste0(nas_directory, "/Camille/Invas_proj/RAC_dfs/", mod_files[iii])) %>% as.data.frame(drop = TRUE)  
	
	# cycle through and strip-off re-add the variables. needed since some were saved as lists and it makes tidyverse angry
	dt_use <- dt0 %>% select(names(dt0)[1]) %>% as_tibble()
	for(i in 2:ncol(dt0)){
		dt_use <- dt_use %>% bind_cols(tibble(tmp = dt0 %>% select(names(dt0)[i]) %>% as_tibble() %>% unlist()) %>% setNames(names(dt0)[i]))
	}
	
	# clean up / rename the variables
	dt_use <- dt_use %>% 
		rename(LAT = lat, LON = avglon) %>% 
		mutate(propinvasive = ifelse(propinvasive == 0, NA, propinvasive)) %>% 
		mutate(invasct = ifelse(invasct == 0, NA, invasct)) %>%
		mutate(instatus = ifelse(in.stat == "invaded", 1, 0)) %>%
		mutate(popdensity = log(popdensity+0.1)) %>% 
		mutate(pinvba = ifelse(instatus == 0, 0, pinvba)) %>% 
		mutate(propinvasive = ifelse(propinvasive == 0, 0, propinvasive)) %>% 
		mutate(d.mntd = ifelse(instatus == 0, NA, d.mntd)) %>% 
		rename(yvar = all_of(yname[iii])) %>% 
		filter(!is.na(yvar)) %>% 
		mutate(id = 1:nrow(.))
	
	if(grepl("FD", mod_files[iii], ignore.case = FALSE)){
		var_lookup$show_name[var_lookup$var == "nat.faith"] <- "Native functional richness"
		var_lookup$show_name[var_lookup$var == "nat.mntd"] <- "Native functional redundancy"
	}else{
		var_lookup$show_name[var_lookup$var == "nat.faith"] <- "Native phylogenetic richness"
		var_lookup$show_name[var_lookup$var == "nat.mntd"] <- "Native phylogenetic redundancy"
	}
	
	
	# specify the predictor variables, depending on the model
	if(yname[iii] != "d.mntd"){
		evars <- c("nat.faith", 
				   "nat.mntd", 
				   "dist.ports", 
				   "popdensity", 
				   "MAT", 
				   "MAP", 
				   "SG_Absolute_depth_to_bedrock", 
				   "SG_Coarse_fragments_000cm", 
				   "SG_Sand_Content_000cm", 
				   "SG_Silt_Content_000cm")
		
	}else{
		evars <- c("MAP", 
				   "MAT", 
				   "dist.ports", 
				   "nat.faith", 
				   "popdensity", 
				   "SG_Absolute_depth_to_bedrock", 
				   "SG_Coarse_fragments_000cm", 
				   "SG_Sand_Content_000cm", 
				   "SG_Soil_pH_H2O_000cm")
	}
	
	# if doing sprich, remove the phylo variables
	if(grepl("sprich", mod_files[iii])){
		evars[evars == "nat.faith"] <- "nativect"
		evars <- evars[evars != "nat.mntd"]
	}
	
	# since we are averaging, convert back to binary for lat long
	if(yname[iii] == "instatus"){
		dt_use$yvar <- as.numeric(dt_use$yvar > 0)
	}
	
	# are we calculating R2 based on spatial cross validation?
	if(do_cv == TRUE){
		
		# get the unique lat longs and add in and id
		ll <- dt_use %>% select(LAT, LON) %>% distinct() %>% mutate(llid = 1:nrow(.)) 
		dt_use <- dt_use %>% left_join(ll)
		
		# do spatial clustering based on distance
		env_mat <- distm(ll %>% select(LON, LAT))/1000
		rownames(env_mat) <- colnames(env_mat) <- ll$llid

		for(ds in dseq){
			set.seed(10)
			dt <- dt_use %>% mutate(yvar = as.numeric(yvar))

			res <- resrac <- tibble()
			
			# do the spatially clustered CV
			nboot <- ifelse(length(dseq) == 1, 500, 100)
			
			for(m in 1:nboot){
				print(paste0(iii," of ",length(yname)," files -- dist = ", ds, " -- group ", m," of ",nboot))
				
				# get a focal location
				my_ll <- ll %>% sample_n(1) %>% select(llid) %>% unlist()
				test <- dt %>% filter(llid == my_ll)
				
				if(nrow(test) > 0){
					
					far_ll <- as.numeric(names(env_mat[as.character(my_ll), env_mat[as.character(my_ll),] > ds]))
					
					# test data
					train <- dt %>% filter(llid %in% far_ll) %>% group_by(LAT, LON) %>% sample_n(1) %>% ungroup 
					
					# train <- train %>% left_join(train %>% group_by(yvar) %>% tally() %>% mutate(wt = sum(n) / n) %>% select(-n), by = "yvar")
					if(nrow(train) > 0){
						# fit the model
						rfit <- ranger(yvar~., data = train %>% select(yvar, all_of(evars), rac), seed = 10, oob.error = FALSE, num.threads = nthreads)
						# predict
						res <- res %>% bind_rows(tibble(obs = as.numeric(test$yvar), pred = as.numeric(predict(rfit, data = test)$prediction), boot = m, ntest = nrow(test), ntrain = nrow(train), buffer_dist = ds))

						# just a check in case there's a prediction type mismatch
						if(any(is.na(res$obs)) | any(is.na(res$pred))){
							stop("NA is output")
						}
					}
				}
			}	
			
			res <- res %>% mutate(obs = as.numeric(as.character(obs)), pred = as.numeric(as.character(pred)))
			# initialize the r2 output table
			my_r2 <- tibble(outcome = yname[iii], r2 = NA, AUC = NA, kappa = NA, accuracy = NA, buffer_dist = ds)
			# get the accuracy
			if(length(unique(dt$yvar))>3){
				my_r2$r2 <- coef_det(res$obs, res$pred)
			}else{
				# my_r2$r2 <- coef_det(res$obs, res$pred)
				my_r2$AUC <- AUC::auc(AUC::accuracy(res$pred, as.factor(res$obs)))
				res <- res %>% mutate(pred = as.factor(ifelse(pred > 0.5, 1, 0)), obs = as.factor(obs))
				my_r2$kappa <- caret::postResample(pred = res$pred, obs = res$obs)[2]
				my_r2$accuracy <- caret::postResample(pred = res$pred, obs = res$obs)[1]
			}
			
			# append the results to the output
			big_r2 <- big_r2 %>% bind_rows(my_r2 %>% mutate(rac = TRUE, sprich = grepl("sprich", mod_files[iii]), FD = grepl("FD", mod_files[iii]), Intact = grepl("Intact", mod_files[iii]), Noupsample = grepl("Noupsample", mod_files[iii])))
		}
		
	}else{
		
		# just do the shap plots, not CV
		
		dt <- dt_use

		# reinitialize seed
		set.seed(15)
		
		# number of out-of-fit 
		ntest <- 1000
		
		if(nrow(dt %>% select(LAT, LON) %>% distinct()) > 3000){
			ntest <- 1000
		}else{
			ntest <- round(nrow(dt %>% select(LAT, LON) %>% distinct())*1/3)
		}
		# separate into a test/train splits. here it's also sampling only one point per cluster
		test_data <- dt %>%
			group_by(LAT, LON) %>% sample_n(1) %>% ungroup %>%
			sample_n(ntest)
		
		train_data <- dt %>%
			filter(!id%in%test_data$id) %>% group_by(LAT, LON) %>% sample_n(1) %>% ungroup
		
		
		# Fit the random forest models to the training data
		r1 <- ranger(yvar~., data = train_data %>% select(yvar, all_of(evars), rac), num.threads = nthreads) #, importance = "permutation")
		
		#  Estimate the shapley values on the test data
		fs1 <- fastshap::explain(r1, X = test_data %>% select(all_of(evars), rac) %>% as.matrix(), pred_wrapper = pfun, nsim = nsim, .parallel = TRUE, adjust = TRUE)
		
		# combine the models and merge in the environmental covariates
		shap_df <- bind_rows(fs1 %>% as_tibble() %>% mutate(id = test_data$id) %>% gather(var, shap, -id) %>% mutate(type = "shap")) %>% 
			left_join(test_data %>% dplyr::select(id, rac, all_of(evars)) %>% gather(var, value, -id))
		
		# select the axis and clean up some of the variables
		my_df <- shap_df %>% left_join(var_lookup) %>% rename(var_short = var) %>% rename(var = show_name) %>% 
			filter(var!="RAC")
		
		# sort the variables by sum(abs(shap value))
		shap_ord <- my_df %>% group_by(var) %>% summarize(feature_imp = mean(abs(shap))) %>% ungroup %>% arrange(desc(feature_imp)) %>% select(var) %>% unlist() %>% as.character()
		
		# colors and shapes for plotting
		col1 <- "#1A85FF"
		col2 <- "#D41159"
		my_bins <- 15
		my_shape <- 23
		my_cols <- c("red", "darkblue", "orange", "darkslategray3")
		
		# ylims for beeswarm plot
		ylim <- c(-0.25,0.25)
		
		# create the box importance plot
		varimp_box <- my_df %>% group_by(var) %>% summarize(shap = mean(abs(shap))) %>% ungroup %>%
			mutate(var = factor(var, levels = rev(shap_ord))) %>% 
			ggplot(aes(x = var, y = shap))+
			geom_col(fill = "#440154ff")+
			coord_flip()+
			theme_bw()+
			ylab(paste0("Mean absolute SHAP value"))+
			scale_y_continuous(expand = c(0,0))+
			theme( axis.title.y = element_blank(),axis.title.x = element_text(size = 14),axis.text.y = element_text(size = 12),
				   panel.border = element_blank(), panel.grid.major = element_blank(),
				   panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
		
		# what variables to plot and which to logIslington, London, UK
		if(yname[iii] == "d.mntd"){
			var_pick <- c("nat.faith", "MAT", "MAP")
			logfig <- c(T,F,F)
		}else{
			var_pick <- c("dist.ports", "nat.faith", "nat.mntd")
			logfig <- c(T,T,F)
		}
		
		# if doing richness, select different variables
		if(grepl("sprich", mod_files[iii])){
			var_pick[var_pick == "nat.faith"] <- "nativect"
			# comment out this next line if nat.mntd is supposed to be in the richness models
			var_pick <- var_pick[var_pick!="nat.mntd"]
		}
		
		# degree of the smoothing polynomial for shap curves
		deg <- c(2, 2, 2)
		
		# number of variables to plot
		numvars <- length(var_pick)
		
		# blank plots for top 3 predictors of each axis
		gg <- ggcurve <- vector(mode = "list", length = 1*numvars)
		
		# get symmetrical ylimits for the variables to be plotted
		plot_df <- my_df %>% filter(var_short %in% var_pick)
		ylims <- plot_df %>% filter(!is.na(shap)) %>% 
			summarize(min(shap), max(shap)) %>% unlist()
		ylims <- c(-max(abs(ylims)), max(abs(ylims)))
		
		# cycle through the top 3 and create the subplot
		for(i in 1:numvars){
			
			# subset the data to the current variable
			sub_df <- my_df %>% filter(var_short == var_pick[i])
			
			# get the x range and create the x sequence. right now set up with a trycatch to debug. shouldn't be needed
			x_range <- tryCatch(sub_df %>% select(value) %>% distinct() %>% summarize(max = max(value), min = min(value)) %>% unlist %>% as.numeric %>% sort(), warning = function(w) return(NULL))
			
			# did we get an error?
			if(is.null(x_range)){
				stop("found error")
			}
			
			# scale the alpha transparencies, if desired
			aluse <- c(1,1)
			
			# create the plot
			ggtmp <- 
				ggplot(data = sub_df, aes(x = value, y = shap))+
				geom_point(color = "#440154ff", alpha = 0.15)+
				geom_hline(yintercept = 0, linetype = 2, color = "gray50")+
				theme_bw()+
				scale_y_continuous(expand = c(0, 0), name = paste0("Influence on\n", yshow[iii])) + 
				scale_color_viridis()+
				coord_cartesian(ylim = ylims)+
				scale_fill_viridis()+
				scale_alpha_manual(values = aluse)
			
			my_name <- gsub("functional ", "", gsub("phylogenetic ", "", gsub("Distance to ports", "Distance to ports (km)", var_lookup$show_name[var_lookup$var == var_pick[i]])))
			
			# are we logging the x axis?
			if(logfig[i]){
				ggtmp <- ggtmp + scale_x_log10(expand = c(0, 0), name = my_name)
			}else{
				ggtmp <- ggtmp + scale_x_continuous(expand = c(0, 0), name = my_name)
			}
			
			# remove the legend
			gg[[i]] <- ggtmp + theme(legend.position = "none",
									 plot.margin = unit(c(3,20,3,3), "pt"))
			ggcurve[[i]] <- ggtmp + geom_smooth(method = "lm", formula = as.formula(paste0("y~poly(x, ",deg[i],", raw = TRUE)")), color = "black")+
				theme(legend.position = "none",  plot.margin = unit(c(3,20,3,3), "pt"),
					  axis.title = element_text(size = 14))	
		}
		
		# create the subplots
		p2 <- plot_grid(plotlist = ggcurve, nrow = 1, byrow = TRUE)	
		
		# setup the print label
		print_lab <- ifelse(grepl("sprich", mod_files[iii]), paste0(yname[iii], "_sprich"), yname[iii])
		print_lab <- ifelse(grepl("FD", mod_files[iii], ignore.case = FALSE), paste0(print_lab, "_FD"), print_lab)
		print_lab <- ifelse(grepl("Intact", mod_files[iii], ignore.case = FALSE), paste0(print_lab, "_Intact"), print_lab)
		print_lab <- ifelse(grepl("Noup", mod_files[iii], ignore.case = FALSE), paste0(print_lab, "_Noupsample"), print_lab)
		
		# save the plots
		ggsave(paste0(nas_directory, "/Camille/Invas_proj/shap_figs_final/REV_rac_shap_curves_",print_lab,".png"), plot = p2, device = "png", dpi = 600, width = 10, height = 3, units = "in")
		ggsave(paste0(nas_directory, "/Camille/Invas_proj/shap_figs_final/REV_rac_imp_",print_lab,".png"), plot = varimp_box, device = "png", dpi = 600, width = 6, height = 4, units = "in")
	}
}

# save the r2 table, if doing cv
if(do_cv){
	if(length(dseq) == 1){
		write_csv(big_r2, paste0(nas_directory, "/Camille/Invas_proj/shap_figs_final/REV_model_fit_r2.csv"))
	}else{
		write_csv(big_r2, paste0(nas_directory, "/Camille/Invas_proj/shap_figs_final/REV_model_fit_r2_distance.csv"))
	}
	
	(g1 <- ggplot(big_r2 %>% select(r2, AUC, kappa, accuracy, buffer_dist) %>%
				  	gather(Metric, value, -buffer_dist) %>% mutate(Metric = ifelse(Metric == "AUC", "AUC", str_to_sentence(Metric))),
				  aes(x = buffer_dist, y = value)) + 
			facet_wrap(~Metric, scales = "free") + 
			geom_point() +
			geom_smooth() +
			xlab("Buffer Distance (km)")+theme_bw())
	
	ggsave(filename = "~/nas/Camille/Invas_proj/shap_figs_final/REV_accuracy_buffer_dist.png", plot = g1, height = 7, width = 7, device = "png", units = "in")
}
