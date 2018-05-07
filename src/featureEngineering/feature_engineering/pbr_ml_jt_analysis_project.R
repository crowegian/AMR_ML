# setwd('/Users/nenadmacesic2/Dropbox/emacs/sequence_tut/pbr/R/pbr_ml_project/group_project_final/')
setwd('/home/ob2285/Desktop/currentClasses/compGen_S18/finalProject/AMR_ML/src/featureEngineering/feature_engineering/')


# some of the packages below might give you some issues when attaching and installing dependencies. 
# The installs were tested on Ubuntu 16.04. The tidyverse installation and the treewas installation
# are the most difficult to install, but once dependencies are installed the commands below
# should work.
library(tidyverse)
library(reshape2)
install.packages("devtools", dep=TRUE)
library(devtools)

## install treeWAS from github:
install_github("caitiecollins/treeWAS/pkg", build_vignettes = TRUE)
library(treeWAS)
library(caret)

####Import datasets####
#Annotation file
annot_file <- read_csv('KP0228_no_mge.csv', col_types = cols(.default = col_character()),
                       col_names = TRUE) %>%
  select(LOCUS_TAG = locus_tag, GENE = Name) %>%
  filter(str_detect(GENE, "CDS"))

##METADATA
#CUMC
pbr_metadata <- read_csv('pbr_metadata_4_20180411.csv', col_names = TRUE) %>%
                mutate(bmd_final_res = ifelse(bmd_final>2, 1, 0),
                       poly_etest_res = ifelse(poly_etest>2, 1, 0),
                       poly_mic = bmd_final,
                       poly_mic = ifelse(is.na(poly_mic), poly_etest, poly_mic),
                       pbr_res = ifelse(poly_mic>2, 1, 0),
                       susc_change = ifelse(poly_etest_res==bmd_final_res, 0, 1))

# pbr_metadata_2 <- select(pbr_metadata_susc, isolate, pbr_res) %>%
#                   filter(isolate %in% pbr_isolate_st258$isolate)
pbr_isolates_to_include <- read_csv('pbr_st258_isolates_snippy_20180411.csv', col_names = TRUE)


pbr_metadata_2 <- select(pbr_metadata, isolate, pbr_res) %>%
                  filter(isolate %in% pbr_isolates_to_include$isolate) %>%
                  mutate(dataset = "cumc")


#DeLeo
deleo_metadata <- read_csv('deleo_metadata.csv', col_names = TRUE)
colnames(deleo_metadata) <- str_to_lower(str_replace_all(colnames(deleo_metadata), '[^a-zA-Z0-9]', "_"))
colnames(deleo_metadata) <- str_trim(colnames(deleo_metadata))

deleo_metadata <- mutate(deleo_metadata, colistin = ifelse(str_detect(colistin, ">[ ]?4"), "5", colistin),
                         colistin = ifelse(str_detect(colistin, "<=[ ]?0.25"), "0.24", colistin),
                         polymyxin_b = ifelse(str_detect(polymyxin_b, ">[ ]?4"), "5", polymyxin_b),
                         polymyxin_b = ifelse(str_detect(polymyxin_b, "<=[ ]?0.25"), "0.24", polymyxin_b)) %>%
                  mutate_at(vars(colistin, polymyxin_b), funs(as.numeric))

colnames(deleo_metadata)[1] <- c("SampleName")

deleo_susceptibility <- select(deleo_metadata, SampleName, colistin, polymyxin_b) %>%
                        mutate(pbr_res = ifelse(colistin>2|polymyxin_b>2, 1, 0))

deleo_runinfo <- read_csv('deleo_runinfo.csv', col_names = TRUE) %>%
                 select(Run, SampleName)

#Some DeLeo isolates did not assemble, so removed them
deleo_spades_assemblies <- read_csv('deleo_st258_isolates_20180417.csv', col_names = TRUE) %>%
  mutate(isolate = str_split_fixed(isolate, "\\.", n = 2)[, 1])


deleo_metadata_2 <- merge(x = deleo_susceptibility, y = deleo_runinfo,
                        all.x = TRUE, by = "SampleName") %>%
                    mutate(Run = ifelse(SampleName=="NJST258_2", "SRR5387142", Run)) %>%
                    rename(isolate = Run) %>%
                    select(isolate, pbr_res) %>%
                    filter(isolate %in% deleo_spades_assemblies$isolate) %>%
                    mutate(dataset = "deleo")

#Wright
wright_metadata <- read_csv('wright_metadata.csv', col_names = TRUE)
colnames(wright_metadata) <- str_to_lower(str_replace_all(colnames(wright_metadata), '[^a-zA-Z0-9]', "_"))
colnames(wright_metadata) <- str_trim(colnames(wright_metadata))

wright_runinfo <- read_csv('wright_runinfo.txt', col_names = TRUE) %>%
                  filter(Run != "Run",
                         LibraryStrategy=="WGS"&Platform=="ILLUMINA") %>%
                  select(isolate = Run, bioproject = BioProject)
wright_downloaded <- read_csv('wright_downloaded.csv', col_names = FALSE) %>%
                     rename(isolate = X1) %>%
                     mutate(isolate = str_split_fixed(isolate, "\\.", n = 2)[, 1],
                            downloaded = 1)
wright_metadata <- left_join(wright_metadata, wright_runinfo, by = "bioproject") %>%
                     left_join(., wright_downloaded, by = "isolate") %>%
                     filter(mlst_sequence_type=="258") %>%
                     mutate(colistin = ifelse(str_detect(colistin, ">[ ]?4"), "5", colistin),
                            colistin = ifelse(str_detect(colistin, "<=[ ]?0.25"), "0.24", colistin),
                            polymixin_b = ifelse(str_detect(polymixin_b, ">[ ]?4"), "5", polymixin_b),
                            polymixin_b = ifelse(str_detect(polymixin_b, "<=[ ]?0.25"), "0.24", polymixin_b)) %>%
                     mutate_at(vars(colistin, polymixin_b), funs(as.numeric)) %>%
                     mutate(pbr_res = ifelse(colistin>2|polymixin_b>2, 1, 0),
                            dataset = "wright")

wright_metadata_2 <- select(wright_metadata, isolate, pbr_res, dataset)

####Joint metadata file####
#This dictates what is included in the final analysis due to a right join
ml_metadata <- bind_rows(pbr_metadata_2, deleo_metadata_2, wright_metadata_2)
ml_metadata_2 <- filter(ml_metadata, !is.na(pbr_res))

####SEQ DATA####
####Snippy####
pbr_snippy_output <- read_csv('snippy_output_20180411.csv', col_names = TRUE,
                              col_types = cols(isolate = col_character()))

deleo_snippy_output <- read_csv('snippy_deleo_output_20180417.csv', col_names = TRUE,
                                col_types = cols(isolate = col_character()))

wright_snippy_output <- read_csv('snippy_wright_output_20180419.csv', col_names = TRUE,
                                col_types = cols(isolate = col_character()))

joint_snippy_output <- bind_rows(pbr_snippy_output, deleo_snippy_output, wright_snippy_output)

####ISseeker####
pbr_is_seeker_output <- read_csv('is_seeker_output_4_20180411.csv', col_names = TRUE) %>%
                        mutate(EVIDENCE = as.character(EVIDENCE))

deleo_is_seeker_output <- read_csv('deleo_is_seeker_output_4_20180417.csv', col_names = TRUE) %>%
                          mutate(EVIDENCE = as.character(EVIDENCE))

wright_is_seeker_output <- read_csv('wright_is_seeker_output_4_20180419.csv', col_names = TRUE) %>%
                          mutate(EVIDENCE = as.character(EVIDENCE))

joint_is_seeker_output <- bind_rows(pbr_is_seeker_output, deleo_is_seeker_output, wright_is_seeker_output)

####BLAST Db####
pbr_blast_db_output <- read_csv('pbr_res_genes_ins_seq_2_20180411.csv', col_names = TRUE)

deleo_blast_db_output <- read_csv('deleo_res_genes_ins_seq_2_20180417.csv', col_names = TRUE)

wright_blast_db_output <- read_csv('wright_res_genes_ins_seq_2_20180419.csv', col_names = TRUE)

joint_blast_db_output <- bind_rows(pbr_blast_db_output, deleo_blast_db_output, wright_blast_db_output)

####Rbind seq data####
#Includes all calls
#Can later filter depending on which dataset you want to run
seq_data <- bind_rows(joint_snippy_output, joint_is_seeker_output, joint_blast_db_output) %>%
            filter(isolate %in% ml_metadata$isolate) %>%
            arrange(isolate, POS) %>%
            mutate(GENE = ifelse(LOCUS_TAG=="mgrB", "mgrB", GENE)) %>%
            #Manually removed one DeLeo isolate which Snippy did not work on
            filter(isolate!="SRR1166976")

#Clean data to label intergenic regions and remove synonymous mutations
seq_data_2 <- mutate(seq_data, EFFECT = ifelse(is.na(EFFECT), "intergenic", EFFECT)) %>%
              filter(!str_detect(EFFECT, "^syn")) #Need to clarify what the 'intragenic mutations' are
                                                  #Can find as locus tags are isolate.csv

seq_data_3 <- left_join(seq_data_2, ml_metadata, by = "isolate")

analysis_isolates_all <- distinct(seq_data, isolate)

####Remove IS before/after gene####
#Keep all non-synonymous mutations
seq_data_ml_5 <- filter(seq_data_2, EFFECT!="intergenic",
                        !str_detect(EFFECT, "intragenic"),
                        !str_detect(EFFECT, 'before_gene'),
                        !str_detect(EFFECT, 'after_gene')) %>%
                 group_by(isolate, LOCUS_TAG) %>%
                 distinct(isolate, LOCUS_TAG, .keep_all = TRUE)

seq_data_ml_6 <- dcast(seq_data_ml_5, isolate~LOCUS_TAG, value.var = "EFFECT") %>%
                 right_join(., ml_metadata_2, by = "isolate") %>%
                 filter(!is.na(pbr_res)) %>%
                 select(-dataset)

seq_data_ml_bi_3 <- mutate_at(seq_data_ml_6, vars(KP0228_00001:mgrB),
                              funs(ifelse(is.na(.), 0, 1))) 

write_csv(seq_data_ml_bi_3, 'dataset_11_full.csv')
write_csv(ml_metadata_2, 'dataset_11_metadata.csv')

####Training/Validation set####
set.seed(1234)
seq_data_ml_bi_3_trainIndex <- createDataPartition(seq_data_ml_bi_3$pbr_res, 
                                                   p=0.75, list=FALSE, times=1)
seq_data_ml_bi_3_train <- seq_data_ml_bi_3[seq_data_ml_bi_3_trainIndex, ]
seq_data_ml_bi_3_test <- seq_data_ml_bi_3[-seq_data_ml_bi_3_trainIndex, ]

write_csv(seq_data_ml_bi_3_train, 'dataset_12_train.csv')
write_csv(seq_data_ml_bi_3_test, 'dataset_12_test.csv')

####GWAS####
####TreeWAS
#Categorical outcome
#treewas_isolates <- distinct(snp_matrix_melt, isolate_1)
rownames(seq_data_ml_bi_3_train) <- c()
treewas_dataset <- column_to_rownames(seq_data_ml_bi_3_train, var = "isolate") %>%
                   select(-pbr_res)
treewas_dataset <- as.matrix(treewas_dataset)

treewas_labels <- as.vector(unlist(seq_data_ml_bi_3_train$pbr_res))
names(treewas_labels) <- rownames(treewas_dataset)

treewas_cat_output <- treeWAS(treewas_dataset, treewas_labels, tree = c("BIONJ"))

treewas_cat_results <- as.data.frame(treewas_cat_output$terminal$corr.dat) %>%
                       rownames_to_column(., var = "LOCUS_TAG") %>%
                       #mutate(LOCUS_TAG = str_sub(allele, 1, 12)) %>%
                       left_join(., annot_file, by = "LOCUS_TAG")
colnames(treewas_cat_results)[2] <- c("corr_dat")
treewas_cat_results <- arrange(treewas_cat_results, desc(corr_dat))

write_csv(treewas_cat_results, 'dataset_12_gwas.csv')