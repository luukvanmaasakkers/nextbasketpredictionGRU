folder_path = "" # set the path to the folder containing the csv files downloaded from https://www.kaggle.com/c/instacart-market-basket-analysis  

aisles <- read.csv(paste(folder_path,"aisles.csv",sep=""))
departments <- read.csv(paste(folder_path,"departments.csv",sep=""))
order_products_prior <- read.csv(paste(folder_path,"order_products__prior.csv",sep=""))
order_products_train <- read.csv(paste(folder_path,"order_products__train.csv",sep=""))
orders <- read.csv(paste(folder_path,"orders.csv",sep=""))
products <- read.csv(paste(folder_path,"products.csv",sep=""))

# Load required packages
library(dplyr)
library(lsa)

# aisle_id column is added to order_products_prior
prod_aisles <- products[,c(1,3)]
orders_aisles_join <- left_join(order_products_prior,prod_aisles,by="product_id")
all_clusters <- integer()

THRESHOLD <- 500 # set minimal support for each cluster

# do the clustering for each aisle separately
for (AISLE in aisles$aisle_id) {
  
  orders_aisle_i <- orders_aisles_join[orders_aisles_join$aisle_id==AISLE,]
  products_aisle_i <- products[products$aisle_id==AISLE,]
  prodfreq_i <- table(factor(orders_aisle_i$product_id, levels = products_aisle_i$product_id)) 
  products_aisle_i <- products_aisle_i[prodfreq_i!=0,]
  prodfreq_i <- prodfreq_i[prodfreq_i!=0]
  freqs_i <- cbind(products_aisle_i$product_id,prodfreq_i)
  
  cat("Start clustering in aisle", AISLE,", containing", NROW(products_aisle_i), "products \n")
  
  # construct a matrix with all possible pairs of products in aisle 
  combinations <- matrix(0,NROW(products_aisle_i),NROW(products_aisle_i))
  orderlist <- list()
  count <- 0
  for (i in products_aisle_i$product_id) {                                       
    count <- count+1
    orderlist[[count]] <- orders_aisle_i$order_id[orders_aisle_i$product_id==i] # count orders that contain the product
  }
  
  for (i in 1:(count-1)) {
    combinations[i,i] <- 1/prodfreq_i[i]
    for (j in (i+1):count) {
      intersect <- NROW(intersect(orderlist[[i]],orderlist[[j]]))
      combinations[i,j] <- intersect/prodfreq_i[i]
      combinations[j,i] <- intersect/prodfreq_i[j]
    }
  }
  combinations[count,count] <- 1/prodfreq_i[count]
  
  cossim <- cosine(t(combinations)) # take cosine similarity of all rows, not all columns
  clusters <- 1:NROW(products_aisle_i) # assign each product to separate cluster
  clusterfreqs_i <- aggregate(prodfreq_i~clusters,data=cbind(freqs_i,clusters),FUN=sum)
  combinations_new <- combinations
  orderlist_new <- orderlist
  
  while (min(clusterfreqs_i[,2])<THRESHOLD) {
    numclust <- NROW(clusterfreqs_i)
    cossim_temp <- cossim
    diag(cossim_temp) <- NA
    cossim_temp <- cossim_temp[clusterfreqs_i[,2]<THRESHOLD,]
    smallclusters <- clusterfreqs_i[clusterfreqs_i[,2]<THRESHOLD,1]
    cat(NROW(smallclusters)," small clusters left  \r")
    
    index <- which(cossim_temp==max(cossim_temp,na.rm=TRUE))
    if (NROW(index)>1) {
      index <- index[1]
    }
    if (NROW(smallclusters)==1) {
      col_index <- index
      row_index <- 1
    } else {
      col_index <- ceiling(index/NROW(cossim_temp))
      row_index <- index %% NROW(cossim_temp)
      if (row_index == 0) {
        row_index <- NROW(cossim_temp)
      }
    }
    
    # two products/clusters to be merged
    product1 <- smallclusters[row_index]
    product2 <- clusterfreqs_i[col_index,1]
    
    first <- min(product1,product2)
    second <- max(product1,product2)
    first_pos <- which(clusterfreqs_i[,1]==first)
    second_pos <- which(clusterfreqs_i[,1]==second)
    
    #  cat("Merging ") # Turned off, can be turned on to print clustered products
    #  cat(products_aisle_i$product_name[first])
    #  cat(" & ")
    #  cat(products_aisle_i$product_name[second],"...\n")
    
    orderlist_new[[first]] <- union(orderlist_new[[first]],orderlist_new[[second]])
    clusterfreqs_i[first_pos,2] <- NROW(orderlist_new[[first]])
    combinations_new[first_pos,first_pos] <- 1/clusterfreqs_i[first_pos,2]
    
    for (i in 1:numclust) {
      group <- unique(clusters)[i]
      if (group != first) {
        intersect <- NROW(intersect(orderlist_new[[group]],orderlist_new[[first]]))
        combinations_new[i,first_pos] <- intersect/clusterfreqs_i[i,2]
        combinations_new[first_pos,i] <- intersect/clusterfreqs_i[first_pos,2]
      }
    }
    
    clusters[clusters==second] <- first
    clusterfreqs_i <- clusterfreqs_i[-second_pos,]
    combinations_new <- combinations_new[-second_pos,-second_pos]
    cossim <- cosine(t(combinations_new))
  }
  all_clusters <- rbind(all_clusters,cbind(products_aisle_i$product_id,products_aisle_i$product_id[clusters],AISLE))
}

write.csv(all_clusters,file="finalclusters.csv")



