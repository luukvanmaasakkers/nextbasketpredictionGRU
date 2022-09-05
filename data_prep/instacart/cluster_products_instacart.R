# Load required packages
library(dplyr)
library(lsa)
library(Matrix)
library(methods)

folder_path = "" # set the path to the folder containing the csv files downloaded from https://www.kaggle.com/c/instacart-market-basket-analysis  

aisles <- read.csv(paste(folder_path,"aisles.csv",sep=""))
departments <- read.csv(paste(folder_path,"departments.csv",sep=""))
order_products_prior <- read.csv(paste(folder_path,"order_products__prior.csv",sep=""))
order_products_train <- read.csv(paste(folder_path,"order_products__train.csv",sep=""))
orders <- read.csv(paste(folder_path,"orders.csv",sep=""))
products <- read.csv(paste(folder_path,"products.csv",sep=""))

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

set.seed(12345)

# The imported matrix all_clusters contains two columns: the first column represents product_id, the
# second column cluster_id. Columns are not sorted and not all cluster_id's exist.

# Sort the unique cluster_id's from low to high
clusterlist <- sort(unique(all_clusters[,2]))

# Assign a cluster_id to each product (this time, cluster_id ranges from 1 to #clusters (9407))
clean_clusters <- integer(NROW(all_clusters))
for (i in 1:NROW(all_clusters)) {
  clean_clusters[i] <- which(clusterlist == all_clusters[i,2])
}

# Order the products from low product_id to high
final_clusters <- cbind(all_clusters[,1],clean_clusters)
final_clusters <- final_clusters[order(final_clusters[,1]),]

# Add clusters that are not in the prior data and assign cluster_id to zero again. Finally, sort
# all products again such that product_id ranges from 1 to #products (49688) without gaps
max_id <- max(products$product_id)
not_in_train <- setdiff(c(1:max_id),all_clusters[,1])
final_clusters <- rbind(final_clusters,cbind(not_in_train,0))
final_clusters <- final_clusters[order(final_clusters[,1]),]

# Select only 5% of customers, to reduce number of rows
FRACTION <- 1
users <- unique(orders$user_id)
user_selection <- sort(sample(users,FRACTION*NROW(users)))
order_selection <- orders[orders$user_id %in% user_selection,]

# Determine which baskets are the last for each of the customers in user_selection, set eval_set 
# to 'test' for these baskets. Remove original test observations.
order_selection <- order_selection[order_selection$eval_set!='test',]

order_num <- order_selection$order_number[1:NROW(order_selection)-1]
order_num_next <- order_selection$order_number[2:NROW(order_selection)]
order_selection$eval_set[order_num > order_num_next] <- 'last'
order_selection$eval_set[NROW(order_selection)] <- 'last'

# Assign half of the last orders to the test set, other half to the validation set
last_orders <- order_selection$order_id[order_selection$eval_set=='last']
val_selection <- sort(sample(last_orders, 0.5*NROW(last_orders)))
order_selection$eval_set[order_selection$order_id %in% val_selection] <- 'val'
order_selection$eval_set[order_selection$eval_set=='last'] <- 'test'

train_selection <- order_products_prior[order_products_prior$order_id %in% order_selection$order_id[order_selection$eval_set=='prior'],]
val_selection1 <- order_products_prior[order_products_prior$order_id %in% order_selection$order_id[order_selection$eval_set=='val'],]
val_selection2 <- order_products_train[order_products_train$order_id %in% order_selection$order_id[order_selection$eval_set=='val'],]
val_selection <- rbind(val_selection1,val_selection2)
test_selection1 <- order_products_prior[order_products_prior$order_id %in% order_selection$order_id[order_selection$eval_set=='test'],]
test_selection2 <- order_products_train[order_products_train$order_id %in% order_selection$order_id[order_selection$eval_set=='test'],]
test_selection <- rbind(test_selection1,test_selection2)

# Replace product_id by cluster_id
training_set <- train_selection
train_orders <- order_selection$order_id[order_selection$eval_set=='prior']
training_set$product_id <- final_clusters[train_selection$product_id,2]
validation_set <- val_selection
val_orders <- order_selection$order_id[order_selection$eval_set=='val']
validation_set$product_id <- final_clusters[val_selection$product_id,2]
test_set <- test_selection
test_orders <- order_selection$order_id[order_selection$eval_set=='test']
test_set$product_id <- final_clusters[test_selection$product_id,2]

# Check which products have cluster_id 0 and thus are never bought in the training data, exclude
# these products from their baskets as they are not assigned to a cluster
val_bought <- validation_set$product_id>0
test_bought <- test_set$product_id>0
validation_set <- validation_set[val_bought,]
test_set <- test_set[test_bought,]

all_purchases <- rbind(training_set,validation_set,test_set)
all_coo23 <- all_purchases[,1:2]
coo1 <- order_selection[,1:4]
all_coordinates <- merge(coo1,all_coo23,by='order_id')
all_coordinates <- all_coordinates[,c(2,4,5,3,1)]
all_coordinates <- all_coordinates[order(all_coordinates[,1],all_coordinates[,2]),]

all_coordinates[,c(1,2,3,5)] <- all_coordinates[,c(1,2,3,5)]-1
all_coordinates$eval_set[all_coordinates$eval_set=='prior'] <- 0
all_coordinates$eval_set[all_coordinates$eval_set=='val'] <- 1
all_coordinates$eval_set[all_coordinates$eval_set=='test'] <- 2
all_coordinates$eval_set <- as.numeric(all_coordinates$eval_set)

order_selection$eval_set[order_selection$eval_set=='prior'] <- 0
order_selection$eval_set[order_selection$eval_set=='val'] <- 1
order_selection$eval_set[order_selection$eval_set=='test'] <- 2
order_selection$eval_set <- as.numeric(order_selection$eval_set)
order_selection$order_hour_of_day <- as.numeric(order_selection$order_hour_of_day)
order_selection$days_since_prior_order[is.na(order_selection$days_since_prior_order)] <- -1
order_selection$days_since_first_order <- integer(NROW(order_selection))

order_selection$days_since_first_order[order_selection$order_number==2] = order_selection$days_since_prior_order[order_selection$order_number==2]
for (i in 3:max(order_selection$order_number)) {
  indices = which(order_selection$order_number==i)
  order_selection$days_since_first_order[indices] =
    order_selection$days_since_first_order[indices-1] + order_selection$days_since_prior_order[indices]
}

night = c(23,0,1,2,3,4,5)
morning = c(6,7)
day = 8:20
evening = c(21,22)
order_selection$part_of_day <- order_selection$order_hour_of_day
order_selection$part_of_day[order_selection$part_of_day %in% night] = 0
order_selection$part_of_day[order_selection$part_of_day %in% morning] = 1
order_selection$part_of_day[order_selection$part_of_day %in% day] = order_selection$part_of_day[order_selection$part_of_day %in% day]-6
order_selection$part_of_day[order_selection$part_of_day %in% evening] = 15

order_selection$part_of_week = order_selection$order_dow*16 + order_selection$part_of_day

write.csv(all_coordinates,"all_3dcoordinates.csv",row.names=FALSE)
write.csv(order_selection,file="order_selection.csv",row.names=FALSE)
