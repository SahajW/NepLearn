import numpy as np
from collections import Counter      #datastructer that makes it easy to get the most common occurance of a certain value in a array 
from DecisionTree import DecisionTree

class RandomForest:

    def __init__(self,n_trees=10,max_depth=10,min_samples_split=2,n_features=None): 
        self.n_trees=n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_features
        self.trees = []
        """
        n_trees          : number of decision trees
        max_depth        : max depth of each tree
        min_samples_split: minimum samples required to split
        n_features       : number of features to consider per tree
        """
    #to train the randomforest  
    def fit(self,X,y):
        self.trees=[]                      #container inside object to store all the trained decision trees

        n_total_features = X.shape[1]
        # if feature count is not defined
        if self.n_features is None:
            self.n_features = int(np.sqrt(n_total_features))

        for _ in range(self.n_trees):

            #we are not going to fit this tree with all the samples we got, so we are doing sampling of X and y
            X_sample,y_sample = self._bootstrap_samples(X,y)
            
            #random feature selection
            feature_indices = np.random.choice(n_total_features,self.n_features,replace=False)
            
            #training tree using only selected features
            tree = DecisionTree(max_depth=self.max_depth,min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feature_indices],y_sample)                                       #: -> take ALL rows and feature_indices -> take ONLY these columns

            #storing both tree and its feature subset                                                                                       
            self.trees.append((tree,feature_indices))


    def _bootstrap_samples(self,X,y):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples,n_samples,replace=True)         #sampling with replacement . Some rows repeat, some are missing
        return X[idx],y[idx]
    
    #majoity voting 
    def most_common_label(self,y):
        counter = Counter(y)        #we creater a counter datastructure
        most_common = counter.most_common(1)[0][0]         #counter.most_common(1) returns a list of tuples with the most common element
        return most_common                                 #eg:[(1, 3)] 1 appeared 3 times [0][0] extract the label itself, 1 in this eg
    
    # or simple using numpy logic
    #def most_common_label(y):
    #    y = np.array(y)
    #    values, counts = np.unique(y, return_counts=True)    #values = array([0, 1]) counts = array([2, 3]) label 0 appeared 2 times and label 1 appeared 3 times
    #    return values[np.argmax(counts)]                     #np.argmax(counts) finds the index of the largest count in counts




    #prediction of labels for input data X using the trained random forest.
    def predict(self, X):
       # This list will store predictions from EACH tree with shape (before transpose): [n_trees][n_samples]
      all_preds = []
    
       # Going through every tree in the forest and each tree also remembers which feature columns it was trained on
      for tree, feature_indices in self.trees:
           # X[:, feature_indices] take ALL rows, but ONLY the columns:
           # Each tree was trained using a random subset of features.
           # Giving it extra or different features would be incorrect.
          preds = tree.predict(X[:, feature_indices])
    
          # preds is a list like:
          # [1, 0, 1, 1, 0] → one prediction per sample
          all_preds.append(preds)

      # all_preds looks like this:
      # [ [preds from tree 1],[preds from tree 2], [preds from tree 3], ... ] with shape = (n_trees, n_samples)
      # .T transposes the array  as we want predictions grouped by sample, not by tree.
      # After transpose:[ [tree1, tree2, tree3 votes for sample 1],[tree1, tree2, tree3 votes for sample 2], ...]
      all_preds = np.array(all_preds).T
      # For each sample, take all tree predictions and choose the most common label (majority vote). Random Forest reduces errors by voting instead of trusting one tree
      final_preditions=[]
      for sample_preds in all_preds:
          majority_label = self.most_common_label(sample_preds)
          final_preditions.append(majority_label)
      return np.array(final_preditions)

    #accuracy
    def accuracy(self, X, y):

        return np.mean(self.predict(X) == y) * 100