import numpy as np 
class Node:
    def __init__(self,left=None,right=None,feature=None,threshold=None,*,values= None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.values = values


    def leaf(self):
        return self.values is not None
    def label(self):
        return self.values[0][np.argmax(self.values[1])]




class DecisionTree:
    def __init__(self,max_depth=100, min_samples=4,n_features=None):
        self.tree = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features

    def fit(self,X,y):
        dim = X.shape[1]
        self.n_features = dim if not self.n_features else min(self.n_features,dim)
        self.root = self.make_tree(X,y)

    def make_tree(self,X,y,depth=0):

        n_dim,p_dim = X.shape
        labels = np.unique(y,return_counts=True)

        if(depth > self.max_depth or n_dim < self.min_samples or labels[0].size == 1):
            return Node(values = labels)
        
        features = np.random.choice(p_dim,self.n_features,replace=False)

        best_gain, best_feature , best_threshold = (None,) * 3

        for feature in features:
            values = np.unique(X[:,feature])

            for value in values:

                gain = self.info_gain(X[:,feature],y,value)

                if best_gain is None or gain > best_gain:
                    best_gain, best_feature , best_threshold  = gain,feature,value


        left_mask,right_mask = X[:,best_feature] < best_threshold,X[:,best_feature] >= best_threshold
        left_node = self.make_tree(X[left_mask,:],y[left_mask],depth=depth+1)
        right_node = self.make_tree(X[right_mask,:],y[right_mask],depth=depth+1)
        return Node(left=left_node,right=right_node,feature=best_feature,threshold=best_threshold)



    def info_gain(self,X_col,y,threshold):
        E = np.unique(y,return_counts=True)[1]

        E_left = np.unique(y[X_col<threshold],return_counts=True)[1]
        E_right = np.unique(y[X_col>=threshold],return_counts=True)[1]

        s_l = E_left.sum()
        s_r = E_right.sum()
        s = E.sum()

        E_prob = E/s
        E_left_prob = E_left/s_l
        E_right_prob = E_right/s_r

        E_entropy = -np.multiply(E_prob,np.log2(E_prob)).sum()
        E_left_entropy = -np.multiply(E_left_prob,np.log2(E_left_prob)).sum()
        E_right_entropy = -np.multiply(E_right_prob,np.log2(E_right_prob)).sum()

        return (E_entropy - E_left_entropy*(s_l/s) - E_right_entropy*(s_r/s))
    
    def predict(self,X):
        return np.array([self.findPath(x,self.root) for x in X])
    
    def findPath(self,x,node):
        if node.leaf():
            return node.label()
        
        if x[node.feature] < node.threshold:
            self.findPath(x,node.left)
        else: 
            self.findPath(x,node.right)
