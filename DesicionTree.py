import numpy as np 



class Node:
    def __init__(self,split_feature=None,threshold=None,left_node=None,right_node=None,*,value=None):     
        self.s_feature = split_feature
        self.s_value = threshold
        self.value = value 
        self.label = self.label_value()

        self.left = left_node
        self.right = right_node


    def leaf_node(self):
        return self.value is not None
    
    def label_value(self):
        values = np.unique(self.value,return_counts=True)
        return values[0][np.argmax(values[1])]
        

        


class DesicionTree:
    def __init__(self,min_samples_split=2,max_depth = 100,n_features_to_use = None):
        self.root = None

        self.mss = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features_to_use
    
    def fit(self,X,y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self.makeTree(X,y)


    def makeTree(self,X,y,depth=0):
        n_samples, n_fs = X.shape
        labels = np.unique(y,return_counts=True)
 
        # Stop or not
        if(depth > self.max_depth or labels[0].size == 1 or n_samples < self.mss): 
            return Node(value=labels)


        # find best split
        fs_indexs = np.random.choice(n_fs, self.n_features, replace=False)
        best_gain, best_feature_idx, best_threshold = (None,) * 3

        for fs_index in fs_indexs:
            split_values= np.unique(X[:,fs_index])
            for split_value in split_values:

                gain = self.info_gain(y,X[:,fs_index],split_value)
                
                if best_gain is None or gain > best_gain:
                    best_gain,best_feature_idx,best_threshold = gain,fs_index,split_value
        
        left_mask = X[:,best_feature_idx] < best_threshold
        right_mask = X[:,best_feature_idx] >= best_threshold
        Left_Node = self.makeTree(X[left_mask,:],y[left_mask],depth=depth+1)
        Right_Node = self.makeTree(X[right_mask,:],y[right_mask],depth=depth+1)
        
        return Node(left_node=Left_Node,right_node=Right_Node,split_feature=best_feature_idx,threshold=best_threshold)



    def info_gain(self,y,col,threshlod):

        _ ,y_counts = np.unique(y,return_counts=True)
        y_left, y_right = y[col < threshlod],y[col >= threshlod]
        _ ,y_left_counts = np.unique(y_left,return_counts=True)
        _ ,y_right_counts = np.unique(y_right,return_counts=True)

        e = y_counts/y_counts.sum()
        e_left = y_left_counts/y_left_counts.sum()
        e_right = y_right_counts/y_right_counts.sum()

        E = -(np.multiply(e,np.log2(e))).sum()
        E_left = -(np.multiply(e_left,np.log2(e_left))).sum()
        E_right = -(np.multiply(e_right,np.log2(e_right))).sum()
    
        return (E - E_left*(y_left_counts.sum()/y_counts.sum()) - E_right*(y_right_counts.sum()/y_counts.sum()))
        
        

    def predict(self, X):
        return np.array([self.findPath(x, self.root) for x in X])

    def findPath(self, x, node):
        if node.leaf_node():
            return node.label

        if x[node.s_feature] <= node.s_value:
            return self.findPath(x, node.left)
        return self.findPath(x, node.right)
