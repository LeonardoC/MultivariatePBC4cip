import math


class DecisionTree():

    def __init__(self, dataset):
        self.__Model = dataset.Model
        self.__TreeRootNode = None
    
    @property
    def TreeRootNode(self):
        return self.__TreeRootNode
    
    @TreeRootNode.setter
    def TreeRootNode(self, root):
        self.__TreeRootNode = root

    @property
    def Model(self):
        return self.__Model

    @property
    def Size(self):
        if self.TreeRootNode is None:
            return 0
        else:
            return self.ComputeSizeTree(self.TreeRootNode)

    @property
    def Leaves(self):
        if self.TreeRootNode is None:
            return 0
        else:
            return self.ComputeLeaves(self.TreeRootNode)

    def ComputeSizeTree(self, decisionTree):
        if decisionTree.Children is None:
            return 1
        else:
            childrenSize = list(map(lambda child: self.ComputeSizeTree(child), decisionTree.Children))
            childrenSize.append(0)
            return max(childrenSize)+1

    def ComputeLeaves(self, decisionTree):
        if decisionTree.IsLeaf:
            return 1
        else:
            numLeaves = sum(map(lambda child: self.ComputeLeaves(child), decisionTree.Children))
            print(f"computeLeaves: {numLeaves} type {type(numLeaves)}")
            return numLeaves


class DecisionTreeNode():

    def __init__(self, data):
        self.__Data = data
        self.__Parent = None
        self.__ChildSelector = None
        self.__Children = []
    
    @property
    def Data(self):
        return self.__Data
    
    @property
    def Children(self):
        return self.__Children
    @Children.setter
    def Children(self, new_children):
        self.__Children = new_children
    
    @property
    def ChildSelector(self):
        return self.__ChildSelector
    @ChildSelector.setter
    def ChildSelector(self, new_selector):
        self.__ChildSelector = new_selector

    @property
    def IsLeaf(self):
        return (not self.Children or len(self.Children) == 0)


    def __repr__(self):
        return f"[{', '.join(map(str, self.Data))}]\n" + self.format_tree(0)
        # if not self.ChildSelector:
        #     return f"[{', '.join(map(str, self.Data))}]"
        # else:
        #     return f"[{', '.join(map(str, self.Data))}] - {self.ChildSelector}"

    def format_tree(self, ident):
        res = ""
        if not self.IsLeaf:
            for child in range(len(self.Children)):
                if self.Children[child].Data:
                    childSelector = self.ChildSelector
                    curChild = self.Children[child]
                    res += "\n"
                    res += (" ") * ((ident + 1) * 3)
                    res += "- "
                    res += childSelector.__format__(child) + f" [{', '.join(map(str, curChild.Data))}]"
                    res += curChild.format_tree(ident + 1)
        return res
