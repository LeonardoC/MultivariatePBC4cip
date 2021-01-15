from copy import copy
from core.Item import SubsetRelation
from core.FilteredCollection import FilteredCollection
from core.DecisionTreeBuilder import SelectorContext
from core.Item import CutPointBasedBuilder, MultipleValuesBasedBuilder, ValueAndComplementBasedBuilder, MultivariateCutPointBasedBuilder
from core.FeatureSelectors import CutPointSelector, MultipleValuesSelector, ValueAndComplementSelector, MultivariateCutPointSelector
from itertools import chain
from collections import OrderedDict


class EmergingPattern(object):
    def __init__(self, dataset, items=None):
        self.Dataset = dataset
        self.Model = self.Dataset.Model
        self.Items = None
        if not items:
            self.Items = list()
        else:
            self.Items = items
        self.Counts = []
        self.Supports = []

    def IsMatch(self, instance):
        #print(f"selfItemsEmerging: {self.Items}")
        for item in self.Items:
            #print(f"item!: {item}")
            if not item.IsMatch(instance):
                #print(f"item: {item} did not match")
                return False
        return True

    def UpdateCountsAndSupport(self, instances):
        matchesCount = [0]*len(self.Dataset.Class[1])

        for instance in instances:
            if(self.IsMatch(instance)):
                matchesCount[instance[self.Dataset.GetClassIdx()]] += 1
        self.Counts = matchesCount
        self.Supports = self.CalculateSupports(matchesCount)

    def CalculateSupports(self, data, classFeatureParam=None):
        #Never seen when it enters here
        #print("Supports are being calculated")
        if classFeatureParam == None:
            classInfo = self.Dataset.ClassInformation
            result = copy(data)
            for i in range(len(result)):
                if classInfo.Distribution[i] != 0:
                    result[i] /= classInfo.Distribution[i]
                else:
                    result[i] = 0
            return result
        else:
            classFeature = self.Dataset.ClassInformation.Feature
            featureInformation = self.Dataset.ClassInformation.Distribution
            #print(f"featureInformation: {featureInformation}")
            result = copy(data)
            for i in range(len(result)):
                if featureInformation[i] != 0:
                    result[i] /= featureInformation[i]
                else:
                    result[i] = 0
            return result
                

    def Clone(self):

        result = EmergingPattern(self.Dataset, self.Items)
        result.Counts = copy(self.Counts)
        result.Supports = copy(self.Supports)
        return result

    def __repr__(self):
        return self.BaseRepresentation() + " " + self.SupportInfo()

    def BaseRepresentation(self):
        return ' AND '.join(map(lambda item: item.__repr__(), self.Items))

    def SupportInfo(self):
        return ' '.join(map(lambda count, support: f"{str(count)} [{str(round(support,2))}]", self.Counts, self.Supports))

    def ToDictionary(self):
        dictOfPatterns = {"Pattern": self.BaseRepresentation()}

        dictOfClasses = {self.Dataset.Class[1][i]+" Count": self.Counts[i]
                         for i in range(0, len(self.Dataset.Class[1]))}

        dictOfClasses.update({self.Dataset.Class[1][i]+" Support": self.Supports[i]
                              for i in range(0, len(self.Dataset.Class[1]))})

        for key in sorted(dictOfClasses.keys()):
            dictOfPatterns.update({key: dictOfClasses[key]})

        return dictOfPatterns


class EmergingPatternCreator(object):
    def __init__(self, dataset):
        #print("Init EmergingPatternCreator")
        self.Dataset = dataset
        self.__builderForType = {
            CutPointSelector: CutPointBasedBuilder,
            MultipleValuesSelector: MultipleValuesBasedBuilder,
            ValueAndComplementSelector: ValueAndComplementBasedBuilder,
            MultivariateCutPointSelector: MultivariateCutPointBasedBuilder
        }

    def Create(self, contexts):
        pattern = EmergingPattern(self.Dataset)
        for context in contexts:
            childSelector = context.Selector
            builder = self.__builderForType[context.Selector.__class__]()
            item = builder.GetItem(childSelector, context.Index)
            pattern.Items.append(item)
        return pattern

    def ExtractPatterns(self, treeClassifier, patternFound):
        #print("\nextractt Patternss")
        #print(f"In ExtractPatterns ClassFeature: {self.Dataset.Class}")
        #print(f"patternFound: {patternFound}")
        context = list()
        self.DoExtractPatterns(
            treeClassifier.DecisionTree.TreeRootNode, context, patternFound)

    def DoExtractPatterns(self, node, contexts, patternFound):
        #print("Do Extract Patterns")
        #print(f"contexts: {contexts}")
        #print(type(node))
        if node.IsLeaf:
            newPattern = self.Create(contexts)
            newPattern.Counts = node.Data
            newPattern.Supports = newPattern.CalculateSupports(node.Data)
            if patternFound is not None:
                #print("About to invoke in DoExtractPatterns")
                patternFound(newPattern)
        else:
            for index in range(len(node.Children)):
                selectorContext = SelectorContext()
                selectorContext.Index = index
                selectorContext.Selector = node.ChildSelector
                context = selectorContext

                contexts.append(context)
                self.DoExtractPatterns(node.Children[index], contexts, patternFound)
                contexts.remove(context)


class EmergingPatternComparer(object):
    def __init__(self, itemComparer):
        #print(f"EmergingPatternCreatore: {itemComparer}")
        self.Comparer = itemComparer

    def Compare(self, leftPattern, rightPattern):
        #print(f"Ever enter Here?")
        directSubset = self.IsSubset(leftPattern, rightPattern)
        inverseSubset = self.IsSubset(rightPattern, leftPattern)
        if (directSubset and inverseSubset):
            return SubsetRelation.Equal
        elif (directSubset):
            return SubsetRelation.Subset
        elif (inverseSubset):
            return SubsetRelation.Superset
        else:
            return SubsetRelation.Unrelated

    def Compare_test(self, leftPattern, rightPattern):
        print(f"Ever enter Here?")
        directSubset = self.IsSubset_test(leftPattern, rightPattern)
        print("inverse")
        inverseSubset = self.IsSubset_test(rightPattern, leftPattern)
        
        print(f"direct: {directSubset} inverse: {inverseSubset}")
        if (directSubset and inverseSubset):
            return SubsetRelation.Equal
        elif (directSubset):
            return SubsetRelation.Subset
        elif (inverseSubset):
            return SubsetRelation.Superset
        else:
            return SubsetRelation.Unrelated

    def IsSubset(self, pat1, pat2):
        def f(x, y):
            relation = self.Comparer.Compare(self, x, y)
            return relation == SubsetRelation.Equal or relation == SubsetRelation.Subset

        for x in pat2.Items:
            all_bool = False
            for y in pat1.Items:
                if f(y, x):
                    #print(f"a 2 or 3 was returned")
                    all_bool = True
                    break
                #all_bool = False
                #print(f"when 2 or 3, avoid this")
            if not all_bool:
                #print(f"returned False")
                return False
        #print(f"returned True")
        return True

        #for x in pat2.Items:
        #    all_bool = True
        #    for y in pat1.Items:
        #        if f(y, x):
        #            break
        #        all_bool = False
        #    if not all_bool:
        #        return False
        #return True

        allComparisons = [[f(x, y) for x in pat1.Items]for y in pat2.Items]
        #print(f"AllComparisons: {allComparisons}")
        result = all(any(x) for x in allComparisons)
        #print(f"resultCom: {result}")
        return result 

    def IsSubset_test(self, pat1, pat2):
        print(f"pat1: {pat1}, pat2: {pat2}")
        def f(x, y):
            relation = self.Comparer.Compare(self, x, y)
            print(f"x: {x} y: {y} relation_test: {relation} bool:{relation == SubsetRelation.Equal or relation == SubsetRelation.Subset}")
            return relation == SubsetRelation.Equal or relation == SubsetRelation.Subset

        for x in pat2.Items:
            all_bool = False
            for y in pat1.Items:
                if f(y, x):
                    print(f"a 2 or 3 was returned")
                    all_bool = True
                    break
                #all_bool = False
                #print(f"when 2 or 3, avoid this")
            if not all_bool:
                print(f"returned False")
                return False
        print(f"returned True")
        return True  

        #allComparisons = [[f(x, y) for x in pat1.Items]for y in pat2.Items]
        #print(f"AllComparisons: {allComparisons}")
        #result = all(any(x) for x in allComparisons)
        #print(f"resultCom: {result}")
        # result = all(any(f(x, y) for y in pat2.Items)
        #              for x in pat1.Items)
        #return result


class EmergingPatternSimplifier(object):
    def __init__(self, itemComparer):
        self.__comparer = itemComparer
        self.__collection = FilteredCollection(
            self.__comparer, SubsetRelation.Subset)

    def Simplify(self, pattern):
        resultPattern = EmergingPattern(pattern.Dataset)
        resultPattern.Counts = copy(pattern.Counts)
        resultPattern.Supports = copy(pattern.Supports)
        self.__collection.SetResultCollection(resultPattern.Items)
        self.__collection.AddRange(pattern.Items)
        return resultPattern