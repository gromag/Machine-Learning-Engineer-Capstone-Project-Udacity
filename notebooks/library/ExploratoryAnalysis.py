import numpy as np
import pandas as pd

class ExploratoryAnalysis():
           
    def __init__(self, train_path=None, test_path=None, train=None, test=None):
        self.train = train if train is not None else pd.read_csv(train_path)
        self.test = test if test is not None else pd.read_csv(test_path)           

    def describe_train(self):
        return self.train.describe()

    def describe_test(self):
        return self.test.describe()
        
    
class JigsawExploratoryAnalysis(ExploratoryAnalysis):
    """
    
    """
    subgroup_columns = ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']

    identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 
                    'white', 'psychiatric_or_mental_illness']
    
    TOXIC_COLUMN = 'toxic'
        
    def __init__(self, train_path = None, test_path = None, train = None, test = None):
        super().__init__(train_path, test_path, train, test)
        
        self.train[self.identity_columns] = self.train[self.identity_columns].fillna(0)
        self._add_toxic_column()
    
    def _add_toxic_column(self):
        train_toxic =(self.train['target'].values > 0.5).astype(bool).astype(np.int)
        self.train[self.TOXIC_COLUMN] = train_toxic
        
    def calculate_stats(self):
        
        assert self.TOXIC_COLUMN in self.train.columns
        
        test_percent = len(self.test)/len(self.train)
        toxic_comment_distribution = self.train['toxic'].sum()/len(self.train['toxic'])
        
        return (test_percent, toxic_comment_distribution)
    
    def __calculate_toxic_stats_for_column(self, col, threshold = 0.0):

        is_relevant = self.train[col] > threshold
        na_count = self.train[col].isnull().sum()
        rel_records = self.train[is_relevant]
        toxic_count = rel_records['toxic'].sum()
        toxic_percent = toxic_count/len(rel_records)
        
        return (col, toxic_count, len(rel_records), toxic_percent, na_count)
    
    def calculate_toxic_stats_for_identities(self, threshold = 0.0):
        
        toxic_stats = pd.DataFrame(columns=["identity", "toxic_count", "count", "toxic_percent", "na_count"])

        for index, col in enumerate(JigsawExploratoryAnalysis.identity_columns):
            toxic_stats.loc[index] = explore.__calculate_toxic_stats_for_column(col, threshold)
            
        return toxic_stats
    
    def calculate_toxic_stats_for_subgroups(self, threshold = 0.0):
        
        toxic_subgroup_stats = pd.DataFrame(columns=["subgroup", "toxic_count", "count", "toxic_percent", "na_count"])

        for index, col in enumerate(JigsawExploratoryAnalysis.subgroup_columns):
            toxic_subgroup_stats.loc[index] = self.__calculate_toxic_stats_for_column(col, threshold)
            
        return toxic_subgroup_stats
    
    
    def __build_no_identities_set_expression(self):
        expr = None
        for col in JigsawExploratoryAnalysis.identity_columns:

            tmp = (self.train[col] == 0)

            if expr is None:
                expr = tmp
            else:        
                expr = expr & tmp
        return expr
    
    
    def __build_at_least_one_identity_set_expression(self):
        
        expr = None
        for col in JigsawExploratoryAnalysis.identity_columns:

            tmp = (self.train[col] > 0)

            if expr is None:
                expr = tmp
            else:        
                expr = expr | tmp
        return expr
    
    def calculate_no_identity_stats(self):
    
        no_identity_train = self.train[self.__build_no_identities_set_expression()]
        identity_train = self.train[self.__build_at_least_one_identity_set_expression()]
        
        return (no_identity_train, identity_train)

  