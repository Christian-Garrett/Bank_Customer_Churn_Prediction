from pathlib import Path
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))


def get_descriptive_stats(self):

    print(self.data.info())
    print('\n', self.data.shape)
    print('\n', self.data.head(10).T)
    print('\n', self.data.describe()) # Describe all numerical columns
    print('\n', self.data.describe(include = ['O'])) # Describe all non-numerical/categorical columns
    print('\nThere are {} records and {} unique customers' .format(self.data.shape[0], 
                                                                   self.data.CustomerId.nunique())) ## Checking number of unique customers in the dataset

    surname_stats = \
        self.data.groupby(['Surname']).agg({'RowNumber':'count', 
                                            'Exited':'mean'}).reset_index().sort_values(by='RowNumber', 
                                                                                        ascending=False)
    print('\n', surname_stats.head())
    print('\n', self.data.Geography.value_counts(normalize=True))


def univariate_visualizations(self):
        
    sns.set_theme(style="whitegrid")

    ## CreditScore
    sns.boxplot(y=self.data.CreditScore)
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/credit_score_boxplot.png"))
    plt.clf()

    ## Age
    sns.boxplot(y=self.data.Age)
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/age_boxplot.png"))
    plt.clf()

    ## Tenure
    sns.violinplot(y=self.data.Tenure)
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/tenure_violinplot.png"))
    plt.clf()

    ## Balance
    sns.violinplot(y=self.data.Balance)
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/balance_violinplot.png"))
    plt.clf()

    ## NumOfProducts
    sns.set_theme(style='ticks')
    sns.displot(self.data.NumOfProducts, kind='hist')
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/numProducts_distplot.png"))
    plt.clf()

    ## EstimatedSalary
    sns.kdeplot(self.data.EstimatedSalary)
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/estimatedSalary_kdeplot.png"))
    plt.clf()
 

def bivariate_visualizations(self):
    
    ## Check linear correlation (rho) between individual features and the target variable
    corr = self.data.corr()
    print(corr)

    ## Heatmap
    sns.heatmap(corr, cmap='coolwarm')
    plt.savefig(os.path.join(self.output_path, "Bivariate_Analysis/train_corr_heatmap.png"))
    plt.clf()   

    ## Churn vs Age Boxplot  
    sns.boxplot(x="Exited", y="Age", data=self.data, palette="Set3")
    plt.savefig(os.path.join(self.output_path, "Bivariate_Analysis/age_churn_boxplot.png"))
    plt.clf()

    ## Churn vs Balance Violinplot
    sns.violinplot(x="Exited", y="Balance", data=self.data, palette="Set3")
    plt.savefig(os.path.join(self.output_path,
                             "Bivariate_Analysis/balance_churn_violinplot.png"))
    plt.clf()

    # Check association of categorical features with target variable
    for col in ['Gender', 'IsActiveMember', 'country_Germany', 'country_France', 'NumOfProducts']:
        print(f"Value Counts (training): {self.train_data.groupby([col]).Exited.mean()}")
