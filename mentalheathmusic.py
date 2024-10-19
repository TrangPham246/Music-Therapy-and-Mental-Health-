# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fancyimpute import IterativeImputer
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer import calculate_kmo
from sklearn import decomposition 
from factor_analyzer import FactorAnalyzer
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Read csv files
df = pd.read_csv('/Users/phamtrang/Documents/Music-Mental health/mxmh.csv', delimiter=',')

# Data exploration
## Check some first rows of dataset
df.head()
## Assign a unique ID to each row, starting from 1
df['ID'] = pd.Series(range(1, len(df) + 1))
## Check null-values, types and descriptive statistic of dataset
df.info()
df.describe()
## Visualize numerical data
sns.histplot(data=df, x='Age') 
sns.histplot(data=df, x='Hours per day') 
sns.histplot(data=df, x='Anxiety') 
sns.histplot(data=df, x='Depression')
sns.histplot(data=df, x='Insomnia')
sns.histplot(data=df, x='OCD')

## Visualize categorical data
sns.histplot(data=df, x= 'Primary streaming service')
plt.xticks(rotation=80)
sns.histplot(data=df, x = 'Fav genre')
plt.xticks(rotation=90)

# Treat outliers
## Find outliers with boxplots
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
sns.boxplot(data=df, x='Age', ax=ax[0, 0]) 
sns.boxplot(data=df, x='Hours per day', ax=ax[0, 1])
sns.boxplot(data=df, x='Anxiety', ax=ax[0, 2]) 
sns.boxplot(data=df, x='Depression', ax=ax[1, 0])
sns.boxplot(data=df, x='Insomnia', ax=ax[1, 1])
sns.boxplot(data=df, x='OCD', ax=ax[1, 2])
sns.boxplot(data=df, x='BPM', ax=ax[2, 0])
ax[2, 1].axis('off')
ax[2, 2].axis('off')
plt.tight_layout()
plt.show()

## Treating the outlier in BPM: replace BPM > 250 or <0 with NA since there is no BPM that can be larger than 250 and lower than 20
df.loc[(df['BPM'] > 250) | (df['BPM'] < 20), 'BPM'] = np.nan

# Treat missing values
## Check missing values
df.isna().sum().plot(kind='bar')

## Drop column 'Primary streaming service' since it is not important
df_new = df.drop(columns = 'Primary streaming service')

## Drop missing values in Music effect
df_new = df_new.dropna(subset='Music effect')

## Convert data for imputation

### Convert ordinal columns into numeric columns
frequency_mapping = {
    'Never': 1,
    'Rarely': 2,
    'Sometimes': 3,
    'Very frequently': 4
}
ordinal_columns = [
    'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]', 'Frequency [Folk]',
    'Frequency [Gospel]', 'Frequency [Hip hop]', 'Frequency [Jazz]', 'Frequency [K pop]',
    'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]',
    'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]'
]
df_new[ordinal_columns]= df_new[ordinal_columns].replace(frequency_mapping)

###Convert binary columns into dummy columns
binary_columns = ['While working', 'Instrumentalist', 'Composer','Exploratory', 'Foreign languages','Music effects']
df_dummies = pd.get_dummies(df_new, columns=binary_columns)
object_columns = df_dummies.select_dtypes(include=['object']).columns
df_dummies_no_objects = df_dummies.drop(columns=object_columns)
df_dummies_no_objects.columns
drop_columns = ['While working_No','Instrumentalist_No','Composer_No','Exploratory_No','Foreign languages_No','Music effects_No effect']
df_dummies_imputed = df_dummies_no_objects.drop(columns=drop_columns)
df_dummies_imputed.columns

## Missing values imputation
### MICE imputation
df_imputed_MICE = df_dummies_imputed.copy(deep='True')
scaler = MinMaxScaler()
df_imputed_MICE = pd.DataFrame(scaler.fit_transform(df_imputed_MICE), columns = df_imputed_MICE.columns)
MICE_imputer = IterativeImputer()
df_imputed_MICE.iloc[:,:] = MICE_imputer.fit_transform(df_imputed_MICE)
df_imputed_MICE = pd.DataFrame(scaler.inverse_transform(df_imputed_MICE), columns=df_imputed_MICE.columns)

## KNN imputation
df_imputed_KNN = df_dummies_imputed.copy(deep='True')
df_imputed_KNN = pd.DataFrame(scaler.fit_transform(df_imputed_KNN), columns = df_imputed_KNN.columns)
from sklearn.impute import KNNImputer
KNN_imputer = KNNImputer(n_neighbors=5)
df_imputed_KNN.iloc[:,:] = KNN_imputer.fit_transform(df_imputed_KNN)
df_imputed_KNN = pd.DataFrame(scaler.inverse_transform(df_imputed_KNN), columns=df_imputed_KNN.columns)

### Check if the imputation was robust
original_stats = df_dummies_no_objects['BPM'].describe()
imputed_MICE_stats = df_imputed_MICE['BPM'].describe()
imputed_KNN_stats = df_imputed_KNN['BPM'].describe()
print(f'Original stats:\n{original_stats}')
print(f'MICE stats:\n{imputed_MICE_stats}')
print(f'KNN stats:\n{imputed_KNN_stats}')
sns.kdeplot(df_dummies_no_objects['BPM'], label='Original Data', fill=False)
sns.kdeplot(df_imputed_MICE['BPM'], label='MICE Imputed Data', fill=False)
sns.kdeplot(df_imputed_KNN['BPM'], label='KNN Imputed Data', fill=False)
plt.title('Density Plot of BPM')
plt.legend()
plt.show() 

### Move on with KNN

# Create final dataset for analyis
## Create one Music effects column
def get_music_effect(row):
    if row['Music effects_Improve'] == True:
        return 'Improve'
    elif row['Music effects_Worsen'] == True:
        return 'Worsen'
    else:
        return 'No Effect'
df_imputed_KNN['Music effects'] = df_imputed_KNN.apply(get_music_effect, axis=1)

## Convert Age into integer
df_imputed_KNN['Age'] = df_imputed_KNN['Age'].astype('int')

## Round float columns into 1 decimal
round(df_imputed_KNN[['BPM','Anxiety','Depression','Insomnia']],1)

##Create Generation column
def get_generation(row):
    if 12 <= row['Age'] <= 27:
        return 'GenZ'
    elif 28 <= row['Age'] <= 43:
        return 'GenY'
    elif 44 <= row['Age'] <= 59:
        return 'GenX'
    elif 60 <= row['Age'] <= 78:
        return 'Baby Boomers'
    else:
        return 'Post Wars'
df_imputed_KNN['Generation'] = df_imputed_KNN.apply(get_generation, axis = 1)

## Drop unnecessary columns
df_imputed_KNN = df_imputed_KNN.drop(columns = ['Music effects_Improve','Music effects_Worsen'])

## Assign new name for the final datasett
df_final = df_imputed_KNN

#df_final.to_csv('/Users/phamtrang/Documents/Music-Mental health/final_df1.csv',index=False)

## Put column Fav genre to df_final
df_final.columns
df_final = pd.merge(df_final, df[['ID', 'Fav genre']], on='ID', how='left')


# Correlation Analysis
## Calculate the correlation matrix
df_final_no_string = df_final.drop(columns= ['Fav genre','Generation','Music effects'])
corr_matrix = df_final_no_string.corr()

## Create a heatmap for the correlation matrix
plt.figure(figsize=(18, 16))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()

## VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = df_final_no_string.columns

## calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(df_final_no_string.values, i)
                          for i in range(len(df_final_no_string.columns))]

print(vif_data)

# VIF and Correlation heatmap shows that some columns are highly correlated -> PCA
#PCA
## Drop irrelevant columns
df_final_frequency = df_final.drop(columns= ['Fav genre','Generation','Music effects','Hours per day','BPM',
                                             'Anxiety','Depression','Insomnia','OCD','While working_Yes',
                                             'Instrumentalist_Yes','Composer_Yes','Exploratory_Yes','Foreign languages_Yes','Age'])

## Scale the dataset
features = df_final_frequency.columns
x = df_final_frequency.loc[:, features].values
scaled_data = StandardScaler().fit_transform(x) # normalizing the features
np.mean(scaled_data),np.std(scaled_data)

## KMO test to see if it is suitable for PCA
kmo_all, kmo_model = calculate_kmo(scaled_data)
print("KMO measure of sampling adequacy for each variable:")
for i, kmo in enumerate(kmo_all):
    print(f"{df_final_frequency.columns[i]}: {kmo:.3f}")
print(f"\nOverall KMO measure of sampling adequacy: {kmo_model}")

pca = PCA() # create a PCA object
pca.fit(scaled_data) # do the math
pca_data = pca.transform(scaled_data)

# Create a scree plot
plt.figure(figsize=(8,5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio')
plt.show()

# Assuming pca is already fitted PCA object from sklearn
eigenvalues = pca.explained_variance_

# Applying Kaiser Criterion
kaiser_criterion = eigenvalues > 1

# Get the number of components that satisfy the Kaiser Criterion
n_components_kaiser = sum(kaiser_criterion)

# Print the components that satisfy the criterion
print(f"Number of components with eigenvalue > 1: {n_components_kaiser}")
important_components = np.arange(1, n_components_kaiser + 1)
print(f"Important components according to Kaiser Criterion: {important_components}")
# Optionally, if you want to see which specific components satisfy the criterion
for i, (ev, satisfy) in enumerate(zip(eigenvalues, kaiser_criterion), start=1):
    if satisfy:
        print(f"Component {i} with eigenvalue: {ev}")
# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Create a plot for cumulative explained variance
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.axhline(y=0.7, color='r', linestyle='-')  # 90% variance line
plt.text(0.5, 0.85, '70% cut-off threshold', color = 'red', fontsize=16)
plt.show()

#keep 5 components

# Fit PCA with 5 components
pca = decomposition.PCA(n_components=5)
pca.fit(scaled_data)

# Get the PCA loadings (components)
loadings = pca.components_.T

# Create a DataFrame with the loadings for each principal component
loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=df_final_frequency.columns)

# Display the loadings DataFrame
loadings_df


# fit factor analyzer with principal components and varimax rotation
fa = FactorAnalyzer(rotation="varimax", n_factors=5, method='principal')
principal_components = fa.fit(scaled_data)

# get the rotated factor pattern
loadings1 = fa.loadings_
loadings_df1 = pd.DataFrame(loadings1,  columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=df_final_frequency.columns)
rotated_factor_pattern = loadings_df1[round(abs(loadings_df1),1) >= 0.5].dropna(how='all')

rotated_factor_pattern = rotated_factor_pattern.fillna(0)
# Ensure the columns used in scaled_data match the factor loadings
matching_columns = rotated_factor_pattern.index  # Features used in factor analysis

# Filter scaled_data to include only the matching columns
scaled_data_filtered = pd.DataFrame(scaled_data, columns=df_final_frequency.columns)[matching_columns]

# Convert the filtered data to a NumPy array
scaled_data_filtered_array = scaled_data_filtered.values

# Calculate the factor scores by multiplying filtered scaled data with factor loadings
factor_loadings = rotated_factor_pattern.values
factor_scores = np.dot(scaled_data_filtered_array, factor_loadings)

# Create a DataFrame for the factor scores
factor_scores_df = pd.DataFrame(factor_scores, columns=rotated_factor_pattern.columns)

# Display the factor scores DataFrame
print(factor_scores_df)
factor_scores_df.rename(columns={'PC1': 'Urban Vibes', 'PC2': 'Rock & Metal',
                                 'PC3': 'Jazz & Classical','PC4': 'Modern Music','PC5': 'Echoes of Traditions'}, inplace=True)

df_final1 = df_final.drop(columns = features)
df_final1 = pd.merge(df_final1, factor_scores_df, left_index=True, right_index=True)
df_final1.columns

df_final1 = pd.get_dummies(df_final1, columns=['Fav genre'])
me_mapping = {'Worsen':0,
              'No Effect': 1,
              'Improve':2}
df_final1['Music effects'] = df_final1['Music effects'].replace(me_mapping)
df_final1[['Fav genre_Classical', 'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
          'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz', 'Fav genre_K pop', 
          'Fav genre_Latin', 'Fav genre_Lofi', 'Fav genre_Metal', 'Fav genre_Pop', 
          'Fav genre_R&B', 'Fav genre_Rap', 'Fav genre_Rock', 'Fav genre_Video game music',
          'While working_Yes', 'Instrumentalist_Yes', 'Composer_Yes', 'Exploratory_Yes', 
          'Foreign languages_Yes']] = df_final1[['Fav genre_Classical', 'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
          'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz', 'Fav genre_K pop', 
          'Fav genre_Latin', 'Fav genre_Lofi', 'Fav genre_Metal', 'Fav genre_Pop', 
          'Fav genre_R&B', 'Fav genre_Rap', 'Fav genre_Rock', 'Fav genre_Video game music',
          'While working_Yes', 'Instrumentalist_Yes', 'Composer_Yes', 'Exploratory_Yes', 
          'Foreign languages_Yes']].astype(int)

df_final1.columns
df_final1.dtypes

df_final1['BPM_Fav_genre_Jazz'] = df_final1['BPM'] * df_final1['Fav genre_Jazz']
df_final1['BPM_Fav_genre_Latin'] = df_final1['BPM'] * df_final1['Fav genre_Latin']
df_final1['BPM_Fav_genre_Rap'] = df_final1['BPM'] * df_final1['Fav genre_Rap']
df_final1['BPM_Fav_genre_Country'] = df_final1['BPM'] * df_final1['Fav genre_Country']
df_final1['BPM_Fav_genre_Video'] = df_final1['BPM'] * df_final1['Fav genre_Video game music']
df_final1['BPM_Fav_genre_Classical'] = df_final1['BPM'] * df_final1['Fav genre_Classical']
df_final1['BPM_Fav_genre_Hiphop'] = df_final1['BPM'] * df_final1['Fav genre_Hip hop']
df_final1['BPM_Fav_genre_Folk'] = df_final1['BPM'] * df_final1['Fav genre_Folk']
df_final1['BPM_Fav_genre_R&B'] = df_final1['BPM'] * df_final1['Fav genre_R&B']

df_final1.columns


from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Prepare your data
X = df_final1[['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression', 'Insomnia',
       'OCD', 'While working_Yes', 'Instrumentalist_Yes', 'Composer_Yes',
       'Exploratory_Yes', 'Foreign languages_Yes', 'Urban Vibes', 'Rock & Metal', 'Jazz & Classical',
       'Modern Music', 'Echoes of Traditions', 'Fav genre_Classical',
       'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
       'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz',
       'Fav genre_K pop', 'Fav genre_Latin', 'Fav genre_Lofi',
       'Fav genre_Metal', 'Fav genre_Pop', 'Fav genre_R&B', 'Fav genre_Rap',
       'Fav genre_Rock', 'Fav genre_Video game music', 'BPM_Fav_genre_Jazz',
       'BPM_Fav_genre_Latin', 'BPM_Fav_genre_Rap', 'BPM_Fav_genre_Country',
       'BPM_Fav_genre_Video', 'BPM_Fav_genre_Classical',
       'BPM_Fav_genre_Hiphop', 'BPM_Fav_genre_Folk', 'BPM_Fav_genre_R&B']]

y = df_final1['Music effects']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
log_model = LogisticRegression()

# Recursive Feature Elimination
selector = RFE(log_model)  # Adjust the number of features to select
selector = selector.fit(X_train, y_train)

# Get the selected features
selected_features = X.columns[selector.support_]
print("Selected Features:", selected_features)

# Train the model with selected features
log_model.fit(X_train[selected_features], y_train)
y_pred = log_model.predict(X_test[selected_features])

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy with selected features:", accuracy)

mod_log = OrderedModel(df_final1['Music effects'],
                        df_final1[['Hours per day', 'Depression', 'While working_Yes',
       'Instrumentalist_Yes', 'Composer_Yes', 'Exploratory_Yes',
       'Rock & Metal', 'Jazz & Classical', 'Echoes of Traditions',
       'Fav genre_Classical', 'Fav genre_EDM', 'Fav genre_Gospel',
       'Fav genre_Hip hop', 'Fav genre_K pop', 'Fav genre_Lofi',
       'Fav genre_Metal', 'Fav genre_Pop', 'Fav genre_Rap', 'Fav genre_Rock',
       'BPM_Fav_genre_Hiphop', 'BPM_Fav_genre_R&B']],
                        distr='logit')
res_log = mod_log.fit(method='bfgs')
print(res_log.summary())
odds_ratios = np.exp(res_log.params)
print("\nOdds Ratios:\n", odds_ratios)