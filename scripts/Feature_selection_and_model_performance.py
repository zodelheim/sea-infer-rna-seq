import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_validate
import os
import sys
print("Modules are imported")

def log_a_table(df):
    print("taking transcripts values log begins")
    df = df.replace(0, 0.000001)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = np.log(df[numerical_cols])
    print("log_a_table is done")
    return df

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    return accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro


def filter_by_non_zero_median(df):
    print("filtering by non zero median begins")
    print(str(len(df.columns)))
    if (df.median() == 0).any():
        cols_to_drop = df.columns[df.median() == 0]
        print(str(len(cols_to_drop)) + " features will be removed, due to a zero median value")
        df = df.drop(columns=cols_to_drop)
        print("Current amount of features is " + str(len(df.columns)))
        if len(df.columns) == 0:
            print("With this filtering there are no features in the dataset")
            sys.exit()
        return df
    print("Zero median columns aren't found")
    print(str(len(df.columns)))
    return df

def merge_df(df,metadata):
    return pd.merge(df, metadata, on='Run', how='inner')

def split_the_table(merged_df, test_size=0.2, random_state=0):
    merged_df.reset_index(inplace=True)
    index_column = merged_df['Run']
    merged_df.drop('Run', axis=1, inplace=True)
    features = merged_df.drop('Descriptor', axis=1)
    labels = merged_df['Descriptor']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, index_column
        

def remove_a_chromosome(gtf_path, transcript_list, chr_to_remove):
    found_transcripts = []
    with open(gtf_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            chrom = parts[0]
            attributes = parts[8]
            if 'transcript_id' in attributes:
                transcript_id = attributes.split('transcript_id "')[1].split('";')[0]
                if transcript_id in transcript_list and chrom in chr_to_remove:
                    found_transcripts.append(transcript_id)
                    transcript_list.remove(transcript_id)

            if not transcript_list:
                print("With this filtering by chromosomes all features were removed")
                sys.exit()
                break
    print(str(len(found_transcripts)) + " transcripts will be removed because of their chromosome")
    return transcript_list




def Gradient_boosting_model(X_train,X_test,y_train,y_test,output_dir):
    X, X_val, y,  y_val = train_test_split(X_train, y_train)
    CBC = CatBoostClassifier(loss_function='MultiClass', od_pval = 0.05,) 
    CBC.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100, 
              use_best_model=True, plot=True, early_stopping_rounds=20)
    CBC.save_model("catboost_model",
           format="json",
           export_parameters=None,
           pool=None)
    print("Catboost model is saved")
    pred = CBC.predict(X_test)
    report = classification_report(y_test, pred)
    report_path = output_dir + "/gradient_boosting_classification_report.txt"

    with open(report_path, "w") as file:
        file.write(report)
    plt.figure(figsize=(10, 6))
    sorted_feature_importance = CBC.feature_importances_.argsort()[::-1]
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': CBC.feature_importances_
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    feature_importance_df.to_csv(output_dir + "/catboost_feature_importance.csv", index=False, sep = "\t")


    sns.barplot(x=CBC.feature_importances_[sorted_feature_importance[:20]],
              y=X_train.columns[sorted_feature_importance[:20]], orient='h')
    plt.tight_layout()
    sns.despine()
    plt.yticks(fontsize=6)
    plt.xlabel("CatBoost Feature Importance")

    plot_path = output_dir + "/feature_importance_plot.png"
    plt.savefig(plot_path)
    


def knn_feature_selection(df,n_features,output_dir):
    target = df["Descriptor"]
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target)
    feature_matrix = df.drop(["Descriptor"],axis = 1)
    mi_df = pd.DataFrame(columns=['Transcript', 'Mutual_Information'])
    for column in feature_matrix.columns:
        mi_score = mutual_info_classif(feature_matrix[column].values.reshape(-1, 1), target, random_state = 0)[0]
        mi_df = pd.concat([mi_df, pd.DataFrame({'Transcript': [column], 'Mutual_Information': [mi_score]})], ignore_index=True)
    mi_df_sorted = mi_df.sort_values(by='Mutual_Information', ascending = False)
    rows_to_drop = mi_df_sorted.loc[mi_df_sorted['Mutual_Information'] < 0.01].index
    mi_df_filtered = mi_df_sorted.drop(rows_to_drop)
    mi_table_path = output_dir + "/mutual_information.csv"

    mi_df_filtered = mi_df_filtered.head(n_features)
    mi_df_filtered.to_csv(mi_table_path,index = False)
    return mi_df_filtered

    


#def cross_val_n_best_transcripts_model(n, mi_df_filtered, X, y):
    #result_df = pd.DataFrame(columns=['number_of_best_transcripts', 'auc', 'f1', 'accuracy', 'n_neighbors'])

    #for i in range(1, n + 1):
        #selected_transcripts = mi_df_filtered.head(i)['Transcript'].tolist()

        #best_auc = 0
        #best_neighbors = 0
        #best_f1 = 0
        #best_accuracy = 0
        #best_recall = 0

        #for n_neighbors in [1, 3, 5, 7, 9, 11]:
            #X_selected = X[selected_transcripts]

            #knn_model = KNeighborsClassifier(n_neighbors=n_neighbors,algorithm = "brute")
            #scoring = {'roc_auc_macro': 'roc_auc', 
               #'f1_macro': make_scorer(f1_score, average='macro'),
               #'accuracy': 'accuracy'}
            #cv_results = cross_validate(knn_model, X_selected, y, cv=5, scoring=scoring)
            
            

            #auc_mean = np.mean(cv_results['test_roc_auc_macro'])
            #f1_mean = np.mean(cv_results['test_f1_macro'])
            #accuracy_mean = np.mean(cv_results['test_accuracy'])

            #if auc_mean > best_auc:
                #best_auc = auc_mean
                #best_neighbors = n_neighbors
                #best_f1 = f1_mean
                #best_accuracy = accuracy_mean

        #result_df = pd.concat([result_df, pd.DataFrame({'number_of_best_transcripts': [i], 'auc': [best_auc], 'f1': [best_f1],
                                                        #'accuracy': [best_accuracy], 'n_neighbors': [best_neighbors]})], ignore_index=True)

    #return result_df
    
    
def cross_val_n_best_transcripts_model(n, mi_df_filtered, X, y):
    result_df = pd.DataFrame(columns=['number_of_best_transcripts', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'n_neighbors'])

    for i in range(1, n + 1):
        
        selected_transcripts = mi_df_filtered.head(i)['Transcript'].tolist()

        X_selected = X[selected_transcripts]

        best_accuracy = 0
        best_precision_macro = 0
        best_recall_macro = 0
        best_f1_macro = 0
        best_neighbors = 0

        for n_neighbors in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
            knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="brute")
            scoring = {'accuracy': 'accuracy',
                       'precision_macro': 'precision_macro',
                       'recall_macro': 'recall_macro',
                       'f1_macro': 'f1_macro'}
            cv_results = cross_validate(knn_model, X_selected, y, cv=5, scoring=scoring)

            accuracy = cv_results['test_accuracy'].mean()
            precision_macro = cv_results['test_precision_macro'].mean()
            recall_macro = cv_results['test_recall_macro'].mean()
            f1_macro = cv_results['test_f1_macro'].mean()

            if f1_macro > best_f1_macro:
                best_accuracy = accuracy
                best_precision_macro = precision_macro
                best_recall_macro = recall_macro
                best_f1_macro = f1_macro
                best_neighbors = n_neighbors

        result_df = pd.concat([result_df, pd.DataFrame({'number_of_best_transcripts': [i],
                                                        'accuracy': [best_accuracy],
                                                        'precision_macro': [best_precision_macro],
                                                        'recall_macro': [best_recall_macro],
                                                        'f1_macro': [best_f1_macro],
                                                        'n_neighbors': [best_neighbors]})], ignore_index=True)
        print(str(i) + " best features knn model has been validated")

    return result_df



if __name__ == '__main__': # Script can be used as a module to take any of its functions
    print("Data reading and preparation begins")
    parser = argparse.ArgumentParser(
                                    prog = 'Feature selection for classification',
                                    description = 'This program finds best features for the classification and validate several models')

    parser.add_argument('--Dataset', type = str, help = 'Path to a Dataset with expressions in tsv format. Run column contains all samples, other columns with expressions')
    parser.add_argument('--log', type = str, help = 'Do you want to log the data? Options: True or False',default = "False")
    parser.add_argument('--Metadata', type = str, help = 'Path to metadata in tsv format. Run column contains samples, Descriptor column contains cathegorical data')
    parser.add_argument('--n_features', type = int, help = 'How many features do you want to get? Note: Less features appear in a result if the rest of them have bad performance')
    #parser.add_argument('--additional', type = str, help = '...')
    parser.add_argument('--output', type = str, help = 'Path to a folder for all results')
    parser.add_argument('--gtf', type = str, help = 'path to gtf file', default = None)
    parser.add_argument('--exclude_chr', type = str, help = 'comma separated chromosomes to exclude', default =  None)
    parser.add_argument('--non_zero_median', type = str, help = 'Do you want to delete median == 0 transcripts? Options: True or False',default = "False")
    
    print("taking all input parameters")
    args = parser.parse_args()
    Dataset_path = args.Dataset
    n_features = args.n_features
    need_a_log = args.log
    non_zero_median = args.non_zero_median
    Metadata_path = args.Metadata
    output_dir = args.output
    print(output_dir, "is a foldaer for an output")
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]
    

    print("reading dataframe and metadata")
    df = pd.read_csv(Dataset_path,sep = "\t")
    df = df.T
    df.reset_index(inplace = True)
    df = df.rename(columns={'index': 'Run'})
    df.set_index('Run', inplace=True)
    metadata_df = pd.read_csv(Metadata_path, sep = "\t")
    
    metadata_df = metadata_df[["Run","Descriptor"]] # We use only these two columns
    metadata_df["Descriptor"] = metadata_df["Descriptor"].apply(str)
    

    
    if non_zero_median == "True":
        
        df = filter_by_non_zero_median(df) # if median(expression among samples) == 0 => remove the transcript
        print("filtering by non zero median ends")
    
    if need_a_log == "True":
        df = log_a_table(df) # zeros are replaced by 0.000001
    print("filtering is done")

    
    if args.exclude_chr is not None and args.gtf is not None:
        gtf_path = args.gtf
        excluded_chromosomes = args.exclude_chr.split(',')
        chr_to_remove = tuple(excluded_chromosomes) if len(excluded_chromosomes) > 1 else (args.exclude_chr,)
        columns_to_keep = remove_a_chromosome(gtf_path, list(df.columns), chr_to_remove)
        df = df[columns_to_keep]
        print("Transcripts from chromosomes are reduced")
    
    merged_df = merge_df(df,metadata_df) # df and metadata inner-join. Only common raws exist in merged_df
    print("df and metadata were merged")
    X_train, X_test, y_train, y_test, index_column = split_the_table(merged_df)
    print("Data is prepared for Gradient-boosting")
    Gradient_boosting_model(X_train,X_test,y_train,y_test,output_dir)
    print("GB is done")
    mi_df_filtered = knn_feature_selection(merged_df, n_features, output_dir)
    print("knn feature selection is done")
    X = merged_df.drop(columns = ["Descriptor"])
    y = merged_df["Descriptor"]
    print("data is prepared for knn validation")
    result_df = cross_val_n_best_transcripts_model(len(mi_df_filtered), mi_df_filtered, X, y)
    print("knn_validation is done")
    #result_df = cross_val_n_best_transcripts_model(15, mi_df_filtered, X, y)
    result_path = output_dir + "/knn_model_validation_result.csv"
    result_df.to_csv(result_path,index = False, sep = "\t")
