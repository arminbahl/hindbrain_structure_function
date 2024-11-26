import copy

from hindbrain_structure_function.functional_type_prediction.classifier_prediction.class_predictor import *

if __name__ == "__main__":
    # load metrics and cells
    test = class_predictor(Path('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud'))
    test.load_cells_df(kmeans_classes=True, new_neurotransmitter=True, modalities=['pa', 'clem', 'em', 'clem_predict'],
                       neg_control=True)
    test.calculate_metrics('FINAL_CLEM_CLEMPREDICT_EM_PA')  #
    # test.calculate_published_metrics()
    test.load_cells_features('FINAL_CLEM_CLEMPREDICT_EM_PA', with_neg_control=True, drop_neurotransmitter=False)

    # throw out truncated, exits and growth cone
    test.remove_incomplete()

    # apply gregors manual morphology annotations

    test.add_new_morphology_annotation()

    test.select_features_RFE('all', 'clem', cv=False, save_features=True, estimator=LogisticRegression(random_state=0),
                             cv_method_RFE='lpo')

    np.random.seed(42)
    copy_features = copy.deepcopy(test.features_fk)
    reference_value = test.do_cv(method='lpo', clf=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                                 feature_type='fk',
                                 train_mod='all', test_mod='clem', idx=test.reduced_features_idx, plot=False)
    mean_mutated_accuracy = []
    importance = []

    K = 50
    for j in tqdm(range(copy_features.shape[1])):
        if test.reduced_features_idx[j]:
            mutated_accuracy = []
            for i in range(K):
                test.load_cells_features('FINAL_CLEM_CLEMPREDICT_EM_PA', with_neg_control=True,
                                         drop_neurotransmitter=False)
                test.remove_incomplete()
                test.add_new_morphology_annotation()
                np.random.shuffle(test.features_fk[:, j])
                a = test.do_cv(method='lpo', clf=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                               feature_type='fk',
                               train_mod='all', test_mod='clem', idx=test.reduced_features_idx, plot=False)
                mutated_accuracy.append(a)
            mean_mutated_accuracy.append(np.mean(mutated_accuracy))
            importance.append(reference_value - (1 / K) * np.sum(mutated_accuracy))
            test.features_fk = copy_features
    plt.show()
    plt.title('mean mutated accuracy')
    plt.plot(mean_mutated_accuracy, marker='x')
    plt.axhline(reference_value, c='red', alpha=0.3)
    plt.xticks(ticks=range(len(mean_mutated_accuracy)), labels=np.array(test.column_labels)[test.reduced_features_idx],
               rotation=40, ha='right')
    plt.subplots_adjust(bottom=0.45)
    plt.show()

    # permutation impportance plot
    importance_df = pd.DataFrame({'features': np.array(test.column_labels)[test.reduced_features_idx],
                                  'importance': importance}).sort_values('importance', ascending=False)

    plt.title(f'Permutation Importance\nPermutations per feature = {K}')
    plt.bar(x=range(len(importance_df['importance'])), height=importance_df['importance'])
    plt.xticks(ticks=range(len(importance_df['importance'])), labels=importance_df['features'], rotation=40, ha='right')
    plt.subplots_adjust(bottom=0.45)
    plt.show()
