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
    cutoffs_success = []
    cutoffs_values = []
    n_used_cells = []
    for i in np.arange(0.01, 1, 0.01):
        a = test.do_cv(method='lpo', clf=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), feature_type='fk',
                       train_mod='all', test_mod='clem', idx=test.reduced_features_idx, plot=False, proba_cutoff=i)
        cutoffs_success.append(a[0])
        cutoffs_values.append(i)
        n_used_cells.append(a[1])
    plt.plot(cutoffs_values, cutoffs_success)
    plt.axvline(cutoffs_values[np.argmax(cutoffs_success)], c='red')
    plt.show()

    # Create some mock data

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('cutoff')
    ax1.set_ylabel('accuracy (%)', color=color)
    ax1.plot(cutoffs_values, cutoffs_success, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('used cells (n)', color=color)  # we already handled the x-label with ax1
    ax2.plot(cutoffs_values, n_used_cells, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
