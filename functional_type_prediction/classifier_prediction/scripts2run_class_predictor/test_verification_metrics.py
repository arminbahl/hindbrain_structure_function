from holoviews.plotting.bokeh.styles import font_size

from hindbrain_structure_function.functional_type_prediction.classifier_prediction.class_predictor import *
from itertools import product
if __name__ == "__main__":
    # load metrics and cells
    with_neurotransmitter = class_predictor(Path('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud'))
    with_neurotransmitter.load_cells_df(kmeans_classes=True, new_neurotransmitter=True,
                                        modalities=['pa', 'clem', 'em', 'clem_predict'], neg_control=True)
    with_neurotransmitter.calculate_metrics('FINAL_CLEM_CLEMPREDICT_EM_PA')

    # with_neurotransmitter.calculate_published_metrics()
    with_neurotransmitter.load_cells_features('FINAL_CLEM_CLEMPREDICT_EM_PA', with_neg_control=True,
                                              drop_neurotransmitter=False)
    # throw out truncated, exits and growth cone
    with_neurotransmitter.remove_incomplete()
    # apply gregors manual morphology annotations
    with_neurotransmitter.add_new_morphology_annotation()
    # select features
    with_neurotransmitter.select_features_RFE('all', 'clem', cv=False, save_features=True,
                                              estimator=LogisticRegression(random_state=0), cv_method_RFE='lpo')
    # select classifiers for the confusion matrices
    clf_fk = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    # make confusion matrices
    with_neurotransmitter.confusion_matrices(clf_fk, method='lpo')
    # predict cells
    with_neurotransmitter.predict_cells(use_jon_priors=False, suffix='_optimize_all_predict', predict_recorded=True)
    with_neurotransmitter.plot_neurons('EM', output_filename='EM_predicted_optimize_all_predict.html')
    with_neurotransmitter.plot_neurons('clem', output_filename='CLEM_predicted_optimize_all_predict.html')
    with_neurotransmitter.calculate_verification_metrics(calculate_smat=False, with_kunst=False,
                                                         calculate4recorded=True)
    with_neurotransmitter.prediction_predict_df.loc[
        with_neurotransmitter.prediction_predict_df['imaging_modality'] == 'clem']

    clem_func_recorded = with_neurotransmitter.prediction_predict_df.query(
        'imaging_modality == "clem" and function != "to_predict" and function != "neg_control"')
    variables = ['NBLAST_general_pass', 'NBLAST_zscore_pass', 'NBLAST_anderson_ksamp_passed',
                 'NBLAST_ks_2samp_passed', 'OCSVM', 'IF', 'LOF']
    combinations1 = list(product([True, False], repeat=len(variables[:np.floor(len(variables) / 2).astype(int)])))
    combinations2 = list(product([True, False], repeat=len(variables[np.floor(len(variables) / 2).astype(int):])))

    # Print the combinations
    combinations1_names = []
    for combination in combinations1:
        temp = []
        for i, var in enumerate(variables[:np.floor(len(variables) / 2).astype(int)]):
            temp.append(var + "_" + str(combination[i]))
        combinations1_names.append(".".join(temp))

    combinations2_names = []
    for combination in combinations2:
        temp = []
        for i, var in enumerate(variables[np.floor(len(variables) / 2).astype(int):]):
            temp.append(var + "_" + str(combination[i]))
        combinations2_names.append(".".join(temp))

    verification_accuracy_matrix = pd.DataFrame(index=combinations1_names, columns=combinations2_names)
    verification_n_cells_matrix = pd.DataFrame(index=combinations1_names, columns=combinations2_names)

    for row in verification_accuracy_matrix.index:
        for column in verification_accuracy_matrix.columns:
            temp_selector = []
            for var in column.split('.'):
                if var.split('_')[-1] == 'True':
                    temp_selector.append("_".join(var.split('_')[:-1]))
            for var in row.split('.'):
                if var.split('_')[-1] == 'True':
                    temp_selector.append("_".join(var.split('_')[:-1]))

            temp = clem_func_recorded[clem_func_recorded[temp_selector].all(axis=1)]

            verification_accuracy_matrix.loc[row, column] = accuracy_score(temp['function'], temp['prediction'])
            verification_n_cells_matrix.loc[row, column] = temp.shape[0]
    import matplotlib.patches as patches

    # plot the accuracy while varying the validation metrics
    plt.figure(dpi=1000)

    plt.imshow(np.array(verification_accuracy_matrix).astype(float))
    plt.colorbar()
    plt.xticks(np.arange(len(verification_accuracy_matrix.columns)),
               [x.replace('.', "\n") for x in verification_accuracy_matrix.columns], fontsize=1)
    plt.yticks(np.arange(len(verification_accuracy_matrix.index)),
               [x.replace('.', "\n") for x in verification_accuracy_matrix.index], fontsize=1)

    max_flat_index = np.argmax(np.array(verification_accuracy_matrix).astype(float), axis=None)

    # Convert flat index to row and column index
    max_row, max_col = np.unravel_index(max_flat_index, verification_accuracy_matrix.shape)
    max_row, max_col = max_row - 0.5, max_col - 0.5
    plt.plot([max_col, max_col], [max_row, max_row + 1], 'r-')
    plt.plot([max_col, max_col + 1], [max_row + 1, max_row + 1], 'r-')
    plt.plot([max_col, max_col + 1], [max_row, max_row], 'r-')
    plt.plot([max_col + 1, max_col + 1], [max_row, max_row + 1], 'r-')
    plt.title('Visualization of Validation Metrics with Maximum Value Highlighted', fontsize='small')
    plt.show()

    # plot the number of cells while varying the validation metrics
    plt.figure(dpi=1000)

    plt.imshow(np.array(verification_n_cells_matrix).astype(float))
    plt.colorbar()
    plt.xticks(np.arange(len(verification_n_cells_matrix.columns)),
               [x.replace('.', "\n") for x in verification_n_cells_matrix.columns], fontsize=1)
    plt.yticks(np.arange(len(verification_n_cells_matrix.index)),
               [x.replace('.', "\n") for x in verification_n_cells_matrix.index], fontsize=1)

    max_flat_index = np.argmax(np.array(verification_n_cells_matrix).astype(float), axis=None)

    # Convert flat index to row and column index
    max_row, max_col = np.unravel_index(max_flat_index, verification_n_cells_matrix.shape)
    max_row, max_col = max_row - 0.5, max_col - 0.5
    plt.plot([max_col, max_col], [max_row, max_row + 1], 'r-')
    plt.plot([max_col, max_col + 1], [max_row + 1, max_row + 1], 'r-')
    plt.plot([max_col, max_col + 1], [max_row, max_row], 'r-')
    plt.plot([max_col + 1, max_col + 1], [max_row, max_row + 1], 'r-')
    plt.title('Heatmap of Verification Accuracy with Highlighted Maximum Value', fontsize='small')
    plt.show()

    plt.figure(dpi=1000)
    verification_accuracy_matrix_delta = verification_accuracy_matrix - accuracy_score(clem_func_recorded['function'],
                                                                                       clem_func_recorded['prediction'])
    plt.imshow(np.array(verification_accuracy_matrix_delta).astype(float))
    plt.colorbar()
    plt.xticks(np.arange(len(verification_accuracy_matrix_delta.columns)),
               [x.replace('.', "\n") for x in verification_accuracy_matrix_delta.columns], fontsize=1)
    plt.yticks(np.arange(len(verification_accuracy_matrix_delta.index)),
               [x.replace('.', "\n") for x in verification_accuracy_matrix_delta.index], fontsize=1)

    max_flat_index = np.argmax(np.array(verification_accuracy_matrix_delta).astype(float), axis=None)

    # Convert flat index to row and column index
    max_row, max_col = np.unravel_index(max_flat_index, verification_accuracy_matrix_delta.shape)
    max_row, max_col = max_row - 0.5, max_col - 0.5
    plt.plot([max_col, max_col], [max_row, max_row + 1], 'r-')
    plt.plot([max_col, max_col + 1], [max_row + 1, max_row + 1], 'r-')
    plt.plot([max_col, max_col + 1], [max_row, max_row], 'r-')
    plt.plot([max_col + 1, max_col + 1], [max_row, max_row + 1], 'r-')
    plt.title('Visualization of Validation Metrics with Maximum Value Highlighted\nDiference from without',
              fontsize='small')
    plt.show()
