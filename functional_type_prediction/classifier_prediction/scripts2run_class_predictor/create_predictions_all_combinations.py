from hindbrain_structure_function.functional_type_prediction.classifier_prediction.class_predictor import *
import copy

if __name__ == "__main__":
    predictor_og = class_predictor(Path(r'D:\hindbrain_structure_function\nextcloud'))
    predictor_og.load_cells_df(kmeans_classes=True, new_neurotransmitter=True, modalities=['pa', 'clem', 'em', 'clem_predict'], neg_control=True)
    for drop_growth_cone in [True, False]:
        suffix = ''
        if drop_growth_cone:
            suffix = '_drop_growth_cone'

        for drop_exits_volume in [True, False]:
            if drop_exits_volume:
                suffix += '_drop_exits_volume'
            for drop_truncated in [True, False]:
                if drop_truncated:
                    suffix += '_drop_truncated'
                for add_morphology_annotation in [True, False]:
                    if add_morphology_annotation:
                        suffix += '_add_morphology_annotation'
                    for drop_neurotransmitter in [True, False]:
                        if drop_neurotransmitter:
                            suffix += '_drop_neurotransmitter'
                        for cv_method in ['ss','lpo']:
                            suffix += f'_{cv_method}'

                            predictor_copy = copy.deepcopy(predictor_og)
                            predictor_copy.load_cells_features('FINAL_CLEM_CLEMPREDICT_EM_PA', with_neg_control=True, drop_neurotransmitter=drop_neurotransmitter)
                            predictor_copy.remove_incomplete(growth_cone=drop_growth_cone, exits_volume=drop_exits_volume, truncated=drop_truncated)
                            if add_morphology_annotation:
                                predictor_copy.add_new_morphology_annotation()

                            if drop_neurotransmitter:
                                estimator_rfe = Perceptron(random_state=0)
                            else:
                                estimator_rfe = LogisticRegression(random_state=0)

                            predictor_copy.select_features_RFE('all', 'clem', cv=False, save_features=True, estimator=estimator_rfe, cv_method_RFE=cv_method)
                            clf_fk = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

                            predictor_copy.predict_cells(use_jon_priors=True,suffix=suffix)
                            test.plot_neurons('EM', output_filename=f'EM_predicted_jon_priors{suffix}.html')
                            predictor_copy.predict_cells(use_jon_priors=False,suffix=suffix)
                            predictor_copy.plot_neurons('EM', output_filename=f'EM_predicted{suffix}.html')

