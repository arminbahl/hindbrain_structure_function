from holoviews.plotting.bokeh.styles import font_size

from hindbrain_structure_function.functional_type_prediction.classifier_prediction.class_predictor import *
from itertools import product
from sklearn.metrics import f1_score

if __name__ == "__main__":
    # load metrics and cells
    with_neurotransmitter = class_predictor(Path('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud'))
    with_neurotransmitter.load_cells_df(kmeans_classes=True, new_neurotransmitter=True,
                                        modalities=['pa', 'clem241211', 'em', 'clem_predict241211'], neg_control=True,
                                        input_em=True)

    with_neurotransmitter.calculate_published_metrics()
