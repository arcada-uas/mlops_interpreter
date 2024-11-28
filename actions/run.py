from common import misc, types
from sklearn.pipeline import Pipeline

from mappings.data_retrieval import repository as retrieval_options
from mappings.features import repository as feature_options
from mappings.models import repository as model_options
from mappings.scalers import repository as scaler_options
from mappings.segmentation import repository as segmentation_options

from components.segmentation.generate_labels import generate_segment_labels

##############################################################################################################
##############################################################################################################

def run():
    try:
        raw_config: dict = misc.load_yaml('pipeline.yaml')
        config = types.config_schema(**raw_config)

        # HIDE ERROR TRACES WHEN REQUESTED
        if config.experiment.hide_traces:
            import sys
            sys.tracebacklimit = 0

        pipeline_components = []

    ########################################################################################
    ### LOAD & SEGMENT DATASET

        retrieval_method = retrieval_options.get(config.dataset.method)
        dataset = retrieval_method(config.dataset.params)
        # dataset = retrieval_method(config.dataset.params, unittest_limit=200)

        segmentation_method = segmentation_options.get(config.training.segmentation.method)
        dataset = segmentation_method(config.training.segmentation.params, dataset)

    ########################################################################################
    ### ADD FIRST LAYER OF FEATURES

        # HIDDEN -- CONVERT INPUT TO DATAFRAME
        feature_instance = feature_options.create('to_dataframe')
        pipeline_components.append(('hidden_to_df', feature_instance))

        # NORMAL -- APPLY EACH CUSTOM FEATURE
        for nth, feature in enumerate(config.features):
            feature_instance = feature_options.create(feature.name, feature.params)
            pipeline_components.append((f'{nth}_{feature.name}', feature_instance))

        # HIDDEN -- DROP ROWS WITH NANS
        feature_instance = feature_options.create('drop_nan_rows')
        pipeline_components.append(('hidden_drop_nans', feature_instance))

    ########################################################################################
    ### GENERATE LABELS WITH THE CURRENT SET OF FEATURES

        labels: dict[str, list] = generate_segment_labels(
            dataset, 
            pipeline_components, 
            config.training.label_column
        )

    ########################################################################################
    ### ADD SECOND LAYER OF FEATURES

        # HIDDEN -- EXTRACT FEATURE COLUMNS
        feature_instance = feature_options.create('extract_columns', { 'columns': config.training.feature_columns })
        pipeline_components.append(('hidden_feature_extraction', feature_instance))

    ########################################################################################
    ### ADD SCALER & MODEL

        scaler_instance = scaler_options.create(config.training.scaler.name, config.training.scaler.params)
        pipeline_components.append(('scaler', scaler_instance))

        model_instance = model_options.create(config.training.model.name, config.training.model.params)
        model_instance.set_metrics(config.training.metrics)
        pipeline_components.append(('model', model_instance))

    ########################################################################################
    ### TRAIN THE PIPELINE

        pipeline = Pipeline(pipeline_components)
        pipeline.fit(dataset['train'], labels['train'])

    ########################################################################################
    ### EVALUATE PIPELINE

        overview = {
            'segment_lengths': {},
            'temp_metrics': {},
            'segment_metrics': {},
            'sklearn_pipeline': [str(item[1]) for item in pipeline.steps]
        }

        for segment_name, segment_dataset in dataset.items():
            overview['segment_lengths'][segment_name] = len(segment_dataset)
            overview['temp_metrics'][segment_name] = pipeline.score(segment_dataset, labels[segment_name])

        # COMPUTE TOTAL LENGTH OF DATASET
        overview['segment_lengths']['total'] = sum(overview['segment_lengths'].values())

        # TRANSPOSE METRICS FOR READABILITY
        for metric_name in config.training.metrics:
            container = {}

            for segment_name, segment_metrics in overview['temp_metrics'].items():
                container[segment_name] = segment_metrics[metric_name]
            overview['segment_metrics'][metric_name] = container

        del overview['temp_metrics']

        misc.pprint(overview)

    # OTHERWISE, AT LEAST ONE TEST FAILED
    # THEREFORE, BLOCK THE EXPERIMENT
    except AssertionError as error:
        print(f'\nINTERPRETER-SIDE ASSERTION ERROR:')
        print('----------------------------------------------------------------------')
        print(error)
        return False

if __name__ == '__main__':
    run()
