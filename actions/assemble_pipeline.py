from common import file_utils, types, misc, interpreter
from sklearn.pipeline import Pipeline

from mappings.data_retrieval import repository as retrieval_options
from mappings.features import repository as feature_options
from mappings.models import repository as model_options
from mappings.scalers import repository as scaler_options
from mappings.segmentation import repository as segmentation_options
from components.segmentation.generate_labels import generate_segment_labels

##############################################################################################################
##############################################################################################################

def run(yaml_config: types.config_schema):
    try:
        misc.print_header('ASSEMBLING PIPELINE')
        pipeline_components = []

    ########################################################################################
    ### LOAD & SEGMENT DATASET

        retrieval_method = retrieval_options.get(yaml_config.dataset.method)
        datasets = []

        # LIMIT QUERY WHEN UNITTESTING
        if yaml_config.debug.limit_dataset > 0:
            datasets = retrieval_method(**yaml_config.dataset.params, unittest_limit=yaml_config.debug.limit_dataset)
        
        # OTHERWISE, FETCH FULL DATASET
        else:
            datasets = retrieval_method(**yaml_config.dataset.params)

        misc.formatted_print(['[1/9] FETCHED DATASET', f'n={len(datasets)}'])

        # THEN SEGMENT WHATEVER WAS QUERIED
        segmentation_method = segmentation_options.get(yaml_config.training.segmentation.method)
        datasets = segmentation_method(**yaml_config.training.segmentation.params, dataset=datasets)

        for segment_name, subset in datasets.items():
            misc.formatted_print([f"[2/9] CREATED '{segment_name}' SEGMENT", f'n={len(subset)}'])

    ########################################################################################
    ### ADD FIRST LAYER OF FEATURES

        # HIDDEN -- CONVERT INPUT TO DATAFRAME
        feature_instance = feature_options.create('to_dataframe')
        pipeline_components.append(('hidden_to_df', feature_instance))

        misc.formatted_print(['[3/9] ADDED HIDDEN FEATURE', 'to_dataframe'])

        # NORMAL -- APPLY EACH CUSTOM FEATURE
        for nth, feature in enumerate(yaml_config.features):
            feature_instance = feature_options.create(feature.method, feature.params)
            feature_label = f'{nth}_{feature.method}'
            pipeline_components.append((feature_label, feature_instance))

            misc.formatted_print([f'[4/9] ADDED STANDARD FEATURE', feature.method])

        # HIDDEN -- DROP ROWS WITH NANS
        feature_instance = feature_options.create('drop_nan_rows')
        pipeline_components.append(('hidden_drop_nans', feature_instance))

        misc.formatted_print(['[5/9] ADDED HIDDEN FEATURE', 'drop_nan_rows'])

    ########################################################################################
    ### GENERATE LABELS WITH THE CURRENT SET OF FEATURES

        labels, minimum_batch_window = generate_segment_labels(
            datasets=datasets,
            pipeline=pipeline_components,
            label_column=yaml_config.training.label_column,
            feature_columns=yaml_config.training.feature_columns
        )

        misc.formatted_print(['[6/9] EXTRACTED LABELS', yaml_config.training.label_column])

    ########################################################################################
    ### ADD SECOND LAYER OF FEATURES

        # HIDDEN -- EXTRACT FEATURE COLUMNS
        feature_instance = feature_options.create('extract_features', { 'columns': yaml_config.training.feature_columns })
        pipeline_components.append(('hidden_feature_extraction', feature_instance))

        misc.formatted_print(['[7/9] ADDED HIDDEN FEATURE', 'extract_features'])

    ########################################################################################
    ### ADD SCALER & MODEL

        scaler_instance = scaler_options.create(yaml_config.training.scaler.method, yaml_config.training.scaler.params)
        pipeline_components.append(('scaler', scaler_instance))

        misc.formatted_print(['[8/9] ADDED SCALER', yaml_config.training.scaler.method])

        # FETCH THE DESIRED MODEL
        model_name_path = f'{yaml_config.training.model.type}.{yaml_config.training.model.method}'
        model_instance = model_options.create(model_name_path, yaml_config.training.model.params)

        # ADD MODEL TYPE PREFIX TO METRIC NAMES & PUSH THEM TO MODEL OBJECT
        metric_paths = [f'{yaml_config.training.model.type}.{x}' for x in yaml_config.training.metrics]
        model_instance.set_metrics(metric_paths)

        # ADD MINIMUM FEATURE BATCH WINDOW TO MODEL
        model_instance.set_prediction_window(minimum_batch_window)

        # ADD MODEL TO PIPELINE
        pipeline_components.append(('model', model_instance))

        misc.formatted_print(['[9/9] ADDED MODEL', model_name_path])

    ########################################################################################
    ### ALL COMPONENTS GATHERED, CONVERT IT TO THE PROPER PIPELEINE FORMAT

        pipeline = Pipeline(pipeline_components)
        return True, pipeline, datasets, labels, minimum_batch_window

    except Exception as error:
        interpreter.handle_errors(error)
        return False, None, None, None, None

##############################################################################################################
##############################################################################################################

# BUILD THE PIPELINE
if __name__ == '__main__':
    raw_config: dict = file_utils.load_yaml('pipeline.yaml')
    yaml_config = types.config_schema(**raw_config)
    run(yaml_config)