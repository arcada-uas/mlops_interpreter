from actions import assemble_pipeline, test_pipeline, analyze_pipeline
from common import file_utils, types, misc, interpreter

##############################################################################################################
##############################################################################################################

def run(yaml_path: str):
    try:

        # LOAD THE PIPELINE YAML CONFIG
        raw_config: dict = file_utils.load_yaml(yaml_path)
        yaml_config = types.config_schema(**raw_config)

        # TEST THE PIPELINE
        success, num_tests = test_pipeline.run(yaml_config)
        if success is not True: return

        # BUILD THE REAL PIPELINE & FETCH THE DATASETS
        success, pipeline, datasets, labels, min_batch_window = assemble_pipeline.run(yaml_config)
        if success is not True: return

        # FIT THE PIPELINE
        misc.print_header('TRAINING PIPELINE')
        pipeline.fit(datasets['train'], labels['train'])

        # ANALYZE MODEL ACCURACY & SAVE LOGS
        analyze_pipeline.run(pipeline, datasets, labels, yaml_config, raw_config)

    except Exception as error:
        interpreter.handle_errors(error)

##############################################################################################################
##############################################################################################################

if __name__ == '__main__':
    run('pipeline.yaml')