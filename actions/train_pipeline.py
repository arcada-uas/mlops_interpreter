from actions import test_pipeline, create_pipeline
from common import files, types, misc

def run(yaml_path: str):

    # LOAD THE PIPELINE YAML CONFIG
    raw_config: dict = files.load_yaml(yaml_path)
    yaml_config = types.config_schema(**raw_config)

    # TEST THE PIPELINE
    success, num_tests = test_pipeline.run(yaml_config)

    # IF ONE OR MORE TESTS FAILED, TERMINATE EARLY
    if not success:
        return print('ONE OR MORE PIPELINE TESTS FAILED')

    print('\n#########################################################################################')
    print('#########################################################################################\n')
    
    # BUILD THE REAL PIPELINE & FETCH THE DATASETS
    pipeline, datasets, labels = create_pipeline.run(yaml_config)

    print('\n#########################################################################################')
    print('#########################################################################################\n')

    # FIT THE PIPELINE & GENERATE ANALYTICAL OVERVIEW
    pipeline.fit(datasets['train'], labels['train'])
    results: dict = pipeline_overview(pipeline, datasets, labels)

    # DUMP PIPELINE CONFIG & RESULTS OVERVIEW TO LOGFILES
    if yaml_config.debug.create_logfiles:
        files.logdump(raw_config, results)

##############################################################################################################
##############################################################################################################

def pipeline_overview(pipeline, datasets, labels):
    results = {
        'segment_lengths': {},
        'temp_metrics': {},
        'segment_metrics': {},
        'sklearn_pipeline': [str(item[1]) for item in pipeline.steps]
    }

    for segment_name, segment_dataset in datasets.items():
        results['segment_lengths'][segment_name] = len(segment_dataset)
        results['temp_metrics'][segment_name] = pipeline.score(segment_dataset, labels[segment_name])

    # COMPUTE TOTAL LENGTH OF DATASET
    results['segment_lengths']['total'] = sum(results['segment_lengths'].values())

    # FISH OUT THE LIST OF METRICS
    metric_names = list(results['temp_metrics'][list(results['temp_metrics'].keys())[0]].keys())

    # TRANSPOSE METRICS FOR READABILITY
    for metric_name in metric_names:
        container = {}

        for segment_name, segment_metrics in results['temp_metrics'].items():
            container[segment_name] = segment_metrics[metric_name]

        results['segment_metrics'][metric_name] = container

    # GET RID OF THE TEMP DICT
    del results['temp_metrics']

    misc.pprint(results)
    return results

##############################################################################################################
##############################################################################################################

if __name__ == '__main__':
    run('pipeline.yaml')