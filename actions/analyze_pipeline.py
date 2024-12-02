from common import file_utils, misc
import time, os

##############################################################################################################
##############################################################################################################

def run(pipeline, datasets, labels, yaml_config, raw_config):
    overview = {
        'segment_lengths': {},
        'segment_metrics': {},
        'sklearn_pipeline': [str(item[1]) for item in pipeline.steps]
    }

    # COMPUTE LENGTH & SCORES FOR EACH SEGMENT
    for segment_name, segment_dataset in datasets.items():
        overview['segment_lengths'][segment_name] = len(segment_dataset)
        overview['segment_metrics'][segment_name] = pipeline.score(segment_dataset, labels[segment_name])

    # COMPUTE TOTAL LENGTH OF DATASET
    overview['segment_lengths']['total'] = sum(overview['segment_lengths'].values())

    # TRANSPOSE METRICS FOR READABILITY
    overview['segment_metrics'] = misc.transpose_dict(overview['segment_metrics'])

    # DUMP PIPELINE CONFIG & RESULTS OVERVIEW TO LOGFILES
    if yaml_config.debug.create_local_files:

        # CREATE A NEW RUN DIRECTORY
        now = int(time.time())
        dir_path = f'runs/{now}'
        os.mkdir(dir_path)

        # PICKLE PIPELINE & DUMP RELEVANT LOGFILES
        file_utils.save_yaml(f'{dir_path}/pipeline.yaml', raw_config)
        file_utils.save_json(f'{dir_path}/pipeline.json', raw_config)
        file_utils.save_json(f'{dir_path}/metrics.json', overview)
        file_utils.save_pickle(f'{dir_path}/pipeline.pkl', pipeline)

    # PRINT OUT METRICS IN TABLE FORMAT
    for metric_name, metric_values in overview['segment_metrics'].items():
        misc.print_header(f'METRIC: {metric_name}')

        for segment_name, score in metric_values.items():
            misc.formatted_print([segment_name.upper(), score])

##############################################################################################################
##############################################################################################################

if __name__ == '__main__':
    run('pipeline.yaml')