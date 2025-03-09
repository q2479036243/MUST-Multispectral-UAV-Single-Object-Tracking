from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.network_path = '/data3/QinHaolin/MUST/UNTrack/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/data/users/qinhaolin01/MUST-BIT/UNTrack'
    settings.result_plot_path = '/data3/QinHaolin/MUST/UNTrack/output/test/result_plots'
    settings.results_path = '/data3/QinHaolin/MUST/UNTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data3/QinHaolin/MUST/UNTrack/output'
    settings.segmentation_path = '/data3/QinHaolin/MUST/UNTrack/output/test/segmentation_results'
    
    settings.musthsi_path = '/data/users/qinhaolin01/MUST-BIT/datasets/MUST-BIT/HSICV/'

    return settings

