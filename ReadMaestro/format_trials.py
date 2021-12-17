import numpy as np
from ReadMaestro.target import MaestroTarget




def combine_targets(maestro_data, targ1, targ2, new_name=None):
    """ Combines target data for trial with more than 1 target where only 1 is
    used at a time. e.g. trials with a "FP" and a "pursuit" target where one
    turns off as the other turns on so that target motion is represented by a
    single vector.

    Changes are made in place to the maestro_data trial elements indicated by
    target_keys. Original target data for targ1 is kept and named by new_name.
    """
    target_keys = ['horizontal_target_position',
                'vertical_target_position',
                'horizontal_target_velocity',
                'vertical_target_velocity']
    if new_name is None:
        new_name = targ1 + targ2
    for trial in maestro_data:
        found_targ12 = [False, False]
        if len(trial['targets']) <= 1:
            continue
        target_mask = np.ones(len(trial['targets']), dtype='bool')
        for targ_ind, targ in enumerate(trial['targets']):
            if targ['target_name'] == targ1:
                targ1_ind = targ_ind
                found_targ12[0] = True
            elif targ['target_name'] == targ2:
                targ2_ind = targ_ind
                target_mask[targ_ind] = False
                found_targ12[1] = True
        if found_targ12[0] and found_targ12[1]:
            # Combine data for these targets
            for key in target_keys:
                isnan_t1 = np.isnan(trial[key][targ1_ind, :])
                isnan_t2 = np.isnan(trial[key][targ2_ind, :])
                combined = np.zeros(trial[key].shape[1])
                combined[~isnan_t1] += trial[key][targ1_ind, :][~isnan_t1]
                combined[~isnan_t2] += trial[key][targ2_ind, :][~isnan_t2]
                trial[key][targ1_ind, :] = combined
                trial[key] = trial[key][target_mask, :]
            trial['targets'][targ1_ind]['target_name'] = new_name
            del trial['targets'][targ2_ind]

    return None



def make_simple_trial_dict(maestro_data):
    """
    """
    keep_keys = ['events',
                 'targets',
                 'horizontal_eye_position',
                 'vertical_eye_position',
                 'horizontal_eye_velocity',
                 'vertical_eye_velocity',
                 'horizontal_target_position',
                 'vertical_target_position',
                 'horizontal_target_velocity',
                 'vertical_target_velocity']
    simple_trials = []
    for trial in maestro_data:
        simple_trial = {}
        fname, fnum = trial['filename'].split("/")[-1].split(".")
        simple_trial['filename'] = fname
        simple_trial['filenum'] = int(fnum)
        simple_trial['name'] = trial['header']['name']
        simple_trial['set_name'] = trial['header']['set_name']
        simple_trial['sub_set_name'] = trial['header']['sub_set_name']
        simple_trial['duration_ms'] = int(trial['header']['_num_saved_scans']
                                          * trial['header']['scan_interval'] * 1000)
        for k in keep_keys:
            simple_trial[k] = trial[k]
        simple_trials.append(simple_trial)

    return simple_trials


def data_to_target(maestro_data):
    """Deletes the target_keys and their data for current targets and replaces
    them with a MaestroTarget object at key 'targets' for each target in each
    trial. Modification is done in-place.

    NOTE: this function is fairly slow as the MaestroTargets run through all
    the data to 'compress' it. """
    target_keys = ['horizontal_target_position',
                    'vertical_target_position',
                    'horizontal_target_velocity',
                    'vertical_target_velocity']

    for t_ind, trial in enumerate(maestro_data):
        # Make a target object for each target in the trial
        n_targets = len(trial['targets'])
        trial['targets'] = []
        for target in range(0, n_targets):
            trial['targets'].append(MaestroTarget(trial, target))
        for key in target_keys:
            del trial[key]

    return None
