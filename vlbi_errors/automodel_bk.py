import os
import numpy as np
import shutil

from automodel import (AutoModeler, TotalFluxStopping,
                       RChiSquaredStopping,
                       AddedComponentFluxLessRMSFluxStopping,
                       AddedComponentFluxLessRMSStopping,
                       AddedTooDistantComponentStopping,
                       AddedTooSmallComponentStopping,
                       AddedNegativeFluxComponentStopping,
                       AddedOverlappingComponentStopping,
                       NLastDifferencesAreSmall,
                       NLastDifferesFromLast,
                       NLastJustStop,
                       FluxBasedModelSelector,
                       SizeBasedModelSelector,
                       SmallSizedComponentsModelFilter,
                       NegativeFluxComponentModelFilter,
                       OverlappingComponentsModelFilter,
                       ComponentAwayFromSourceModelFilter)


def automodel_bk(simulated_uv_fits_path, best_dfm_model_path, core_elliptic=False,
                 n_max_components=20, mapsize_clean=(512, 0.1), out_dir=None,
                 path_to_script=None, start_model_fname=None, niter_difmap=100,
                 n_just_stop=None, remove_nonbest_models=True,
                 ra_range=None, dec_range=None):

    automodeler = AutoModeler(simulated_uv_fits_path, out_dir, path_to_script,
                              n_comps_terminate=n_max_components,
                              core_elliptic=core_elliptic,
                              mapsize_clean=mapsize_clean,
                              show_difmap_output_clean=False,
                              show_difmap_output_modelfit=True,
                              niter_difmap=niter_difmap,
                              ra_range_plot=ra_range,
                              dec_range_plot=dec_range)

    # Stoppers define when to stop adding components to model
    # stoppers = [AddedComponentFluxLessRMSStopping(mode="or"),
    #             AddedComponentFluxLessRMSFluxStopping(),
    stoppers = [AddedTooDistantComponentStopping(n_rms=2, mode="or"),
                AddedTooSmallComponentStopping(),
                # AddedNegativeFluxComponentStopping(),
                # for 0430 exclude it
                # AddedOverlappingComponentStopping(),
                NLastDifferesFromLast(),
                NLastDifferencesAreSmall(),
                # TotalFluxStopping(rel_threshold=0.05, mode="or"),
                RChiSquaredStopping(delta_min=0.1)]
    if n_just_stop is not None:
        stoppers.insert(2, NLastJustStop(int(n_just_stop)))
    # Keep iterating while this stopper fires
    # TotalFluxStopping(rel_threshold=0.2, mode="while")]
    # Selectors choose best model using different heuristics
    selectors = [FluxBasedModelSelector(delta_flux=0.001),
                 SizeBasedModelSelector(delta_size=0.001)]

    # Run number of iterations that is defined by stoppers
    automodeler.run(stoppers, start_model_fname=start_model_fname)

    # Select best model using custom selectors
    files = automodeler.fitted_model_paths
    files_toremove = files[:]
    if selectors:
        id_best = max(selector.select(files) for selector in selectors)
    else:
        id_best = len(files)-1
    files = files[:id_best + 1]

    # Filters additionally remove complex models with non-physical
    # components (e.g. too small faint component or component
    # located far away from source.)
    filters = [SmallSizedComponentsModelFilter(),
               ComponentAwayFromSourceModelFilter(ccimage=automodeler.ccimage),
               # NegativeFluxComponentModelFilter(),
               # ToElongatedCoreModelFilter()]
               #OverlappingComponentsModelFilter(),
               ]

    # Additionally filter too small, too distant components
    for fn in files[::-1]:
        if np.any([flt.do_filter(fn) for flt in filters]):
            id_best -= 1
        else:
            break
    automodeler.best_id = id_best
    print("Best model is {}".format(files[id_best]))

    best_model = files[id_best]
    first_model = files[0]

    automodeler.plot_results(id_best)

    # Leaving only best model
    if remove_nonbest_models:
        files_toremove.remove(best_model)
        try:
            files_toremove.remove(first_model)
        except ValueError:
            pass
        if files_toremove:
            for fn in files_toremove:
                os.unlink(fn)
    # Move best model to the specified location
    save_dir, _ = os.path.split(best_dfm_model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.move(best_model, best_dfm_model_path)

    return automodeler


if __name__ == "__main__":

    path_to_script = '/home/ilya/github/vlbi_errors/difmap/final_clean_nw'
    out_dir = "/home/ilya/github/bck/jetshow/automodel"
    data_dir = "/home/ilya/github/bck/jetshow/uvf_mf_adds/"

    for i in range(1, 31):
        uv_fits_fname = "bk_{}_8.1.fits".format(str(i).zfill(2))
        uv_fits_path = os.path.join(data_dir, uv_fits_fname)

        try:
            automodeler = AutoModeler(uv_fits_path, out_dir, path_to_script,
                                      n_comps_terminate=20,
                                      core_elliptic=True,
                                      mapsize_clean=(512, 0.1))
        except IOError:
            continue
        # Stoppers define when to stop adding components to model
        stoppers = [TotalFluxStopping(),
                    AddedComponentFluxLessRMSStopping(mode="or"),
                    AddedComponentFluxLessRMSFluxStopping(),
                    AddedTooDistantComponentStopping(mode="or"),
                    AddedTooSmallComponentStopping(),
                    AddedNegativeFluxComponentStopping(),
                    # for 0430 exclude it
                    # AddedOverlappingComponentStopping(),
                    NLastDifferesFromLast(),
                    NLastDifferencesAreSmall()]
        # Keep iterating while this stopper fires
        # TotalFluxStopping(rel_threshold=0.2, mode="while")]
        # Selectors choose best model using different heuristics
        selectors = [FluxBasedModelSelector(delta_flux=0.001),
                     SizeBasedModelSelector(delta_size=0.001)]

        # Run number of iterations that is defined by stoppers
        automodeler.run(stoppers)

        # Select best model using custom selectors
        files = automodeler.fitted_model_paths
        files_toremove = files[:]
        id_best = max(selector.select(files) for selector in selectors)
        files = files[:id_best + 1]

        # Filters additionally remove complex models with non-physical
        # components (e.g. too small faint component or component
        # located far away from source.)
        filters = [SmallSizedComponentsModelFilter(),
                   ComponentAwayFromSourceModelFilter(ccimage=automodeler.ccimage),
                   NegativeFluxComponentModelFilter(),
                   # ToElongatedCoreModelFilter()]
                   OverlappingComponentsModelFilter()]

        # Additionally filter too small, too distant components
        for fn in files[::-1]:
            if np.any([flt.do_filter(fn) for flt in filters]):
                id_best -= 1
            else:
                break
        print("Best model is {}".format(files[id_best]))

        best_model = files[id_best]
        first_model = files[0]

        automodeler.plot_results(id_best)

        # Leaving only best model
        files_toremove.remove(best_model)
        try:
            files_toremove.remove(first_model)
        except ValueError:
            pass
        if files_toremove:
            for fn in files_toremove:
                os.unlink(fn)