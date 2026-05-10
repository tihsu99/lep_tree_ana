import awkward as ak
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import observables_builder as ob
from utils.common_functions import print_and_write_to_opened_file, get_event_category_from_signal_name


if __name__ == "__main__":
    # input_parquet = '/eos/user/c/cmo/project/ZtautauLep/tree_ana/run/20260426-hadhad/Ztautau/filtered___raw.parquet'
    # events = ak.from_parquet(input_parquet)
    events = []
    for f in glob.glob('/pscratch/sd/c/cmo/LEP/tree_ana/run/20260427-dataset/Ztautau_*/filtered___raw.parquet'):
        events.append(ak.from_parquet(f))
    events = ak.concatenate(events)

    output_base_dir = f'example_plots/'
    output_text_file = os.path.join(output_base_dir, 'truth_analysis.txt')
    os.makedirs(output_base_dir, exist_ok=True)
    f_out = open(output_text_file, 'w')


    decay_products = [
        'pi', 
        'rho', 
        'e', 
        'mu'
    ]
    channel_results = {}
    for dp_pos in decay_products:
        for dp_neg in decay_products:
            channel_name = f"{dp_pos}{dp_neg}"
            output_dir = os.path.join(output_base_dir, channel_name)
            os.makedirs(output_dir, exist_ok=True)

            event_category = get_event_category_from_signal_name(channel_name)

            # channel selection
            mask = events['event_category'] == event_category
            # phase space selection
            mask = mask & (events['truth_theta_cm'] > 0.6) & (events['truth_mtautau'] > 80)

            selected_events = events[mask]
            print_and_write_to_opened_file(f"Channel: {channel_name}, Number of selected events: {len(selected_events)}", f_out)

            # plot quantum observable distribution
            hist_dict = {}
            for obs_key in ob.get_observable_names():
                obs = "truth_" + obs_key
                if obs not in selected_events.fields:
                    print_and_write_to_opened_file(f"Observable {obs} not found in events for channel {channel_name}", f_out)
                    continue
                obs_values = ak.to_numpy(selected_events[obs], allow_missing=False)
                weights = np.ones_like(obs_values)
                bin_edges = np.linspace(-1, 1, 51)
                fig, ax = plt.subplots()
                # get Hist
                hist_values, _ = np.histogram(obs_values, bins=bin_edges, weights=weights)
                hist_errors, _ = np.histogram(obs_values, bins=bin_edges, weights=weights**2)
                hist_errors = np.sqrt(hist_errors) 
                hist_dict[obs_key] = ob.Hist(bin_edges=bin_edges, values=hist_values, errors=hist_errors)

                # plot with error bars
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.hist(bin_edges[:-1], bins=bin_edges, weights=hist_values, histtype='step', color='black')
                ax.errorbar(bin_centers, hist_values, yerr=hist_errors, fmt='.', color='black')
                mean, err_of_mean = ob.get_mean_and_err_of_mean(bin_centers, hist_values, hist_errors)
                ax.axvline(mean, linestyle='--', label=f'Weighted Mean: {mean:.4f} ± {err_of_mean:.4f}')
                ax.set_xlabel(f'{obs}')
                ax.set_ylabel('Weighted Counts')
                ax.set_title(f'{channel_name} Channel: {obs} Distribution')
                ax.legend()
                ax.set_ylim(bottom=0)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{obs}.png")
                plt.close()

            # derive quantum results
            BC_matrices, quantum_results = ob.derive_results(hist_dict, selected_events['analyzing_power_a'][0]*(-1), selected_events['analyzing_power_b'][0])
            dict_to_print = {**BC_matrices, **quantum_results}
            channel_results[channel_name] = dict_to_print
            for key, value in dict_to_print.items():
                nominal, err_up, err_down = value.value, value.err_up, value.err_down
                print_and_write_to_opened_file(f"    {key}: {nominal:.4f} +{err_up:.4f}/-{err_down:.4f}", f_out)
            print_and_write_to_opened_file("\n", f_out)

    f_out.close()

    # compare results across channels
    pdf_name = os.path.join(output_base_dir, "quantum_results_comparison_across_channels.pdf")
    results_per_page = 8
    n_rows, n_cols = 4, 2
    result_keys = list(dict_to_print.keys())
    with PdfPages(pdf_name) as pdf:
        for page_start in range(0, len(result_keys), results_per_page):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.27, 11.69), squeeze=False)
            page_result_keys = result_keys[page_start:page_start + results_per_page]
            for ax, result_key in zip(axes.flat, page_result_keys):
                channels = list(channel_results.keys())
                x_pos = np.arange(len(channels))
                values = [channel_results[channel][result_key].value for channel in channels]
                values_up = [channel_results[channel][result_key].value + channel_results[channel][result_key].err_up for channel in channels]
                values_down = [channel_results[channel][result_key].value - channel_results[channel][result_key].err_down for channel in channels]
                ax.errorbar(x_pos, values, yerr=[np.array(values) - np.array(values_down), np.array(values_up) - np.array(values)], fmt='.', label=result_key, color='black', ecolor='black', capsize=3)

                # plot weighted mean and error of weighted mean. weight is 1/err^2
                weights = [1/(channel_results[channel][result_key].err_up**2) for channel in channels]
                mean = np.average(values, weights=weights)
                mean_err = np.sqrt(1/np.sum(weights))
                ax.axhspan(mean - mean_err, mean + mean_err, color='grey', alpha=0.2, label='Mean uncertainty')
                ax.axhline(mean, color='grey', linestyle='--', label=f'Mean: {mean:.4f}±{mean_err:.4f}')
                ax.set_xlim(-0.5, len(channels) - 0.5)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(channels)
                ax.set_xlabel('Decay Channel', fontsize=8)
                ax.set_ylabel('Value', fontsize=8)
                ax.set_title(result_key, fontsize=9)
                ax.tick_params(axis='x', labelrotation=45, labelsize=7)
                ax.tick_params(axis='y', labelsize=7)
                ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.5)
                ax.legend(fontsize=7)
            for ax in axes.flat[len(page_result_keys):]:
                ax.axis('off')
            fig.suptitle('Comparison of Quantum Results Across Channels', fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)
    
