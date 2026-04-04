import ROOT
import numpy as np
import awkward as ak


def build_response(var_recon, var_truth, num_bins, weight=None):
    try:
        response = ROOT.RooUnfoldResponse(num_bins, -0.5, num_bins - 0.5)
    except Exception as e:
        print(f"Error occurred while creating RooUnfoldResponse: {e}")
        response = ROOT.RooUnfoldResponse(num_bins, -0.5, num_bins - 0.5)
    weight = weight if weight is not None else np.ones_like(var_recon)
    for reco_val, truth_val, w in zip(var_recon, var_truth, weight):
        if not np.isnan(truth_val) and not np.isnan(reco_val):
            response.Fill(reco_val, truth_val, w)
        elif not np.isnan(truth_val):
            response.Miss(truth_val, w)
    return response


def build_TH1D(hname, var, num_bins, weight=None):
    h = ROOT.TH1D(hname, hname, num_bins, -0.5, num_bins - 0.5)
    weight = weight if weight is not None else np.ones_like(var)
    for val, w in zip(var, weight):
        if not np.isnan(val):
            h.Fill(val, w)
    return h


def plot_unfolded_results(hUnfold, save_path, h_truth=None, h_reco=None, var_name="Observable"):
    canvas = ROOT.TCanvas("RooUnfold", "SVD")
    ROOT.gStyle.SetOptStat(0)

    # plot title and axis labels
    hUnfold.SetTitle(f"Unfolded {var_name} distribution")
    hUnfold.GetXaxis().SetTitle(var_name)
    hUnfold.GetYaxis().SetTitle("Events")

    hUnfold.Draw("HIST E1")

    legend = ROOT.TLegend(0.65, 0.7, 0.88, 0.88)  # (x1, y1, x2, y2) in NDC
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)  # transparent
    legend.AddEntry(hUnfold, "Unfolded", "l")

    if h_reco is not None:
        h_reco.Draw("SAME HIST E1")
        h_reco.SetLineColor(2)
        legend.AddEntry(h_reco, "Reco (before unfolding)", "l")
    if h_truth is not None:
        h_truth.SetLineColor(8)
        h_truth.Draw("SAME HIST E1")
        legend.AddEntry(h_truth, "Truth", "l")


    legend.Draw()

    if save_path:
        canvas.SaveAs(str(save_path))