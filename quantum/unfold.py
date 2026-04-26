import ROOT
import numpy as np
import awkward as ak
from quantum.observables_builder import Hist


def build_response(var_recon, var_truth, num_bins, weight=None, name="response"):
    try:
        response = ROOT.RooUnfoldResponse(num_bins, -0.5, num_bins - 0.5, name, name)
    except Exception as e:
        print(f"Error occurred while creating RooUnfoldResponse: {e}")
        response = ROOT.RooUnfoldResponse(num_bins, -0.5, num_bins - 0.5)
    weight = weight if weight is not None else np.ones_like(var_recon)
    for reco_val, truth_val, w in zip(var_recon, var_truth, weight):
        if not np.isnan(truth_val) and not np.isnan(reco_val):
            response.Fill(reco_val, truth_val, w)
        elif (not np.isnan(truth_val)) and np.isnan(reco_val):
            response.Miss(truth_val, w)
        elif (not np.isnan(reco_val)) and np.isnan(truth_val):
            response.Fake(reco_val, w)
    return response


def build_TH1D(hname, var, num_bins, weight=None):
    h = ROOT.TH1D(hname, hname, num_bins, -0.5, num_bins - 0.5)
    weight = weight if weight is not None else np.ones_like(var)
    for val, w in zip(var, weight):
        if not np.isnan(val):
            h.Fill(val, w)
    return h


def build_Hist_from_TH1D(h, bin_edges=None):
    if bin_edges is None:
        bin_edges = np.array([h.GetBinLowEdge(i) for i in range(1, h.GetNbinsX() + 2)])
    values = np.array([h.GetBinContent(i) for i in range(1, h.GetNbinsX() + 1)])
    errors = np.array([h.GetBinError(i) for i in range(1, h.GetNbinsX() + 1)])
    # errors = np.sqrt(values)  # use sqrt of values as error to mimic Poisson unc
    return Hist(bin_edges=bin_edges, values=values, errors=errors)


def plot_unfolded_results(hUnfold, save_path, h_truth=None, h_reco=None, var_name="Observable"):
    canvas = ROOT.TCanvas("RooUnfold", "RooUnfold", 450, 500)
    ROOT.gStyle.SetOptStat(0)

    # create mainPad and ratioPad if truth distribution is provided for comparison
    if h_truth is not None:
        mainPad = ROOT.TPad("mainPad", "top", 0., 0.3, 1.0, 1.)
        ratioPad = ROOT.TPad("ratioPad", "bottom", 0., 0., 1.0, 0.3)
        mainPad.SetBottomMargin(0.01)  # remove x-axis labels and ticks from mainPad
        ratioPad.SetTopMargin(0.02)  # remove space between mainPad and ratioPad
        ratioPad.SetBottomMargin(0.4)  # increase bottom margin for ratioPad
        mainPad.Draw()
        ratioPad.Draw()
    else:
        mainPad = ROOT.TPad("mainPad", "mainPad", 0, 0, 1, 1)
        mainPad.Draw()

    mainPad.cd()
    # plot title and axis labels
    hUnfold.SetTitle(f"Unfolded {var_name} distribution")
    hUnfold.GetXaxis().SetTitle(var_name)
    hUnfold.GetYaxis().SetTitle("Events")
    hUnfold.GetYaxis().SetRangeUser(-1, hUnfold.GetMaximum() * 1.2)

    hUnfold.Draw("SAME HIST E1")

    legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)  # (x1, y1, x2, y2) in NDC
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


    hUnfold.Draw("SAME HIST E1") # redraw unfolded histogram to make sure it's on top of the legend
    legend.Draw("SAME")

    # draw ratio plot if truth is provided
    if h_truth is not None:
        ratioPad.cd()
        hRatio = hUnfold.Clone("hRatio")
        hRatio.Divide(h_truth)
        hRatio.SetTitle("")
        hRatio.GetXaxis().SetTitle(var_name)
        hRatio.GetXaxis().SetTitleSize(12/(ratioPad.GetWh()*ratioPad.GetAbsHNDC()))
        hRatio.GetXaxis().SetLabelSize(12/(ratioPad.GetWh()*ratioPad.GetAbsHNDC()))
        hRatio.GetYaxis().SetTitle("Unfolded / Truth")
        hRatio.GetYaxis().SetRangeUser(0.71, 1.29)
        hRatio.GetYaxis().SetTitleSize(12/(ratioPad.GetWh()*ratioPad.GetAbsHNDC()))
        hRatio.GetYaxis().SetLabelSize(11/(ratioPad.GetWh()*ratioPad.GetAbsHNDC()))

        hRatio.GetYaxis().SetTitleOffset(0.55)
        hRatio.GetXaxis().SetTitleOffset(0.9)

        hRatio.SetMarkerStyle(20)
        hRatio.SetMarkerSize(0.4)
        hRatio.SetLineColor(1)
        hRatio.SetLineWidth(2)
        hRatio.Draw("SAME LPEX0")
        # add horizontal line at y=1
        line = ROOT.TLine(hRatio.GetXaxis().GetXmin(), 1, hRatio.GetXaxis().GetXmax(), 1)
        line.SetLineStyle(2)
        line.Draw("SAME")


    if save_path:
        canvas.SaveAs(str(save_path))