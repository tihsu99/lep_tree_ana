import ROOT
import argparse
import os
ROOT.gROOT.SetBatch(True)

def plot_response_matrices(response_matrices_list, output_file_name, titles=[]):
    # c = ROOT.TCanvas("c", "Response Matrices", 800, 600)

    num_matrices = len(response_matrices_list)
    num_per_row = 3
    num_per_column = (num_matrices + num_per_row - 1) // num_per_row

    c = ROOT.TCanvas("c", "Response Matrices", 400*num_per_row, 300*num_per_column)
    c.Divide(num_per_row, num_per_column)

    for i, response_matrix in enumerate(response_matrices_list):
        c.cd(i + 1)
        r = response_matrix.HresponseNoOverflow()
        r.SetStats(0)
        r.Draw("COLZ")
        if i < len(titles):
            r.SetTitle(titles[i])

    c.SaveAs(output_file_name)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", nargs='+', help="Input ROOT files that contain the response matrices")
    parser.add_argument("-o", "--output_dir", default=None, help="Output directory")
    args = parser.parse_args()

    for input_file in args.input_root:
        output_dir = args.output_dir if args.output_dir else os.path.dirname(input_file)
        os.makedirs(output_dir, exist_ok=True)

        output_file_name = os.path.join(output_dir, os.path.basename(input_file).replace(".root", "_response_matrices.pdf"))

        rf = ROOT.TFile(input_file)
        response_matrices = []
        titles = []
        for key in rf.GetListOfKeys():
            rf_name = key.GetName()
            if not ("cos_theta" in rf_name): continue
            response_matrices.append(rf.Get(rf_name))
            titles.append(rf_name)

        plot_response_matrices(response_matrices, output_file_name, titles)