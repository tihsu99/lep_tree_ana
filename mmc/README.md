Adapted from https://github.com/UW-EPE-ML/Quantum_Informaton_Analysis/tree/main/mmc

To build the PDF, run `python build_pdf.py build <input_parquet_file> --output <output_hdf5_file>`. This will read the input parquet file, build the PDF for each p_tau bin for lep and rho decay, and save the results to an HDF5 file.

To perform closure tests, run `python build_pdf.py closure --input-parquet <input_parquet_file> --input-pdf <input_hdf5_file> --output <output_directory>`. This will read the PDF parameters from the HDF5 file, apply the PDF to the input parquet data, and generate closure test plots for each p_tau bin, saving them to the specified output directory.