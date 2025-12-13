import pandas as pd

def parse_stic_lumi_file(path):
    data = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # skip comment or empty lines
            if not line or line.startswith("!"):
                continue

            # split the line into tokens
            parts = line.split()
            # Example line:
            # 45043 1 2058 940512 223745 45.612 95 1.724 +- 0.180 .436E-03 0 5 1257
            #
            # indices:
            # 0 run
            # 1 file
            # 2 fill
            # 3 date (YYMMDD)
            # 4 time (HHMMSS)
            # 5 energy
            # 6 bhabhas
            # 7 lumi_value
            # 8 "+-"
            # 9 lumi_unc
            # 10 fill_sys
            # 11 status
            # 12 event_start
            # 13 event_end

            run = int(parts[0])
            file_num = int(parts[1])
            fill = int(parts[2])
            date = parts[3]
            time = parts[4]
            energy = float(parts[5])
            bhabhas = int(parts[6])
            lumi_value = float(parts[7])
            lumi_unc = float(parts[9])
            fill_sys = float(parts[10])
            status = int(parts[11])
            event_start = int(parts[12])
            event_end = int(parts[13])

            data.append({
                "run": run,
                "file": file_num,
                "fill": fill,
                "date": date,
                "time": time,
                "energy": energy,
                "bhabhas": bhabhas,
                "lumi": lumi_value,
                "lumi_unc": lumi_unc,
                "fill_sys": fill_sys,
                "status": status,
                "event_start": event_start,
                "event_end": event_end,
            })
        data = pd.DataFrame(data)

    return data


if __name__ == "__main__":
    filename = "/cvmfs/delphi.cern.ch/releases/almalinux-9-x86_64/v16082025/dstana/161018/dat/STILUM94.03FEB98"
    records = parse_stic_lumi_file(filename)

    print(records)
    total_lumi = records['lumi'].sum()
    lumi_unc = records['lumi_unc'].sum()
    print(f"Total Lumi: {total_lumi} +/- {lumi_unc} nb^-1")
