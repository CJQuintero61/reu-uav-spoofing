"""
csv_merger.py
06/22/2026

This file converts a ulog file to many csv files then merges relevant csv files into one csv file.

To use:
    instantiate this class and specify the input ulog file, the input folder to store the
    csv files, and the output file to store the merged csv file

    ex:
    merger = CSVMerger(
        ulog_file="./data/raw/flight_test.ulg",
        input_folder="./data/csv/test_flight",
        output_file="./data/merged/test_flight_merged.csv",
        label = "benign"
    )
    merger.convert_ulog_file()
    merger.merge_csv_files()

NOTE: Please name ulog files and create new directories for new flights to keep
data organized. Each ulog file produces many csv files, so it is important to keep
them organized in separate folders.

TODO: Eventually change the label to be (0/1) instead of (benign/malicious)
"""
import os
import subprocess
import pandas as pd

class CSVMerger:
    # include only import csv files that are relevant to gps spoofing
    CSV_TOPICS = [
        "vehicle_attitude_0.csv",
        "vehicle_global_position_0.csv",
        "vehicle_gps_position_0.csv",
        "vehicle_local_position_0.csv",
        "estimator_gps_status_0.csv",
        "estimator_innovations_0.csv",
        "estimator_innovation_test_ratios_0.csv",
    ]


    def __init__(self, ulog_file, input_folder, output_file, label):
        """
        params:
        ulog_file: the ulog file to convert to csv files
        input_folder: the folder to store (and read) the csv files
        output_file: the file to store the merged csv file
        label: the label to assign to the merged csv dataframe (0 = benign, 1 = malicious)
        """
        self.ulog_file = ulog_file
        self.input_folder = input_folder
        self.output_file = output_file
        self.label = label
        self.merged_df = None
    

    def convert_ulog_file(self):
        # convert ulog file to csv files
        subprocess.run(f"ulog2csv {self.ulog_file} -o {self.input_folder}", stdout=subprocess.DEVNULL)


    def merge_csv_files(self):
        # merge only relevant csv files
        files = self._filter_files()

        frames = []
        for file in files:
            df = pd.read_csv(os.path.join(self.input_folder, file))
            frames.append(df)

        # concat the frames and drop duplicate columns
        merged_df = pd.concat(frames, axis=1)
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

        # defragment dataframe to silence a warning
        merged_df = merged_df.copy()

        # assign the label for this flight to the merged dataframe
        merged_df['label'] = self.label

        # store the merged dataframe as a csv file
        merged_df.to_csv(self.output_file, index=False)

        self.merged_df = merged_df


    def _filter_files(self):
        """filters out files that are not relevant to gps spoofing"""
        files = []
        for f in os.listdir(self.input_folder):
            if not f.endswith(".csv"):
                continue

            if any(f.endswith(topic) for topic in self.CSV_TOPICS):
                files.append(f)
        
        return files


# change constructor params as needed or call in a separate module
if __name__ == "__main__":
    merger = CSVMerger(
        ulog_file="./data/raw/flight_test.ulg",
        input_folder="./data/csv/test_flight",
        output_file="./data/merged/test_flight_merged.csv",
        label = "benign"
    )
    merger.convert_ulog_file()
    merger.merge_csv_files()

    print(f'Shape of merged data: {merger.merged_df.shape}')