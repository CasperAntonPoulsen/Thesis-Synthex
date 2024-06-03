from arguments import add_nbia_args
import argparse
import pandas as pd
from tcia_utils import nbia
import ast
import os
import datetime


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = add_nbia_args(parser)
    known_args, _ = parser.parse_known_args()

    collection = known_args.collection_name
    output_dir = known_args.output_dir
    api_url = known_args.api_url

    series = pd.DataFrame()
    seriesDescription = nbia.getSeries(collection, api_url = api_url)
    series = pd.concat([series, pd.DataFrame(seriesDescription)], ignore_index=True)




    series.to_csv(f'{output_dir}/study_metadata_{datetime.date.today()}.csv')

    # Do not download series I already have. This can corrupt volumes if they are only partially downloaded
    series = series[~series["SeriesInstanceUID"].isin(os.listdir(output_dir))].to_dict(orient="records")

    df = nbia.downloadSeries(series,path=output_dir, format="df")