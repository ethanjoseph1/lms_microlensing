import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

# -----------------------------
# 0) YOUR BASE LC LOADER (as-is)
# -----------------------------


filters = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}  # SDSS-like hardcoded order
bands_sdss = ["u", "g", "r", "i", "z"]


def concat_light_curves_df(
   filter_object_ids=None,
   skip=None,
   N=None,
   data_dir="./light_curves",
):
   cat = pd.read_parquet(f"{data_dir}/Catalog.parquet").set_index("idx")


   sdss = pd.read_parquet(f"{data_dir}/dr16s82_sdssLCRaw.parquet")
   ps1 = pd.read_parquet(f"{data_dir}/dr16s82_ps1LCRaw.parquet")
   ztf = pd.read_parquet(f"{data_dir}/dr16s82_ZuberLCRaw.parquet")


   sdss = sdss.loc[sdss["mjd"].notna()]


   sdss_objids = pd.Index(sdss["objectId"].unique())
   if filter_object_ids is not None:
       sdss_objids = sdss_objids.intersection(pd.Index(filter_object_ids))


   cat = cat.loc[cat["objectId"].isin(sdss_objids)]
   if skip is not None:
       cat = cat.iloc[skip:]
   if N is not None:
       cat = cat.iloc[:N]


   if len(cat) == 0:
       return pd.DataFrame(columns=["objectId", "band", "time", "mag", "magerr", "survey"])


   cat_keys = cat.reset_index(drop=False).copy()
   cat_keys["ps1objID"] = pd.to_numeric(cat_keys["ps1objID"], errors="coerce").astype("Int64")


   offset_wide = pd.DataFrame(
       {
           "objectId": cat_keys["objectId"].values,
           "g": cat_keys["sdss_g_qg"].values - cat_keys["ps1_g_qg"].values,
           "r": cat_keys["sdss_r_qg"].values - cat_keys["ps1_r_qg"].values,
           "i": cat_keys["sdss_i_qg"].values - cat_keys["ps1_i_qg"].values,
           "z": cat_keys["sdss_z_qg"].values - cat_keys["ps1_z_qg"].values,
       }
   )
   offset_long = offset_wide.melt(id_vars=["objectId"], var_name="band", value_name="offset")
   u_offsets = pd.DataFrame({"objectId": cat_keys["objectId"].values, "band": "u", "offset": 0.0})
   offset_long = pd.concat([offset_long, u_offsets], ignore_index=True)
   offset_long["offset"] = pd.to_numeric(offset_long["offset"], errors="coerce").fillna(0.0)


   band_map = pd.DataFrame({"band": bands_sdss, "filterID": [filters[b] for b in bands_sdss]})


   sdss_df = (
       sdss.loc[sdss["objectId"].isin(cat_keys["objectId"])]
       .merge(band_map, on="filterID", how="inner")
       .loc[:, ["objectId", "band", "mjd", "psMag", "psMagErr_p3"]]
       .rename(columns={"mjd": "time", "psMag": "mag", "psMagErr_p3": "magerr"})
   )
   sdss_df["survey"] = "sdss"


   ps1_df = (
       ps1.merge(cat_keys.loc[:, ["ps1objID", "objectId"]], on="ps1objID", how="inner")
       .merge(band_map, on="filterID", how="inner")
       .loc[:, ["objectId", "band", "obsTime", "psfMag", "psfMagErr_p3"]]
       .rename(columns={"obsTime": "time", "psfMag": "mag", "psfMagErr_p3": "magerr"})
   )
   ps1_df["survey"] = "ps1"


   ztf_df = (
       ztf.merge(cat_keys.loc[:, ["ps1objID", "objectId"]], on="ps1objID", how="inner")
       .merge(band_map, on="filterID", how="inner")
       .loc[:, ["objectId", "band", "mjd", "mag", "magerr_p3"]]
       .rename(columns={"mjd": "time", "magerr_p3": "magerr"})
   )
   ztf_df["survey"] = "ztf"


   ps1_df = ps1_df.merge(offset_long, on=["objectId", "band"], how="left")
   ztf_df = ztf_df.merge(offset_long, on=["objectId", "band"], how="left")
   ps1_df["mag"] = ps1_df["mag"] + ps1_df["offset"].fillna(0.0)
   ztf_df["mag"] = ztf_df["mag"] + ztf_df["offset"].fillna(0.0)
   ps1_df = ps1_df.drop(columns=["offset"])
   ztf_df = ztf_df.drop(columns=["offset"])


   df = pd.concat([sdss_df, ps1_df, ztf_df], ignore_index=True)


   for col in ["time", "mag", "magerr"]:
       df[col] = pd.to_numeric(df[col], errors="coerce")
   df = df.loc[df["time"].notna() & df["mag"].notna() & df["magerr"].notna()]
   df = df.loc[df["band"].isin(bands_sdss)]
   df = df.sort_values(["objectId", "band", "time"], kind="mergesort").reset_index(drop=True)
   return df

def plot_mag_vs_time(df: pd.DataFrame, object_id=None, band=None) -> plt.Axes:
   df_plot = df.copy()
   if object_id is not None:
       df_plot = df_plot.loc[df_plot["objectId"] == object_id]
   if band is not None:
       df_plot = df_plot.loc[df_plot["band"] == band]
   if df_plot.empty:
       raise ValueError("No matching light curve data to plot.")

   fig, ax = plt.subplots(figsize=(10, 6))
   for survey, group in df_plot.groupby("survey"):
       ax.errorbar(
           group["time"],
           group["mag"],
           yerr=group.get("magerr", None),
           fmt="o",
           label=survey,
           alpha=0.8,
           capsize=2,
       )

   ax.invert_yaxis()
   ax.set_xlabel("Time (MJD)")
   ax.set_ylabel("Magnitude")
   title = "Magnitude vs Time"
   if object_id is not None:
       title += f" for object {object_id}"
   if band is not None:
       title += f" ({band}-band)"
   ax.set_title(title)
   ax.legend()
   ax.grid(True, linestyle="--", alpha=0.4)
   plt.tight_layout()
   return ax


if __name__ == "__main__":
   df_total = concat_light_curves_df()
   print(df_total.head(n=10))
   if not df_total.empty:
       sample_obj = df_total["objectId"].iloc[3]
       plot_mag_vs_time(df_total, object_id=sample_obj, band="g")
       plt.show()