import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Astropy imports
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

# JAX & ezTaoX imports for DRW fitting
import jax
import jax.numpy as jnp
# Always recommended for precision with EzTaoX
jax.config.update("jax_enable_x64", True) 

import numpyro
import numpyro.distributions as dist
from eztaox.kernels.quasisep import Exp
from eztaox.models import MultiVarModel
from eztaox.ts_utils import formatlc
from eztaox.fitter import random_search
import tinygp

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


# -----------------------------
# MULTIBAND DRW FITTING FUNCTION
# -----------------------------

def fit_multiband_drw_eztaox(
    df: pd.DataFrame, 
    object_id, 
    n_sample=20000, 
    n_best=5, 
    seed=1, 
    has_lag=True, 
    zero_mean=False
):
    """
    Fits a Damped Random Walk (DRW) model simultaneously across all available bands 
    for a given object using ezTaoX MultiVarModel.
    """
    sub_df = df.loc[df["objectId"] == object_id].sort_values(["band", "time"])
    
    # Extract available bands with sufficient data points for this object
    available_bands = []
    ts, ys, yerrs = {}, {}, {}
    
    for band in bands_sdss:
        band_df = sub_df.loc[sub_df["band"] == band]
        if len(band_df) >= 3:
            available_bands.append(band)
            ts[band] = jnp.array(band_df["time"].values)
            ys[band] = jnp.array(band_df["mag"].values)
            yerrs[band] = jnp.array(band_df["magerr"].values)
            
    if len(available_bands) < 2:
        raise ValueError(f"Object {object_id} needs at least 2 bands with data for multiband fitting (found {len(available_bands)}).")

    # Map bands to indices (first band acts as the reference band index 0)
    band_index = {band: i for i, band in enumerate(available_bands)}
    n_band = len(available_bands)

    # Format multi-band light curves using eztaox.ts_utils.formatlc
    X, y, yerr = formatlc(ts, ys, yerrs, band_index)

    # Initialize model kernel and MultiVarModel instance
    k = Exp(scale=100.0, sigma=1.0)
    model = MultiVarModel(X, y, yerr, k, n_band, has_lag=has_lag, zero_mean=zero_mean)

    ymin, ymax = float(y.min()), float(y.max())

    # Define the initialization/prior sampler for multiband random search
    def init_sampler():
        log_drw_scale = numpyro.sample("log_drw_scale", dist.Uniform(jnp.log(10.0), jnp.log(200.0)))
        log_drw_sigma = numpyro.sample("log_drw_sigma", dist.Uniform(jnp.log(0.01), jnp.log(10)))
        log_kernel_param = jnp.stack([log_drw_scale, log_drw_sigma])
        numpyro.deterministic("log_kernel_param", log_kernel_param)
        
        sample_params = {"log_kernel_param": log_kernel_param}
        
        if n_band > 1:
            log_amp_scale = numpyro.sample(
                "log_amp_scale", 
                dist.Uniform(-2.0, 2.0).expand([n_band - 1])
            )
            sample_params["log_amp_scale"] = log_amp_scale
            
            if has_lag:
                lag = numpyro.sample("lag", dist.Uniform(-10.0, 10.0).expand([n_band - 1]))
                sample_params["lag"] = lag
        
        if not zero_mean:
            mean = numpyro.sample(
                "mean", 
                dist.Uniform(low=ymin, high=ymax).expand([n_band])
            )
            sample_params["mean"] = mean
            
        return sample_params

    fit_key = jax.random.PRNGKey(seed)
    best_params, log_likelihoods = random_search(model, init_sampler, fit_key, n_sample=n_sample, n_best=n_best)
    
    log_scale_best = best_params["log_kernel_param"][0]
    log_sigma_best = best_params["log_kernel_param"][1]
    
    results = {
        "objectId": object_id,
        "bands": available_bands,
        "band_index": band_index,
        "log_drw_scale": float(log_scale_best),
        "log_drw_sigma": float(log_sigma_best),
        "tau_drw": float(np.exp(log_scale_best)),
        "sigma_drw": float(np.exp(log_sigma_best)),
        "best_params": best_params,
        "log_likelihoods": log_likelihoods
    }
    
    return results


if __name__ == "__main__":
    df_total = concat_light_curves_df()
    print(df_total.head(n=10))
    
    if not df_total.empty:
        sample_obj = df_total["objectId"].iloc[3]
        
        # Run multiband EzTaoX DRW fit on the sample object
        print(f"\nFitting Multiband DRW model for object {sample_obj}")
        fit_res = fit_multiband_drw_eztaox(df_total, object_id=sample_obj)
        
        print("Multiband Fit Results:")
        print(f"  -> Active Bands Fitted: {fit_res['bands']}")
        print(f"  -> Natural Log Scale (tau): {fit_res['log_drw_scale']:.4f}")
        print(f"  -> Natural Log Sigma: {fit_res['log_drw_sigma']:.4f}")
        print(f"  -> Damping Timescale (tau_DRW): {fit_res['tau_drw']:.2f} days")
        print(f"  -> Variability Amplitude (sigma_DRW): {fit_res['sigma_drw']:.4f}")
        if "lag" in fit_res["best_params"]:
            print(f"  -> Best-fit Lags (relative to reference band): {fit_res['best_params']['lag']}")