import string

labels = [
    "gps_SASSA_SARAL_2018",
    "gps_SASSA_Sentinel_2018",
    "argos_SASSA_SARAL_2018",
    "argos_SASSA_Sentinel_2018",
    "argos_SASSA_Sentinel_2016",
    "argos_PEACHI_Sentinel_2018",
    "gps_PEACHI_Sentinel_2018",
]

zarr_dir = "/home1/datawork/mdemol/m2"

lon_180_to_360 = lambda lon: lon % 360
lon_360_to_180 = lambda lon: (lon + 180) % 360 - 180

lettres = ["(" + l + ")" for l in list(string.ascii_lowercase)]
