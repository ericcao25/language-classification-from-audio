import json
from datasets import load_dataset

REGIONS = {
    "western_europe": [
        "ast_es", "bs_ba", "ca_es", "hr_hr", "da_dk", "nl_nl", "en_us",
        "fi_fi", "fr_fr", "gl_es", "de_de", "el_gr", "hu_hu", "is_is",
        "ga_ie", "it_it", "kea_cv", "lb_lu", "mt_mt", "nb_no", "oc_fr",
        "pt_br", "es_419", "sv_se", "cy_gb",
    ],
    "eastern_europe": [
        "hy_am", "be_by", "bg_bg", "cs_cz", "et_ee", "ka_ge", "lv_lv",
        "lt_lt", "mk_mk", "pl_pl", "ro_ro", "ru_ru", "sr_rs", "sk_sk",
        "sl_si", "uk_ua",
    ],
    "central_asia_middle_east_north_africa": [
        "ar_eg", "az_az", "he_il", "kk_kz", "ky_kg", "mn_mn", "ps_af",
        "fa_ir", "ckb_iq", "tg_tj", "tr_tr", "uz_uz",
    ],
    "sub_saharan_africa": [
        "af_za", "am_et", "ff_sn", "lg_ug", "ha_ng", "ig_ng", "kam_ke",
        "ln_cd", "luo_ke", "nso_za", "ny_mw", "om_et", "sn_zw", "so_so",
        "sw_ke", "umb_ao", "wo_sn", "xh_za", "yo_ng", "zu_za",
    ],
    "south_asia": [
        "as_in", "bn_in", "gu_in", "hi_in", "kn_in", "ml_in", "mr_in",
        "ne_np", "or_in", "pa_in", "sd_in", "ta_in", "te_in", "ur_pk",
    ],
    "south_east_asia": [
        "my_mm", "ceb_ph", "fil_ph", "id_id", "jv_id", "km_kh", "lo_la",
        "ms_my", "mi_nz", "th_th", "vi_vn",
    ],
    "cjk": [
        "yue_hant_hk", "cmn_hans_cn", "ja_jp", "ko_kr",
    ],
}


def build_region_json(output_path="fleurs_regions.json"):
    schema_ds = load_dataset("google/fleurs", "all", split="train", streaming=True)
    config_names = schema_ds.features["lang_id"].names
    config_to_id = {name: idx for idx, name in enumerate(config_names)}
    result = {}
    all_ok = True
    for region, configs in REGIONS.items():
        lang_ids = []
        for cfg in configs:
            if cfg not in config_to_id:
                print(f"WARNING: '{cfg}' (region '{region}') not found in "
                      f"FLEURS config names. Check spelling.")
                all_ok = False
                lang_ids.append(-1)
            else:
                lang_ids.append(config_to_id[cfg])

        result[region] = {
            "configs": configs,
            "lang_ids": lang_ids,
        }

    if not all_ok:
        print("JSON not written — fix the warnings above first.")
        return result

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {len(result)} regions to '{output_path}'")
    return result

def load_configs(region, json_path="fleurs_regions.json"):
    if region not in REGIONS:
        raise ValueError(f"Region '{region}' not found.")
    with open(json_path, "r") as f:
        return json.load(f)[region]


if __name__ == "__main__":
    build_region_json()