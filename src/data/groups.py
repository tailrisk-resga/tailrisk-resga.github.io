"""Characteristic group definitions used by group-importance analyses."""

GROUPS = {
    "ShortTermReversal": [
        'ret_1_0','iskew_capm_21d','iskew_ff3_21d','iskew_hxz4_21d','rskew_21d'
    ],

    "Profitability": [
        'o_score', 'ebit_sale', 'f_score', 'ocf_at', 'ope_be', 'ni_be',
        'ebit_bev', 'niq_be', 'ope_bel1', 'turnover_var_126d', 'dolvol_var_126d'
    ],

    "LowRisk": [
        'betabab_1260d', 'beta_60m', 'betadown_252d', 'beta_dimson_21d', 'seas_6_10na',
        'zero_trades_126d', 'turnover_126d', 'zero_trades_252d', 'zero_trades_21d',
        'ivol_hxz4_21d', 'ivol_ff3_21d', 'ivol_capm_21d', 'ivol_capm_252d', 'rmax5_21d',
        'rmax1_21d', 'rvol_21d', 'ocfq_saleq_std', 'earnings_variability'
    ],

    "Value": [
        'eqnetis_at', 'chcsho_12m', 'netis_at', 'fcf_me', 'eqpo_me', 'div12m_me',
        'eqnpo_me', 'eqnpo_12m', 'bev_mev', 'at_me', 'be_me', 'debt_me', 'eq_dur',
        'ival_me', 'sale_me', 'ebitda_mev', 'ocf_me', 'ni_me'
    ],

    "Investment": [
        'emp_gr1', 'aliq_at', 'be_gr1a', 'at_gr1', 'capx_gr1', 'saleq_gr1', 'sale_gr1', 'col_gr1a',
        'inv_gr1a', 'inv_gr1', 'coa_gr1a', 'nncoa_gr1a', 'ncoa_gr1a', 'lnoa_gr1a', 'noa_gr1a',
        'mispricing_mgmt', 'ppeinv_gr1a', 'capx_gr3', 'capx_gr2', 'sale_gr3', 'seas_2_5na', 'ret_60_12'
    ],

    "Seasonality": [
        'coskew_21d', 'corr_1260d', 'kz_index', 'dbnetis_at', 'lti_gr1a', 'sti_gr1a', 'pi_nix',
        'seas_6_10an', 'seas_11_15an', 'seas_16_20an', 'seas_2_5an', 'seas_11_15na'
    ],

    "DebtIssuance": [
        'noa_at', 'ncol_gr1a', 'capex_abn' ,'ni_ar1' ,'nfna_gr1a' ,'fnl_gr1a' ,'debt_gr3'
    ],

    "Size": [
        'rd_me', 'prc', 'market_equity', 'ami_126d', 'dolvol_126d'
    ],

    "Accruals": [
        'taccruals_at', 'oaccruals_at', 'cowc_gr1a', 'taccruals_ni', 'oaccruals_ni', 'seas_16_20na'
    ],

    "LowLeverage": [
        'netdebt_me', 'cash_at', 'z_score', 'at_be', 'rd5_at', 'rd_sale', 'aliq_mat',
        'tangibility', 'ni_ivol', 'bidaskhl_21d', 'age'
    ],

    "ProfitGrowth": [
        'seas_1_1an', 'ret_12_7', 'dsale_drec', 'tax_gr1a', 'saleq_su', 'niq_be_chg1',
        'niq_at_chg1', 'niq_su', 'ocf_at_chg1', 'dsale_dinv', 'sale_emp_gr1', 'dsale_dsga'
    ],

    "Momentum": [
        'ret_12_1', 'ret_9_1', 'ret_6_1', 'ret_3_1', 'prc_highprc_252d', 'seas_1_1na',
        'resff3_6_1', 'resff3_12_1'
    ],

    "Quality": [
        'qmj_prof', 'niq_at', 'mispricing_perf', 'op_atl1', 'op_at', 'cop_atl1', 'cop_at',
        'qmj_growth', 'qmj', 'ni_inc8q', 'dgp_dsale', 'qmj_safety', 'opex_at', 'at_turnover',
        'sale_bev', 'gp_atl1', 'gp_at'
    ],
}

__all__ = ["GROUPS"]
