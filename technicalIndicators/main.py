import technicalIndicators.parabolicSar.main as sar
import technicalIndicators.aroonOscilators.main as ao
import technicalIndicators.onBalanceVolume.main as obv
import technicalIndicators.relativeStrengthIndex.main as rsi
import technicalIndicators.averageDirectionalIndex.main as adx
import technicalIndicators.accumulationDistribution.main as ad
import technicalIndicators.movingAverageConvergenceDivergence.main as macd

def generate_all_technical_indicators(
        data, 
        acceleration_factor=0.02, 
        adx_rolling_window=14, 
        ad_1_rolling_window=5, 
        ad_1_threshold=0.1, 
        ad_2_rolling_window=10, 
        ad_2_threshold=0.1
    ):

    original_columns = set(data.columns)

    data = sar.identify_parabolic_sar_indicators(data, acceleration_factor)
    data = ao.identify_ao_indicators(data)
    data = obv.identify_obv_indicators(data)
    data = rsi.identify_rsi_indicators(data)
    data = adx.identify_adx_indicators(data, adx_rolling_window)
    data = ad.identify_ad_indicators(data, ad_1_rolling_window, ad_1_threshold)
    data = ad.identify_ad_indicators(data, ad_2_rolling_window, ad_2_threshold)
    data = macd.identify_macd_indicators(data)

    updated_columns = set(data.columns)
    feature_columns = list(updated_columns - original_columns)
    with open('modelDevelopment/technical_indicator_features.txt', "w") as file:
        for fea_col in feature_columns:
            file.write(fea_col + "\n")

    return data