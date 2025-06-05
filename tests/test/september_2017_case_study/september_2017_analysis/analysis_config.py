# September 6, 2017 X9.3 Flare Analysis Configuration
# Generated on 2025-06-04 05:54:30.990379

CONFIG = {'event_name': 'September_6_2017_X9.3', 'flare_class': 'X9.3', 'peak_time': '2017-09-06T12:02:00Z', 'active_region': 'AR_2673', 'model_config': {'name': 'EVEREST', 'weights': 'tests/model_weights_EVEREST_72h_M5.pt', 'prediction_horizon': '72h', 'features': ['TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTBSQ', 'R_VALUE', 'TOTPOT', 'SAVNCPP', 'AREA_ACR', 'ABSNJZH']}, 'analysis_window': {'pre_flare_hours': 72, 'post_flare_hours': 24, 'total_hours': 96}, 'expected_performance': {'note': 'X9.3 is much stronger than X1.4 (July 2012)', 'hypothesis': 'Should achieve higher probabilities than 15.76%', 'population_context': 'Closer to 46% optimal threshold'}}
