┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃    ┃ Name                                  ┃ Type                 ┃ Params ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 0  │ detail_network                        │ DetailNetwork        │  6.0 M │
│ 1  │ detail_network.fpn                    │ FPN                  │  6.0 M │
│ 2  │ detail_network.fpn.input_adapters     │ ModuleList           │  100 K │
│ 3  │ detail_network.fpn.bottom_up_layers   │ ModuleList           │  5.9 M │
│ 4  │ detail_network.fpn.toplayer           │ Conv2d               │  6.2 K │
│ 5  │ detail_network.fpn.lateral_layers     │ ModuleList           │  4.6 K │
│ 6  │ detail_network.fpn.smooth_layers      │ ModuleList           │    660 │
│ 7  │ detail_network.temporal_mlp_before    │ Sequential           │  1.7 K │
│ 8  │ detail_network.temporal_mlp_before.0  │ Linear               │    896 │
│ 9  │ detail_network.temporal_mlp_before.1  │ GELU                 │      0 │
│ 10 │ detail_network.temporal_mlp_before.2  │ Dropout              │      0 │
│ 11 │ detail_network.temporal_mlp_before.3  │ Linear               │    774 │
│ 12 │ detail_network.temporal_mlp_after     │ Sequential           │  1.7 K │
│ 13 │ detail_network.temporal_mlp_after.0   │ Linear               │    896 │
│ 14 │ detail_network.temporal_mlp_after.1   │ GELU                 │      0 │
│ 15 │ detail_network.temporal_mlp_after.2   │ Dropout              │      0 │
│ 16 │ detail_network.temporal_mlp_after.3   │ Linear               │    774 │
│ 17 │ detail_network.idr                    │ ModuleList           │ 21.3 K │
│ 18 │ detail_network.idr.0                  │ Sequential           │  7.1 K │
│ 19 │ detail_network.idr.1                  │ Sequential           │  7.1 K │
│ 20 │ detail_network.idr.2                  │ Sequential           │  7.1 K │
│ 21 │ approx_network                        │ ApproximationNetwork │ 15.4 M │
│ 22 │ approx_network.encoder                │ Conv2d               │ 28.2 K │
│ 23 │ approx_network.temporal_injector      │ Sequential           │  102 K │
│ 24 │ approx_network.temporal_injector.0    │ Conv3d               │  3.6 K │
│ 25 │ approx_network.temporal_injector.1    │ GELU                 │      0 │
│ 26 │ approx_network.temporal_injector.2    │ Conv3d               │ 98.4 K │
│ 27 │ approx_network.fusion                 │ Conv2d               │  328 K │
│ 28 │ approx_network.norm_before            │ GroupNorm            │  1.0 K │
│ 29 │ approx_network.mixer_layers           │ Sequential           │ 15.0 M │
│ 30 │ approx_network.mixer_layers.0         │ SpatioTemporalBlock  │  5.0 M │
│ 31 │ approx_network.mixer_layers.1         │ SpatioTemporalBlock  │  5.0 M │
│ 32 │ approx_network.mixer_layers.2         │ SpatioTemporalBlock  │  5.0 M │
│ 33 │ approx_network.norm_after             │ GroupNorm            │  1.0 K │
│ 34 │ approx_network.decoder                │ Conv2d               │  3.1 K │
│ 35 │ refine_mixer                          │ RefineMixer          │ 21.8 M │
│ 36 │ refine_mixer.ad_interaction           │ ResnetBlock          │  3.1 M │
│ 37 │ refine_mixer.ad_interaction.block1    │ Block                │ 63.9 K │
│ 38 │ refine_mixer.ad_interaction.block2    │ Block                │  3.0 M │
│ 39 │ refine_mixer.ad_interaction.res_conv  │ Conv2d               │  7.5 K │
│ 40 │ refine_mixer.ad_interaction.dropout   │ Dropout              │      0 │
│ 41 │ refine_mixer.ad_norm                  │ GroupNorm            │  1.2 K │
│ 42 │ refine_mixer.drift_corrector          │ ResnetBlock          │  3.1 M │
│ 43 │ refine_mixer.drift_corrector.block1   │ Block                │ 63.9 K │
│ 44 │ refine_mixer.drift_corrector.block2   │ Block                │  3.0 M │
│ 45 │ refine_mixer.drift_corrector.res_conv │ Conv2d               │  7.5 K │
│ 46 │ refine_mixer.drift_corrector.dropout  │ Dropout              │      0 │
│ 47 │ refine_mixer.drift_norm               │ GroupNorm            │  1.2 K │
│ 48 │ refine_mixer.mixer                    │ Sequential           │ 15.7 M │
│ 49 │ refine_mixer.mixer.0                  │ GroupNorm            │  2.3 K │
│ 50 │ refine_mixer.mixer.1                  │ ResnetBlock          │  9.7 M │
│ 51 │ refine_mixer.mixer.2                  │ GroupNorm            │  1.2 K │
│ 52 │ refine_mixer.mixer.3                  │ ResnetBlock          │  6.0 M │
│ 53 │ refine_mixer.out_conv                 │ Conv2d               │  3.5 K │
└────┴───────────────────────────────────────┴──────────────────────┴────────┘
Trainable params: 43.2 M                                                                                                                                                                                                     
Non-trainable params: 0                                                                                                                                                                                                      
Total params: 43.2 M                                                                                                                                                                                                         
Total estimated model params size (MB): 172  

