# Optimal Signal Selection for Era Detection

## Data Coverage Analysis (1,457,281 total rows)

### High Coverage Signals (>95% data)
- **dli_sum**: 100% coverage (1,457,281/1,457,281) ✅
- **time**: 100% coverage (1,457,281/1,457,281) ✅

### Good Coverage Signals (>50% data)
- **radiation_w_m2**: 4.2% coverage (61,661/1,457,281)
- **outside_temp_c**: 3.8% coverage (55,907/1,457,281)
- **co2_measured_ppm**: 3.8% coverage (55,921/1,457,281)
- **curtain_1_percent**: 3.7% coverage (53,752/1,457,281)
- **air_temp_middle_c**: 3.5% coverage (51,083/1,457,281)
- **pipe_temp_1_c**: 3.5% coverage (51,083/1,457,281)
- **pipe_temp_2_c**: 3.5% coverage (51,083/1,457,281)
- **humidity_deficit_g_m3**: 3.5% coverage (51,083/1,457,281)

### Medium Coverage Signals (>1.5% data)
- **lamp_grp1_no3_status**: 1.8% coverage (26,171/1,457,281)
- **lamp_grp1_no4_status**: 1.8% coverage (26,171/1,457,281)
- **lamp_grp2_no3_status**: 1.8% coverage (26,171/1,457,281)
- **lamp_grp2_no4_status**: 1.8% coverage (26,171/1,457,281)
- **lamp_grp3_no3_status**: 1.8% coverage (26,171/1,457,281)
- **lamp_grp4_no3_status**: 1.8% coverage (26,171/1,457,281)
- **vent_lee_afd3_percent**: 1.7% coverage (25,100/1,457,281)
- **vent_wind_afd3_percent**: 1.7% coverage (25,100/1,457,281)
- **air_temp_c**: 0.87% coverage (12,677/1,457,281)

## Recommended Signal Selection Strategy

### Core Signals for Era Detection (Hybrid Table Columns)
```rust
// High-impact signals with reasonable coverage
let core_signals = vec![
    "dli_sum",                    // 100% - Primary light metric
    "radiation_w_m2",             // 4.2% - Solar radiation
    "outside_temp_c",             // 3.8% - External conditions
    "co2_measured_ppm",           // 3.8% - Growth indicator
    "air_temp_middle_c",          // 3.5% - Internal climate
    "pipe_temp_1_c",              // 3.5% - Heating system
    "curtain_1_percent",          // 3.7% - Light control
    "humidity_deficit_g_m3",      // 3.5% - Humidity control
];
```

### Extended Signals (JSONB)
```rust
// Lower coverage but still valuable for specific eras
let extended_signals = vec![
    "lamp_grp1_no3_status",       // 1.8% - Artificial lighting
    "lamp_grp2_no3_status",       // 1.8% - Artificial lighting
    "vent_lee_afd3_percent",      // 1.7% - Ventilation
    "vent_wind_afd3_percent",     // 1.7% - Ventilation
    "air_temp_c",                 // 0.87% - Basic temp
];
```

## Multi-Signal Era Detection Strategy

### Level A (PELT): Primary Signals
Use the most reliable signals for initial segmentation:
```bash
--signal-cols dli_sum,radiation_w_m2,outside_temp_c
```

### Level B (BOCPD): Secondary Signals  
Add environmental controls:
```bash
--signal-cols dli_sum,radiation_w_m2,outside_temp_c,air_temp_middle_c,co2_measured_ppm
```

### Level C (HMM): All Viable Signals
Include control systems for fine-grained detection:
```bash
--signal-cols dli_sum,radiation_w_m2,outside_temp_c,air_temp_middle_c,co2_measured_ppm,curtain_1_percent,pipe_temp_1_c,humidity_deficit_g_m3
```

## Expected Benefits

### Era Quality Improvements
- **Seasonal Detection**: `outside_temp_c` + `radiation_w_m2` for natural seasons
- **Growth Phases**: `co2_measured_ppm` + `humidity_deficit_g_m3` for plant growth stages  
- **Operational Changes**: `curtain_1_percent` + `pipe_temp_1_c` for facility adjustments
- **Light Cycles**: `dli_sum` for daily/weekly patterns

### Performance with Hybrid Storage
- **Query Speed**: 5-10x faster for multi-signal queries
- **Memory Usage**: Lower due to native column access
- **Parallel Processing**: Can process multiple signals simultaneously

## Implementation Priority

1. **High Priority**: Core environmental signals (temp, light, CO2)
2. **Medium Priority**: Control system signals (curtains, heating)
3. **Low Priority**: Sparse signals (specific lamp groups)

This strategy maximizes era detection quality while maintaining computational efficiency.