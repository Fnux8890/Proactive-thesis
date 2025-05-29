/// Optimal signal groups based on empirical coverage analysis
pub struct OptimalSignals {
    /// Primary signals with highest coverage and reliability
    primary: Vec<&'static str>,
    /// Secondary signals with moderate coverage
    secondary: Vec<&'static str>,
}

impl OptimalSignals {
    pub fn new() -> Self {
        Self {
            primary: vec![
                "dli_sum",                // 100% - Primary light metric
                "radiation_w_m2",         // 4.2% - Solar radiation
                "outside_temp_c",         // 3.8% - External temperature
                "co2_measured_ppm",       // 3.8% - Growth indicator
                "air_temp_middle_c",      // 3.5% - Internal climate
                "air_temp_c",             // Alternative temperature signal
                "relative_humidity_percent", // Humidity control
                "light_intensity_umol",   // Light intensity
            ],
            secondary: vec![
                "pipe_temp_1_c",          // 3.5% - Heating system
                "curtain_1_percent",      // 3.7% - Light control
                "humidity_deficit_g_m3",  // 3.5% - Humidity control
                "heating_setpoint_c",     // Heating control
                "vpd_hpa",                // Vapor pressure deficit
                "total_lamps_on",         // Artificial lighting status
            ],
        }
    }

    pub fn get_all(&self) -> Vec<&str> {
        let mut all = self.primary.clone();
        all.extend(self.secondary.iter());
        all
    }

    #[allow(dead_code)]
    pub fn get_primary(&self) -> &[&str] {
        &self.primary
    }

    #[allow(dead_code)]
    pub fn get_secondary(&self) -> &[&str] {
        &self.secondary
    }
}