use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;
use std::collections::HashMap;

pub async fn create_features_table_if_not_exists(
    pool: &PgPool,
    table_name: &str,
) -> Result<()> {
    // Create table
    let create_table_query = format!(
        r#"
        CREATE TABLE IF NOT EXISTS {} (
            id BIGSERIAL PRIMARY KEY,
            era_id INTEGER NOT NULL,
            era_level TEXT NOT NULL,
            features JSONB NOT NULL,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            
            -- Create unique constraint for ON CONFLICT
            CONSTRAINT {}_unique_era UNIQUE (era_id, era_level)
        )"#,
        table_name, table_name
    );
    
    sqlx::query(&create_table_query).execute(pool).await?;
    
    // Create indexes separately
    let indexes = vec![
        format!("CREATE INDEX IF NOT EXISTS idx_{}_era_id ON {} (era_id)", table_name, table_name),
        format!("CREATE INDEX IF NOT EXISTS idx_{}_era_level ON {} (era_level)", table_name, table_name),
        format!("CREATE INDEX IF NOT EXISTS idx_{}_computed_at ON {} (computed_at DESC)", table_name, table_name),
    ];
    
    for index_query in indexes {
        sqlx::query(&index_query).execute(pool).await?;
    }
    
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Era {
    pub era_id: i32,
    pub era_level: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub row_count: i32,
}

#[derive(Debug, Clone)]
pub struct EraData {
    pub era: Era,
    #[allow(dead_code)]
    pub timestamps: Vec<DateTime<Utc>>,
    pub sensor_data: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    pub era_id: i32,
    pub era_level: String,
    pub features: HashMap<String, f64>,
    pub computed_at: DateTime<Utc>,
}

#[allow(dead_code)]
pub trait FeatureWriter {
    async fn write(&self, pool: &PgPool, table_name: &str) -> Result<()>;
}

#[allow(dead_code)]
impl FeatureWriter for Vec<FeatureSet> {
    async fn write(&self, pool: &PgPool, table_name: &str) -> Result<()> {
        write_features(pool, table_name, self.clone()).await
    }
}

pub async fn fetch_eras(
    pool: &PgPool,
    era_level: &str,
    min_rows: usize,
) -> Result<Vec<Era>> {
    // Table name follows pattern: era_labels_level_{a|b|c}
    let table_name = format!("era_labels_level_{}", era_level.to_lowercase());
    
    let query = format!(r#"
        SELECT 
            DISTINCT era_id,
            'level_{}' as era_level,
            start_time,
            end_time,
            rows as row_count
        FROM {}
        WHERE rows >= $1
        ORDER BY start_time
    "#, era_level.to_lowercase(), table_name);
    
    let eras = sqlx::query_as::<_, Era>(&query)
        .bind(min_rows as i32)
        .fetch_all(pool)
        .await?;
    
    Ok(eras)
}

pub async fn fetch_era_data(
    pool: &PgPool,
    era: &Era,
) -> Result<EraData> {
    let query = r#"
        SELECT 
            time as timestamp,
            air_temp_c,
            relative_humidity_percent,
            co2_measured_ppm,
            light_intensity_umol,
            radiation_w_m2,
            vpd_hpa,
            dli_sum,
            pipe_temp_1_c,
            flow_temp_1_c,
            vent_pos_1_percent,
            vent_pos_2_percent,
            curtain_1_percent
        FROM sensor_data_merged
        WHERE time >= $1 AND time < $2
          -- Sample: take one reading every 5 minutes for long eras
          AND (($2 - $1 < interval '7 days') OR EXTRACT(EPOCH FROM time) % 300 = 0)
        ORDER BY time
        LIMIT 100000  -- Safety limit
    "#;
    
    #[derive(sqlx::FromRow)]
    struct SensorRow {
        timestamp: DateTime<Utc>,
        air_temp_c: Option<f64>,
        relative_humidity_percent: Option<f64>,
        co2_measured_ppm: Option<f64>,
        light_intensity_umol: Option<f64>,
        radiation_w_m2: Option<f64>,
        vpd_hpa: Option<f64>,
        dli_sum: Option<f64>,
        pipe_temp_1_c: Option<f64>,
        flow_temp_1_c: Option<f64>,
        vent_pos_1_percent: Option<f64>,
        vent_pos_2_percent: Option<f64>,
        curtain_1_percent: Option<f64>,
    }
    
    let rows = sqlx::query_as::<_, SensorRow>(query)
        .bind(era.start_time)
        .bind(era.end_time)
        .fetch_all(pool)
        .await?;
    
    let mut timestamps = Vec::with_capacity(rows.len());
    let mut sensor_data: HashMap<String, Vec<f32>> = HashMap::new();
    
    // Initialize vectors for each sensor
    let sensors = vec![
        "air_temp_c",
        "relative_humidity_percent",
        "co2_measured_ppm",
        "light_intensity_umol",
        "radiation_w_m2",
        "vpd_hpa",
        "dli_sum",
        "pipe_temp_1_c",
        "flow_temp_1_c",
        "vent_pos_1_percent",
        "vent_pos_2_percent",
        "curtain_1_percent",
    ];
    
    for sensor in &sensors {
        sensor_data.insert(sensor.to_string(), Vec::with_capacity(rows.len()));
    }
    
    // Fill data
    for row in rows {
        timestamps.push(row.timestamp);
        
        sensor_data.get_mut("air_temp_c").unwrap()
            .push(row.air_temp_c.unwrap_or(0.0) as f32);
        sensor_data.get_mut("relative_humidity_percent").unwrap()
            .push(row.relative_humidity_percent.unwrap_or(0.0) as f32);
        sensor_data.get_mut("co2_measured_ppm").unwrap()
            .push(row.co2_measured_ppm.unwrap_or(0.0) as f32);
        sensor_data.get_mut("light_intensity_umol").unwrap()
            .push(row.light_intensity_umol.unwrap_or(0.0) as f32);
        sensor_data.get_mut("radiation_w_m2").unwrap()
            .push(row.radiation_w_m2.unwrap_or(0.0) as f32);
        sensor_data.get_mut("vpd_hpa").unwrap()
            .push(row.vpd_hpa.unwrap_or(0.0) as f32);
        sensor_data.get_mut("dli_sum").unwrap()
            .push(row.dli_sum.unwrap_or(0.0) as f32);
        sensor_data.get_mut("pipe_temp_1_c").unwrap()
            .push(row.pipe_temp_1_c.unwrap_or(0.0) as f32);
        sensor_data.get_mut("flow_temp_1_c").unwrap()
            .push(row.flow_temp_1_c.unwrap_or(0.0) as f32);
        sensor_data.get_mut("vent_pos_1_percent").unwrap()
            .push(row.vent_pos_1_percent.unwrap_or(0.0) as f32);
        sensor_data.get_mut("vent_pos_2_percent").unwrap()
            .push(row.vent_pos_2_percent.unwrap_or(0.0) as f32);
        sensor_data.get_mut("curtain_1_percent").unwrap()
            .push(row.curtain_1_percent.unwrap_or(0.0) as f32);
    }
    
    Ok(EraData {
        era: era.clone(),
        timestamps,
        sensor_data,
    })
}

pub async fn write_features(
    pool: &PgPool,
    table_name: &str,
    features: Vec<FeatureSet>,
) -> Result<()> {
    // Use a transaction for batch writes
    let mut tx = pool.begin().await?;
    
    for feature_set in features {
        let features_json = serde_json::to_value(&feature_set.features)?;
        
        let query = format!(
            r#"
            INSERT INTO {} (era_id, era_level, features, computed_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (era_id, era_level) 
            DO UPDATE SET 
                features = EXCLUDED.features,
                computed_at = EXCLUDED.computed_at
            "#,
            table_name
        );
        
        sqlx::query(&query)
            .bind(feature_set.era_id)
            .bind(&feature_set.era_level)
            .bind(features_json)
            .bind(feature_set.computed_at)
            .execute(&mut *tx)
            .await?;
    }
    
    // Commit all writes at once
    tx.commit().await?;
    
    Ok(())
}