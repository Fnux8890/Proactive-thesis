-- Create phenotypes view/table for model builder compatibility
-- This script creates a 'phenotypes' view that references the literature_kalanchoe_phenotypes table

-- First, check if the source table exists
DO $$
BEGIN
    -- Check if literature_kalanchoe_phenotypes exists
    IF EXISTS (
        SELECT 1 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'literature_kalanchoe_phenotypes'
    ) THEN
        -- Drop existing view if it exists
        DROP VIEW IF EXISTS public.phenotypes CASCADE;
        
        -- Create view with simplified column names for model compatibility
        CREATE OR REPLACE VIEW public.phenotypes AS
        SELECT 
            entry_id,
            publication_source,
            species,
            cultivar_or_line,
            experiment_description,
            phenotype_name,
            phenotype_value,
            phenotype_unit,
            measurement_condition_notes,
            control_group_identifier,
            environment_temp_day_C AS temp_day_c,
            environment_temp_night_C AS temp_night_c,
            environment_photoperiod_h AS photoperiod_h,
            environment_DLI_mol_m2_d AS dli_mol_m2_d,
            environment_light_intensity_umol_m2_s AS light_intensity_umol_m2_s,
            environment_light_quality_description AS light_quality,
            environment_CO2_ppm AS co2_ppm,
            environment_RH_percent AS rh_percent,
            environment_stressor_type AS stressor_type,
            environment_stressor_level AS stressor_level,
            data_extraction_date,
            additional_notes,
            -- Add computed columns that might be useful
            COALESCE(phenotype_value, 0) AS phenotype_value_filled,
            CASE 
                WHEN phenotype_name LIKE '%growth%' THEN 'growth'
                WHEN phenotype_name LIKE '%height%' THEN 'morphology'
                WHEN phenotype_name LIKE '%flower%' THEN 'flowering'
                WHEN phenotype_name LIKE '%water%' THEN 'water_use'
                ELSE 'other'
            END AS phenotype_category
        FROM public.literature_kalanchoe_phenotypes;
        
        RAISE NOTICE 'Created phenotypes view successfully';
    ELSE
        -- If source table doesn't exist, create a dummy phenotypes table with synthetic data
        CREATE TABLE IF NOT EXISTS public.phenotypes (
            entry_id SERIAL PRIMARY KEY,
            publication_source VARCHAR(255) DEFAULT 'synthetic',
            species VARCHAR(255) DEFAULT 'Kalanchoe blossfeldiana',
            cultivar_or_line VARCHAR(255) DEFAULT 'test_cultivar',
            experiment_description TEXT DEFAULT 'synthetic_data_for_testing',
            phenotype_name VARCHAR(255) NOT NULL,
            phenotype_value REAL,
            phenotype_unit VARCHAR(50),
            measurement_condition_notes TEXT,
            control_group_identifier VARCHAR(255) DEFAULT 'control',
            temp_day_c REAL DEFAULT 22.0,
            temp_night_c REAL DEFAULT 18.0,
            photoperiod_h REAL DEFAULT 12.0,
            dli_mol_m2_d REAL DEFAULT 15.0,
            light_intensity_umol_m2_s REAL DEFAULT 150.0,
            light_quality TEXT DEFAULT 'LED',
            co2_ppm REAL DEFAULT 400.0,
            rh_percent REAL DEFAULT 70.0,
            stressor_type VARCHAR(100),
            stressor_level VARCHAR(100),
            data_extraction_date TIMESTAMPTZ DEFAULT NOW(),
            additional_notes TEXT,
            phenotype_value_filled REAL GENERATED ALWAYS AS (COALESCE(phenotype_value, 0)) STORED,
            phenotype_category VARCHAR(50) GENERATED ALWAYS AS (
                CASE 
                    WHEN phenotype_name LIKE '%growth%' THEN 'growth'
                    WHEN phenotype_name LIKE '%height%' THEN 'morphology'
                    WHEN phenotype_name LIKE '%flower%' THEN 'flowering'
                    WHEN phenotype_name LIKE '%water%' THEN 'water_use'
                    ELSE 'other'
                END
            ) STORED
        );
        
        -- Insert some synthetic phenotype data for testing
        INSERT INTO public.phenotypes (phenotype_name, phenotype_value, phenotype_unit)
        VALUES 
            ('plant_height', 25.0, 'cm'),
            ('growth_rate', 0.5, 'cm/day'),
            ('leaf_area', 150.0, 'cm2'),
            ('stem_diameter', 8.0, 'mm'),
            ('time_to_flower', 90.0, 'days'),
            ('flower_count', 12.0, 'count'),
            ('water_use_efficiency', 3.5, 'g/L'),
            ('chlorophyll_content', 45.0, 'SPAD'),
            ('biomass_fresh', 120.0, 'g'),
            ('biomass_dry', 15.0, 'g')
        ON CONFLICT DO NOTHING;
        
        RAISE NOTICE 'Created phenotypes table with synthetic data';
    END IF;
END $$;